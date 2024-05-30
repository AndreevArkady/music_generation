import csv
import json
import os
from shutil import copyfile

import torch
import torch.nn as nn
import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from dataset.e_piano import create_epiano_datasets
from model.music_transformer import MusicTransformer
from utilities.constants import *
from utilities.device import get_device, use_cuda
from utilities.lr_scheduling import LrStepTracker, get_lr
from utilities.run_model import eval_model, parse_json, train_epoch

# Baseline is an untrained epoch that we evaluate as a baseline loss and accuracy
BASELINE_EPOCH = -1


def main():
    """Author: Damon Gwinn.

    Entry point. Trains a model specified by command line arguments.
    """
    model_params, args = parse_json(file_type='train')

    if args['seed']:
        torch.manual_seed(args['seed'])

    if args['force_cpu']:
        use_cuda(False)
        print('WARNING: Forced CPU usage, expect model to perform slower')
        print()

    os.makedirs(args['output_dir'], exist_ok=True)

    # Output prep
    weights_folder = os.path.join(args['output_dir'], 'weights')
    os.makedirs(weights_folder, exist_ok=True)

    results_folder = os.path.join(args['output_dir'], 'results')
    os.makedirs(results_folder, exist_ok=True)

    results_file = os.path.join(results_folder, 'results.csv')
    best_loss_file = os.path.join(results_folder, 'best_loss_weights.pickle')
    best_acc_file = os.path.join(results_folder, 'best_acc_weights.pickle')
    best_text = os.path.join(results_folder, 'best_epochs.txt')

    # Datasets
    with open('dataset/dataset_parameters.json', 'r') as dataset_params_file:
        dataset_params = json.load(dataset_params_file)

    train_dataset, val_dataset, test_dataset = create_epiano_datasets(
        max_seq=model_params['max_sequence'],
        **dataset_params,
    )
    copyfile(
        'dataset/additional_features_state.json',
        f"{args['output_dir']}/additional_features_state.json",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args['batch_size'],
        num_workers=args['n_workers'],
        shuffle=True,
    )
    val_loader = DataLoader(  # noqa: F841
        val_dataset,
        batch_size=args['batch_size'],
        num_workers=args['n_workers'],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args['batch_size'],
        num_workers=args['n_workers'],
    )

    wandb_config = {
        'epochs': args['epochs'],
        'batch_size': args['batch_size'],
        'n_layers': model_params['n_layers'],
        'num_heads': model_params['num_heads'],
        'd_model': model_params['d_model'],
        'dim_feedforward': model_params['dim_feedforward'],
        'dropout': model_params['dropout'],
        'max_sequence': model_params['max_sequence'],
    }

    wandb.init(
        project=args['wandb_project_name'],
        name='rock_all_genres_sent_epochs_{0}_d_model_{1}_dim_f_{2}'.format(
            args['epochs'],
            model_params['d_model'],
            model_params['dim_feedforward'],
        ),
        config=wandb_config,
    )
    model = MusicTransformer(
        **model_params,
    )
    print('Let us use', torch.cuda.device_count(), 'GPUs!')

    model = nn.DataParallel(model)

    # Continuing from previous training session
    start_epoch = BASELINE_EPOCH
    if args['continue_weights'] is not None:
        if args['continue_epoch'] is None:
            print('ERROR: Need epoch number to continue from (-continue_epoch) when using continue_weights')
            return
        else:
            model.load_state_dict(torch.load(args['continue_weights'], map_location='cpu'))
            start_epoch = args['continue_epoch']
    elif args['continue_epoch'] is not None:
        print('ERROR: Need continue weights (-continue_weights) when using continue_epoch')
        return

    model = model.to(get_device())

    # Lr Scheduler vs static lr
    if args['lr'] is None:
        if args['continue_epoch'] is None:
            init_step = 0
        else:
            init_step = args['continue_epoch'] * len(train_loader)

        lr = LR_DEFAULT_START
        lr_stepper = LrStepTracker(model_params['d_model'], SCHEDULER_WARMUP_STEPS, init_step)
    else:
        lr = args['lr']

    # Not smoothing evaluation loss
    eval_loss_func = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)

    # SmoothCrossEntropyLoss or CrossEntropyLoss for training
    if args['ce_smoothing'] is None:
        train_loss_func = eval_loss_func
    else:
        train_loss_func = nn.CrossEntropyLoss(
            label_smoothing=args['ce_smoothing'],
            ignore_index=TOKEN_PAD,
        )

    # Optimizer
    opt = Adam(model.parameters(), lr=lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)

    if args['lr'] is None:
        lr_scheduler = LambdaLR(opt, lr_stepper.step)
    else:
        lr_scheduler = None

    # Tracking best evaluation accuracy
    best_eval_acc = 0
    best_eval_acc_epoch = -1
    best_eval_loss = float('inf')
    best_eval_loss_epoch = -1

    # Results reporting
    if not os.path.isfile(results_file):
        with open(results_file, 'w', newline='') as o_stream:
            writer = csv.writer(o_stream)
            writer.writerow(CSV_HEADER)

    # TRAIN LOOP
    for epoch in range(start_epoch, args['epochs']):
        epoch_print = epoch + 1
        # Baseline has no training and acts as a base loss and accuracy (epoch 0 in a sense)
        if epoch > BASELINE_EPOCH:
            print(SEPERATOR)
            print('NEW EPOCH:', epoch_print)
            print(SEPERATOR)
            print()

            # Train
            train_epoch(
                epoch_print,
                model,
                train_loader,
                train_loss_func,
                opt,
                lr_scheduler,
                args['print_modulus'],
            )

            print(SEPERATOR)
            print('Evaluating:')
        else:
            print(SEPERATOR)
            print('Baseline model evaluation (Epoch 0):')

        # Eval
        train_loss, train_acc = eval_model(model, train_loader, train_loss_func)
        eval_loss, eval_acc = eval_model(model, val_loader, eval_loss_func)

        wandb.log(
            {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'eval_loss': eval_loss,
                'eval_acc': eval_acc,
            },
        )

        # Learn rate
        lr = get_lr(opt)

        print('Epoch:', epoch_print)
        print('Avg train loss:', train_loss)
        print('Avg train acc:', train_acc)
        print('Avg eval loss:', eval_loss)
        print('Avg eval acc:', eval_acc)
        print(SEPERATOR)
        print()

        new_best = False
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            best_eval_acc_epoch = epoch_print
            torch.save(model.state_dict(), best_acc_file)
            new_best = True

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_eval_loss_epoch = epoch_print
            torch.save(model.state_dict(), best_loss_file)
            new_best = True

        # Writing out new bests
        if new_best:
            with open(best_text, 'w') as best_text_o_stream:
                print('Best eval acc epoch:', best_eval_acc_epoch, file=best_text_o_stream)
                print('Best eval acc:', best_eval_acc, file=best_text_o_stream)
                print()
                print('Best eval loss epoch:', best_eval_loss_epoch, file=best_text_o_stream)
                print('Best eval loss:', best_eval_loss, file=best_text_o_stream)

        if epoch_print % args['weight_modulus'] == 0:
            epoch_str = str(epoch_print).zfill(PREPEND_ZEROS_WIDTH)
            path = os.path.join(weights_folder, f'epoch_{epoch_str}.pickle')
            torch.save(model.state_dict(), path)

        with open(results_file, 'a', newline='') as results_o_stream:
            writer = csv.writer(results_o_stream)
            writer.writerow([epoch_print, lr, train_loss, train_acc, eval_loss, eval_acc])

    wandb.run.summary['best_acc_epoch'] = best_eval_acc_epoch
    wandb.run.summary['best_loss_epoch'] = best_eval_loss_epoch

    print('Testing best loss epoch:')
    model.load_state_dict(torch.load(best_loss_file))
    best_loss_test_loss, best_loss_test_acc = eval_model(model, test_loader, eval_loss_func)

    wandb.run.summary['best_loss_epoch_test_accuracy'] = best_loss_test_acc
    wandb.run.summary['best_loss_epoch_test_loss'] = best_loss_test_loss

    if best_eval_loss_epoch != best_eval_acc_epoch:
        print('Testing best accuracy epoch:')
        model.load_state_dict(torch.load(best_acc_file))
        best_acc_test_loss, best_acc_test_acc = eval_model(model, test_loader, eval_loss_func)

        wandb.run.summary['best_acc_epoch_test_accuracy'] = best_acc_test_acc
        wandb.run.summary['best_acc_epoch_test_loss'] = best_acc_test_loss


if __name__ == '__main__':
    main()
