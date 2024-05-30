import argparse
import json
import os
import time

import torch

from dataset.e_piano import compute_epiano_accuracy
from utilities.constants import SEPERATOR
from utilities.device import get_device
from utilities.lr_scheduling import get_lr


def train_epoch(  # noqa: WPS213, WPS211
    cur_epoch,
    model,
    dataloader,
    loss,
    opt,
    lr_scheduler=None,
    print_modulus=50,
):
    """Trains a single model epoch.

    Args:
        cur_epoch (int) : number of current epoch
        model : model to train
        dataloader : dataloader
        loss : loss function
        opt : optimizer
        lr_scheduler : learning rate scheduler
        print_modulus (int) : modulus for ptinting results

    Author: Damon Gwinn
    """
    model.train()
    for batch_num, batch in enumerate(dataloader):
        time_before = time.time()

        opt.zero_grad()

        x, local_features, global_features, target = batch

        pred = model(
            x.to(get_device()),
            local_features.to(get_device()),
            global_features.to(get_device())
        )
        pred = pred.reshape(pred.shape[0] * pred.shape[1], -1)
        target = target.flatten()

        loss_value = loss.forward(pred.cpu(), target)

        loss_value.backward()
        opt.step()

        if (lr_scheduler is not None):
            lr_scheduler.step()

        time_after = time.time()
        time_took = time_after - time_before

        if (batch_num + 1) % print_modulus == 0:
            print(SEPERATOR)
            print('Epoch {0}  Batch {1} / {2}'.format(
                cur_epoch, batch_num + 1, len(dataloader),
            ))
            print('LR:', get_lr(opt))
            print('Train loss:', float(loss_value))
            print('')
            print('Time (s):', time_took)
            print(SEPERATOR)
            print('')


def eval_model(model, dataloader, loss):
    """Evaluates model.

    Args:
        model : model
        dataloader : dataloader
        loss : loss function

    Returns:
        float, float : average loss in batches, average accuracy in batches

    Author: Damon Gwinn
    """
    model.eval()

    avg_acc = -1
    avg_loss = -1

    with torch.set_grad_enabled(False):
        n_test = len(dataloader)
        sum_loss = 0.0  # noqa: WPS358
        sum_acc = 0.0  # noqa: WPS358
        for batch in dataloader:
            x, local_features, global_features, target = batch
            pred = model(
                x.to(get_device()),
                local_features.to(get_device()),
                global_features.to(get_device())
            )

            sum_acc += compute_epiano_accuracy(pred.cpu(), target)

            pred = pred.reshape(pred.shape[0] * pred.shape[1], -1)
            target = target.flatten()

            loss_value = loss.forward(pred.cpu(), target)
            sum_loss += loss_value

        avg_loss = sum_loss / n_test
        avg_acc = sum_acc / n_test

    return avg_loss, avg_acc


def parse_json(file_type=None):
    """Parse model arguments from json file.

    Args:
        file_type: flag to indicate type of file, train/eval/None

    Returns:
        model_params: parsed model arguments
        args: parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-path_to_model_params',
        default='model/model_params.json',
        help='Path to json with model parameters',
    )
    if file_type == 'train':
        default_file_path = 'parameters.json'
    elif file_type == 'eval':
        default_file_path = 'generation/evaluate_loss.json'
    else:
        default_file_path = ''

    parser.add_argument(
        '-path_to_params',
        default=default_file_path,
        help='Path to json with default arguments',
    )
    if os.path.exists(parser.parse_args().path_to_model_params):
        with open(parser.parse_args().path_to_model_params, 'r') as model_parameters_file:  # noqa: E501
            model_params = json.load(model_parameters_file)

    if file_type:
        if os.path.exists(parser.parse_args().path_to_params):
            with open(parser.parse_args().path_to_params, 'r') as parameters_file:
                args = json.load(parameters_file)
        return model_params, args

    return model_params
