import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.e_piano import create_epiano_datasets
from model.music_transformer import MusicTransformer
from utilities.constants import *
from utilities.device import get_device, use_cuda
from utilities.run_model import eval_model, parse_json


def main():
    """Author: Damon Gwinn.

    Entry point. Evaluates a model specified by command line arguments.
    """
    model_params, args = parse_json(file_type='eval')

    if args['seed']:
        torch.manual_seed(args['seed'])

    if args['force_cpu']:
        use_cuda(False)
        print('WARNING: Forced CPU usage, expect model to perform slower')
        print()

    # Test Dataset
    with open('dataset/dataset_parameters.json', 'r') as model_params_file:
        dataset_params = json.load(model_params_file)

    dataset_root = dataset_params['dataset_root']
    sequence_size = dataset_params['sequence_size']
    additional_features_params = dataset_params['additional_features_params']

    _, _, test_dataset = create_epiano_datasets(
        dataset_root,
        sequence_size,
        additional_features_params,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args['batch_size'],
        num_workers=args['n_workers'],
    )
    model = MusicTransformer(**model_params).to(get_device())

    model.load_state_dict(torch.load(args['model_weights']))

    # No smoothed loss
    loss = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)

    print('Evaluating:')
    model.eval()

    avg_loss, avg_acc = eval_model(model, test_loader, loss)

    print('Avg loss:', avg_loss)
    print('Avg acc:', avg_acc)
    print(SEPERATOR)
    print()


if __name__ == '__main__':
    main()
