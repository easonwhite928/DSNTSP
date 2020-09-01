import sys
sys.path.append('.')

import argparse
from typing import Type
from data.temporal_sets_data_loader import get_temporal_sets_data_loader
from learner.tsp_learner import TSPLearner
from config import get_config
from registry.registry import registry


def temporal_sets_prediction(parser: argparse.ArgumentParser):
    parser.add_argument('--mode', '-m', type=str, choices=['train', 'test'], required=True, help='Running mode: should be train or test')
    parser.add_argument('--config', '-c', type=str, required=True, help='Configure path: located in config/')
    parser.add_argument('--cuda', type=int, default=-1, required=True, help='Whether use cuda?')
    args = parser.parse_args()
    config_path = args.config
    config = get_config(config_path)
    print(config)

    train_dl, valid_dl, test_dl = get_temporal_sets_data_loader(
        data_path=config['data_path'],
        data_info_path=config['data_info_path'],
        batch_size=config['batch_size'])
    
    if not config['cls'] in registry.keys():
        print('model cls may not be registered in main/registry.py')
        return
    model_cls = registry[config['cls']]

    learner = TSPLearner(model_cls=model_cls, train_dl=train_dl, valid_dl=valid_dl, test_dl=test_dl, config=config, cuda=args.cuda)

    if args.mode == 'train': learner.train()
    elif args.mode == 'test': learner.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model.')
    temporal_sets_prediction(parser)
