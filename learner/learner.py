from typing import Dict, Type, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import ParameterGrid


class Learner(object):

    def __init__(self, 
                 model_cls: Type, 
                 train_dl: DataLoader, 
                 valid_dl: DataLoader, 
                 test_dl: DataLoader, 
                 config: Dict[str, Any],
                 cuda: int):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl
        self.config = config
        self.model_cls = model_cls
        self.model = model_cls(**config["model"])
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        if hasattr(nn, config['loss']):
            self.loss_func = getattr(nn, config['loss'])(reduction=config["reduction"])

        self.optimizer = config['optimizer']
        self.learning_rate = config['learning_rate']
        self.weight_decay = config['weight_decay']
        self.early_stop_rounds = config["early_stop"]
        self.tensor_board = config['tensor_board']
        if self.tensor_board:
            self.writer = SummaryWriter(config['tensor_board_folder'])
        self.save_folder = config['save_folder']
        self.evaluate_path = config["evaluate_path"]
        self.predict_path = config["predict_path"]
        self.use_cuda = cuda
        self.param_space = config['grid_search']
    
    def train(self):
        raise NotImplementedError()
    
    def test(self):
        raise NotImplementedError()
    
    def get_preds(self):
        raise NotImplementedError()

    def save(self, file):
        torch.save(self.model.state_dict(), file)

    def load(self, file):
        self.model.load_state_dict(torch.load(file))

    def cuda(self, x):
        return x.cuda(self.use_cuda) if self.use_cuda != -1 else x

    def grid_search(self):
        model_grid = self.param_space['model']
        for idx, model_params in enumerate(list(ParameterGrid(model_grid))[0]):
            if self.tensor_board:
                self.writer = SummaryWriter(f"{self.config['tensor_board_folder']}/model_params_{idx}")
            self.model = self.model_cls(**model_params)
            self.train()
