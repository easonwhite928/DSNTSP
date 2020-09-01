from typing import Dict, Type, Any
from pathlib import Path
import json

from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from learner.learner import Learner
from learner.metric import recall, f1_score, ndcg


class TSPLearner(Learner):

    def __init__(self,
                 model_cls: Type,
                 train_dl: DataLoader,
                 valid_dl: DataLoader,
                 test_dl: DataLoader,
                 config: Dict[str, Any],
                 cuda: int):
        super(TSPLearner, self).__init__(model_cls, train_dl, valid_dl, test_dl, config, cuda)

        # get loss_func
        self.loss_func = getattr(nn, config['loss'])(reduction=config["reduction"])

    def train(self):
        self.model = self.cuda(self.model)
        self.loss_func = self.cuda(self.loss_func)
        if self.save_folder: Path(self.save_folder).mkdir(parents=True, exist_ok=True)
        print(self.model)

        optimizer_func = getattr(torch.optim, self.optimizer)(self.model.parameters(),
                                                              lr=self.learning_rate,
                                                              weight_decay=self.weight_decay)

        early_stop_rounds = self.early_stop_rounds
        min_valid_loss = float('inf')
        best_epoch = -1

        for epoch in range(self.epochs):
            # train model
            self.model.train()

            total_train_loss = 0.0
            pbar = tqdm(self.train_dl, ncols=100)
            for step, (inputs, targets) in enumerate(pbar):
                inputs, targets = self.cuda(inputs), self.cuda(targets)
                # predicts = self.model(inputs)
                output = self.model(inputs)
                if type(output) == tuple:
                    predicts = output[0]
                    secondary_loss = output[1]
                    main_loss = self.loss_func(predicts, targets)
                    loss = main_loss + secondary_loss
                    total_train_loss += main_loss.item()
                else:
                    predicts = output
                    loss = self.loss_func(predicts, targets)
                    total_train_loss += loss.item()

                optimizer_func.zero_grad()
                loss.backward()
                optimizer_func.step()

                pbar.set_description(f'train epoch: {epoch}, train loss: {total_train_loss / (step + 1):.8f}')

            train_loss = total_train_loss / len(pbar)

            valid_loss, scores = self._evaluate(self.valid_dl, desc='validate ...')

            print(f'validate loss: {valid_loss}')
            print(f'validate scores: {scores}')

            if self.writer:
                self.writer.add_scalar('loss/train_loss', train_loss, global_step=epoch)
                self.writer.add_scalar('loss/valid_loss', valid_loss, global_step=epoch)
                for key, value in scores.items():
                    self.writer.add_scalar(f'scores/{key}', value, global_step=epoch)

            # save_model
            self.save(Path(self.save_folder) / f'model_{epoch}')

            # early stop
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                best_epoch = epoch

            if epoch - best_epoch > early_stop_rounds:
                print(f'Best epoch is {best_epoch}, minimum validation loss is {min_valid_loss}, early stop!')
                break

    def test(self):
        self.load(Path(self.save_folder) / f'model_{self.config["best_epoch"]}')
        self.model = self.cuda(self.model)
        self.loss_func = self.cuda(self.loss_func)

        test_loss, scores = self._evaluate(self.test_dl, desc='test ...')
        print(f'test loss: {test_loss}')
        print(f'test scores: {scores}')

        predict_result = {'predict': [], 'target': []}
        with torch.no_grad():
            pbar = tqdm(self.test_dl, ncols=100)
            for step, (inputs, targets) in enumerate(pbar):
                inputs, targets = self.cuda(inputs), self.cuda(targets)
                # predicts = self.model(inputs)

                output = self.model(inputs)
                if type(output) == tuple: predicts = output[0]
                else: predicts = output

                # print predict and target
                for predict, target in zip(predicts, targets):
                    _, predict_indices = predict.topk(k=10)
                    predict_result['predict'].append(predict_indices.cpu().detach().tolist())
                    predict_result['target'].append(target.nonzero().flatten().cpu().detach().tolist())

        predict_df = pd.DataFrame(data=predict_result)
        print(predict_df)

        Path(self.evaluate_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.evaluate_path, 'w') as f:
            f.write(json.dumps(scores, indent=4))

        Path(self.predict_path).parent.mkdir(parents=True, exist_ok=True)
        predict_df.to_csv(self.predict_path)

    def _evaluate(self, data_loader, desc):
        with torch.no_grad():
            self.model.eval()
            total_loss, num = 0.0, 0
            scores = {}
            for top_k in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                scores.update({
                    f'recall_{top_k}': 0,
                    f'f1_score_{top_k}': 0,
                    f'ndcg_{top_k}': 0
                })
            pbar = tqdm(data_loader, ncols=100)
            for step, (inputs, targets) in enumerate(pbar):
                inputs, targets = self.cuda(inputs), self.cuda(targets)
                # predicts = self.model(inputs)

                output = self.model(inputs)
                if type(output) == tuple: predicts = output[0]
                else: predicts = output

                loss = self.loss_func(predicts, targets)

                total_loss += loss.item() * targets.shape[0]

                for top_k in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                    scores[f'recall_{top_k}'] += recall(predicts, targets, top_k=top_k) * targets.shape[0]
                    scores[f'f1_score_{top_k}'] += f1_score(predicts, targets, top_k=top_k) * targets.shape[0]
                    scores[f'ndcg_{top_k}'] += ndcg(predicts, targets, top_k=top_k) * targets.shape[0]

                num += targets.shape[0]

                pbar.set_description(desc=desc)
            loss = total_loss / num
            for name in scores.keys():
                scores[name] /= num
        return loss, scores
