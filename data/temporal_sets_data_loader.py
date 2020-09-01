import json
from functools import partial
from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class TemporalSetsDataset(Dataset):

    def __init__(self, data_path, data_info_path, data_type='train'):
        with open(data_path, 'r') as file:
            data_dict = json.load(file)

        self.max_num_sets = 512

        self.sets = []
        self.times = []
        self.users = []
        self.targets = []
        for user in data_dict:
            sets = user['sets']
            num_sets = len(sets)
            if data_type == 'train':
                for idx in range(1, num_sets - 2):
                    self._add_data(sets, user['user_id'], idx)
            elif data_type == 'validate':
                self._add_data(sets, user['user_id'], num_sets - 2)
            elif data_type == 'test':
                self._add_data(sets, user['user_id'], num_sets - 1)
            else:
                raise NotImplementedError()

        assert len(self.sets) == len(self.targets)
        assert len(self.sets) == len(self.times)

        with open(data_info_path, 'r') as file:
            data_info_dict = json.load(file)
        self.items_total = data_info_dict['num_items']

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, idx):
        return self.sets[idx], self.times[idx], self.users[idx], self.targets[idx]

    def _add_data(self, sets, user_id, idx):
        user_sets, times = [], []
        select_sets = sets[:idx] if idx - self.max_num_sets < 0 else sets[idx - self.max_num_sets : idx]
        for user_set in select_sets:
            user_sets.append(user_set['items'])
            # times.append(user_set['timestamp'])
            times.append(sets[idx]['timestamp'] - user_set['timestamp'])
        self.sets.append(user_sets)
        self.times.append(times)
        self.users.append(user_id)
        self.targets.append(sets[idx]['items'])


class TemporalSetsInput(object):

    def __init__(self, sets_batch, times_batch, users_batch, items_total):
        """
        Args:
            sets_batch (List[List[List]]): shape (batch_size, num_sets, num_items)
            times_batch (List[List]): shape (batch_size, num_sets)
        """
        self.sets_batch = [[torch.tensor(user_set) for user_set in user_sets] for user_sets in sets_batch]
        self.times_batch = [torch.tensor(user_times) for user_times in times_batch]
        self.users_batch = torch.tensor(users_batch)
        self.items_total = items_total
        self.batch_size = len(sets_batch)
        self.negative_sample = False

    def get_users(self) -> torch.Tensor:
        return self.users_batch

    def get_items(self) -> List[torch.Tensor]:
        """
        Returns:
            output (List[Tensor]): shape (batch_size, num_items)
        """
        items_batch = []
        for user_sets in self.sets_batch:
            user_items = torch.cat([user_set for user_set in user_sets], dim=-1)
            user_items = user_items[-512:] if len(user_items) > 512 else user_items
            items_batch.append(user_items)
        return items_batch
        # return [torch.cat([user_set for user_set in user_sets], dim=-1) for user_sets in self.sets_batch]

    # def get_item_positions(self) -> List[torch.Tensor]:
    #     """
    #     Returns:
    #         output (List[Tensor]): shape (batch_size, num_items)
    #     """
    #     return [torch.cat([user_set.new_full(user_set.shape, idx) for idx, user_set in enumerate(user_sets)]) for user_sets in self.sets_batch]

    def get_item_times(self) -> List[torch.Tensor]:
        """
        Returns:
            output (List[Tensor]): shape (batch_size, num_items)
        """
        times_batch = []
        for user_sets, user_times in zip(self.sets_batch, self.times_batch):
            user_items_times = []
            for user_set, time in zip(user_sets, user_times):
                user_items_times.append(user_set.new_full(user_set.shape, time))
            user_items_times = torch.cat(user_items_times, dim=-1)
            user_items_times = user_items_times[-512:] if len(user_items_times) > 512 else user_items_times
            times_batch.append(user_items_times)
        return times_batch
        # return [torch.cat([user_set.new_full(user_set.shape, time) for user_set, time in zip(user_sets, user_times)])
        #         for user_sets, user_times in zip(self.sets_batch, self.times_batch)]

    def get_sets(self) -> List[List[torch.Tensor]]:
        """
        Returns:
            output: shape (batch_size, num_sets, num_items)
        """
        return self.sets_batch

    def get_set_times(self) -> List[torch.Tensor]:
        """
        Returns:
            output: shape (batch_size, num_sets)
        """
        return self.times_batch

    def cuda(self, device):
        self.sets_batch = [[user_set.cuda(device) for user_set in user_sets] for user_sets in self.sets_batch]
        self.times_batch = [user_times.cuda(device) for user_times in self.times_batch]
        self.users_batch = self.users_batch.cuda(device)
        if self.negative_sample:
            self.positives_batch = [positives.cuda(device) for positives in self.positives_batch]
            self.negatives_batch = [negatives.cuda(device) for negatives in self.negatives_batch]
        return self

    def generate_negatives(self, positives_batch, num_samples=-1):
        self.negative_sample = True
        self.num_samples = num_samples
        self.positives_batch = [torch.tensor(positives) for positives in positives_batch]
        self.negatives_batch = []
        for positives in self.positives_batch:
            negatives = []
            for _ in range(self.num_samples - positives.shape[0]):
                negative = positives[0].item()
                while negative in positives:
                    negative = torch.randint(0, self.items_total, (1,)).item()
                negatives.append(negative)
            negatives = torch.tensor(negatives)
            self.negatives_batch.append(negatives)
        self.samples = torch.stack([torch.cat([positives, negatives]) for positives, negatives in zip(self.positives_batch, self.negatives_batch)])
        assert self.samples.shape == torch.Size([self.batch_size, self.num_samples])


def collate_fn(data, items_total, negative_sample=False, num_samples=10):
    """
    Args:
        data (List[Tuple]):
            tuple[0] (List[List[List]]): sets batch, shape (batch_size, num_sets, num_items)
            tuple[1] (List[List]): times batch, shape (batch_size, num_sets)
            tuple[2] (List): users batch, shape (batch_size,)
            tuple[3] (List): targets batch, shape (batch_size, num_items)
    Returns:
        inputs (Tensor): shape (batch_size, max_num_items)
        targets (Tensor): shape (batch_size, items_total)
    """
    # print(data)
    sets_batch = [sets for sets, times, user, targets in data]
    times_batch = [times for sets, times, user, targets in data]
    users_batch = [user for sets, times, user, targets in data]
    temporal_sets_input = TemporalSetsInput(sets_batch=sets_batch,
                                            times_batch=times_batch,
                                            users_batch=users_batch,
                                            items_total=items_total)
    if negative_sample:
        positives_batch = [targets for sets, times, user, targets in data]
        temporal_sets_input.generate_negatives(positives_batch, num_samples)
        targets = torch.zeros(temporal_sets_input.batch_size, num_samples)
        for idx, positives in enumerate(positives_batch):
            targets[idx][:len(positives)] = 1
    else:
        targets = torch.stack([torch.zeros(items_total).index_fill_(0, torch.tensor(target), 1) for _, _, _, target in data])
    return temporal_sets_input, targets


def get_temporal_sets_data_loader(data_path, data_info_path, batch_size, negative_sample=False, num_samples=10):
    train_dataset = TemporalSetsDataset(data_path, data_info_path, data_type='train')
    validate_dataset = TemporalSetsDataset(data_path, data_info_path, data_type='validate')
    test_dataset = TemporalSetsDataset(data_path, data_info_path, data_type='test')

    # print(f'train_dataset.items_total -> {train_dataset.items_total}')
    # for idx in range(5): print(f'train_dataset[{idx}] -> {train_dataset[idx]}')
    # for idx in range(5): print(f'validate_dataset[{idx}] -> {validate_dataset[idx]}')
    # for idx in range(5): print(f'test_dataset[{idx}] -> {test_dataset[idx]}')

    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   collate_fn=partial(collate_fn, items_total=train_dataset.items_total, negative_sample=negative_sample, num_samples=num_samples))
    validate_data_loader = DataLoader(dataset=validate_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      collate_fn=partial(collate_fn, items_total=validate_dataset.items_total))
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  collate_fn=partial(collate_fn, items_total=test_dataset.items_total))

    return train_data_loader, validate_data_loader, test_data_loader


if __name__ == '__main__':
    train_data_loader, validate_data_loader, test_data_loader = get_temporal_sets_data_loader(data_path='data/tafeng.json',
                                                                                              data_info_path='data/tafeng_info.json',
                                                                                              batch_size=3)

    print('===== train_data_loader =====')
    for idx, (inputs, targets) in enumerate(train_data_loader):
        print(f'targets.shape -> {targets.shape}')
        # print(f'inputs -> {inputs}')
        # print(f'targets -> {targets}')
        break

    print('===== validate_data_loader =====')
    for idx, (inputs, targets) in enumerate(validate_data_loader):
        print(f'targets.shape -> {targets.shape}')
        # print(f'inputs -> {inputs}')
        # print(f'targets -> {targets}')
        break

    print('===== test_data_loader =====')
    for idx, (inputs, targets) in enumerate(test_data_loader):
        print(f'targets.shape -> {targets.shape}')
        # print(f'inputs -> {inputs}')
        # print(f'targets -> {targets}')
        break
