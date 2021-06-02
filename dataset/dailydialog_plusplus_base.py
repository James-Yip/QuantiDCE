import os
import json

import torch
import texar.torch as tx

from dataset.dataset_base import Dataset
from dataset.dataset_base import DataConfig


class DailyDialogPlusPlus(Dataset):
    """#TODO
    """

    def __init__(self, args):
        self.device = torch.device("cuda:{}".format(args.gpu))
        super().__init__()

    def get_data_iterator(self):
        train_dataset = tx.data.MultiAlignedData(
            hparams=self.data_config.train_hparams, device=self.device)
        valid_dataset = tx.data.MultiAlignedData(
            hparams=self.data_config.valid_hparams, device=self.device)
        test_dataset = tx.data.MultiAlignedData(
            hparams=self.data_config.test_hparams, device=self.device)
        data_iterator = tx.data.DataIterator(
            {'train': train_dataset,
             'valid': valid_dataset,
             'test': test_dataset})
        return data_iterator

    def switch_to_train_data(self):
        self.data_iterator.switch_to_dataset('train')

    def switch_to_val_data(self):
        self.data_iterator.switch_to_dataset('valid')

    def switch_to_test_data(self):
        self.data_iterator.switch_to_dataset('test')

    @property
    def num_train_batch(self):
        return len(self.data_iterator._datasets['train'])

    @property
    def num_val_batch(self):
        return len(self.data_iterator._datasets['valid'])

    @property
    def num_test_batch(self):
        return len(self.data_iterator._datasets['test'])


class DailyDialogPlusPlusConfig(DataConfig):
    """#TODO
    """

    def __init__(self, args, load_feature_types=True):
        super().__init__(args.dataset,
                         args.train_batch_size,
                         args.eval_batch_size,
                         args.test_batch_size)
        if load_feature_types:
            feature_type_path = os.path.join(self.data_dir_path,
                                            'feature_types.json')
            with open(feature_type_path, 'r') as f:
                self.feature_types = json.load(f)

    def _get_dataset_hparam_list(self, split_name):
        raise NotImplementedError

    @property
    def train_hparams(self):
        dataset_hparam_list = self._get_dataset_hparam_list('train')
        train_hparams = {
            'shuffle': True,
            'allow_smaller_final_batch': False,
            'batch_size': self.train_batch_size,
            'datasets': dataset_hparam_list,
        }
        return train_hparams

    @property
    def valid_hparams(self):
        dataset_hparam_list = self._get_dataset_hparam_list('validation')
        valid_hparams = {
            'shuffle': False,
            'allow_smaller_final_batch': True,
            'batch_size': self.eval_batch_size,
            'datasets': dataset_hparam_list,
        }
        return valid_hparams

    @property
    def test_hparams(self):
        dataset_hparam_list = self._get_dataset_hparam_list('test')
        test_hparams = {
            'shuffle': False,
            'allow_smaller_final_batch': True,
            'batch_size': self.test_batch_size,
            'datasets': dataset_hparam_list,
        }
        return test_hparams
