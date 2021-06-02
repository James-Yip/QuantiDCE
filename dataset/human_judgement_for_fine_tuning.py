import os
import json

import torch
import texar.torch as tx

from dataset.dataset_base import Dataset
from dataset.dataset_base import DataConfig


class HumanJudgementForFineTuning(Dataset):
    """#TODO
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:{}".format(args.gpu))
        super().__init__()

    def get_data_iterator(self):
        train_data = tx.data.RecordData(
            hparams=self.data_config.train_hparams, device=self.device)
        valid_data = tx.data.RecordData(
            hparams=self.data_config.valid_hparams, device=self.device)
        data_iterator = tx.data.DataIterator(
            {'train': train_data,
             'valid': valid_data})
        return data_iterator

    def get_data_config(self):
        data_config = HumanJudgementForFineTuningConfig(self.args)
        return data_config

    def switch_to_train_data(self):
        self.data_iterator.switch_to_dataset('train')

    def switch_to_val_data(self):
        self.data_iterator.switch_to_dataset('valid')

    @property
    def num_train_batch(self):
        return len(self.data_iterator._datasets['train'])

    @property
    def num_val_batch(self):
        return len(self.data_iterator._datasets['valid'])


class HumanJudgementForFineTuningConfig(DataConfig):
    """#TODO
    """

    def __init__(self, args):
        self.args = args
        super().__init__(dataset_name='human_judgement_for_fine_tuning',
                         train_batch_size=args.train_batch_size,
                         eval_batch_size=args.eval_batch_size,
                         test_batch_size=0)
        feature_type_path = os.path.join(self.data_dir_path,
                                         self.args.data,
                                         'feature_types.json')
        with open(feature_type_path, 'r') as f:
            self.feature_types = json.load(f)

    @property
    def train_hparams(self):
        file_path = os.path.join(
            self.data_dir_path, self.args.data,
            'train', 'context_response.pkl')
        train_hparams = {
            'shuffle': True,
            'allow_smaller_final_batch': True,
            'batch_size': self.train_batch_size,
            'dataset': {
                'feature_types': self.feature_types,
                'files': file_path,
            },
        }
        return train_hparams

    @property
    def valid_hparams(self):
        file_path = os.path.join(
            self.data_dir_path, self.args.data,
            'validation', 'context_response.pkl')
        train_hparams = {
            'shuffle': False,
            'allow_smaller_final_batch': True,
            'batch_size': self.eval_batch_size,
            'dataset': {
                'feature_types': self.feature_types,
                'files': file_path,
            },
        }
        return train_hparams
