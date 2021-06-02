import os

from dataset.dailydialog_plusplus_base import DailyDialogPlusPlus
from dataset.dailydialog_plusplus_base import DailyDialogPlusPlusConfig


class DailyDialogPlusPlusMLR(DailyDialogPlusPlus):
    """#TODO
    """

    def __init__(self, args):
        self.args = args
        super().__init__(args)

    def get_data_config(self):
        data_config = DailyDialogPlusPlusMLRLossConfig(self.args)
        return data_config


class DailyDialogPlusPlusMLRLossConfig(DailyDialogPlusPlusConfig):
    """#TODO
    """

    def __init__(self, args):
        super().__init__(args)
        train_dir_path = os.path.join(self.data_dir_path, 'train')
        self.cluster_names = os.walk(train_dir_path).__next__()[1]
        self.cluster_names.sort()

    def _get_dataset_hparam(self, split_name, cluster_name):
        file_path = os.path.join(
            self.data_dir_path, split_name, cluster_name, 'context_response.pkl')
        dataset_hparam = {
            'data_name': cluster_name,
            'data_type': 'record',
            'feature_types': self.feature_types,
            'files': file_path,
        }
        return dataset_hparam

    def _get_dataset_hparam_list(self, split_name):
        dataset_hparam_list = [
            self._get_dataset_hparam(split_name, cluster_name)
            for cluster_name in self.cluster_names
        ]
        return dataset_hparam_list
