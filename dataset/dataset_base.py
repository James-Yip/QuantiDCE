import os

from util.config_base import Config


class Dataset:
    """Base class for all dataset classes.#TODO
    """

    def __init__(self):
        self.data_config = self.get_data_config()
        self.data_iterator = self.get_data_iterator()

    def get_data_config(self):
        raise NotImplementedError

    def get_data_iterator(self):
        raise NotImplementedError

    def switch_to_train_data(self):
        raise NotImplementedError

    def switch_to_val_data(self):
        raise NotImplementedError

    def switch_to_test_data(self):
        raise NotImplementedError

    @property
    def num_train_batch(self):
        raise NotImplementedError

    @property
    def num_val_batch(self):
        raise NotImplementedError

    @property
    def num_test_batch(self):
        raise NotImplementedError

    def __iter__(self):
        return self.data_iterator.__iter__()


class DataConfig(Config):
    """#TODO
    """

    def __init__(self,
                 dataset_name,
                 train_batch_size,
                 eval_batch_size,
                 test_batch_size,
                 root_data_dir='data'):
        self.root_data_dir = root_data_dir
        self.dataset_name = dataset_name
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size

    @property
    def data_dir_path(self):
        return os.path.join(self.root_data_dir, self.dataset_name)
