import os
import random
import logging
import shutil
import sys

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from typing import Dict, Union
import prettytable as pt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA

sys.path.append('..')
from evaluation.evaluator import Evaluator


class Trainer:
    """#TODO: adds docstring
    """

    TRAIN = 'train'
    VALIDATION = 'valid'
    TEST = 'test'

    def __init__(self, model, dataset, args):
        # constants
        self.model = model
        self.dataset = dataset
        self.num_epochs = args.num_epochs
        self.learning_rate = args.learning_rate
        self.gpu = args.gpu
        self.checkpoint_dir_path = args.checkpoint_dir_path
        self.logging_level = args.logging_level
        self.display_steps = args.display_steps
        self.monitor_metric_list = [
            {'name': n, 'type': t}
            for n, t in zip(args.monitor_metric_name, args.monitor_metric_type)
        ]
        # variables
        self.cur_train_results = None
        self.cur_eval_results = None
        self.cur_best_results = {key: None for key in args.monitor_metric_name}
        self.cur_monitor_metric = None
        self.cur_epoch_id = None
        self.num_eval_steps_per_epoch = None
        self.global_step =  {
            Trainer.TRAIN: 0,
            Trainer.VALIDATION: 0,
            Trainer.TEST: 0,
        }
        self._create_output_dir()
        self.logger = self._get_logger()
        self.feature_visualizer = self._get_feature_visualizer()
        self.score_visualizer = self._get_score_visualizer()
        self.evaluator = Evaluator(
            checkpoint_dir_path=args.checkpoint_dir_path,
            result_file_name='human_correlation_results_train.txt',
            console_output=False)

    def run(self, mode):
        if mode == Trainer.TRAIN:
            self._before_training()
            self._train()
            self._test()
            self._after_training()
        if mode == Trainer.TEST:
            self._test()

    def _before_training(self):
        self.tensorboard_writer = self._get_tensorboard_writer()

    def _after_training(self):
        self.tensorboard_writer.close()

    def _train(self):
        for epoch_id in tqdm(range(self.num_epochs)):
            self.cur_epoch_id = epoch_id
            self.cur_train_results = self._train_epoch()
            self.cur_eval_results = self._eval_epoch(Trainer.VALIDATION)
            self.evaluator.evaluate(
                self.model,
                additional_eval_info='train_epoch{}'.format(self.cur_epoch_id))
            self._checkpoint()

    def _test(self):
        for metric in self.monitor_metric_list:
            self.cur_monitor_metric = metric
            self._load()
            self.cur_epoch_id = self.num_epochs - 1
            test_results = self._eval_epoch(Trainer.TEST)
            self._print(test_results)

    def _print(self, results: Dict[str, Union[int, float]]):
        table = pt.PrettyTable(['metrics', 'results'])
        for metric, result in results.items():
            table.add_row([metric, '%.4f' % result])
        print(table)

    def _checkpoint(self):
        for metric in self.monitor_metric_list:
            self.cur_monitor_metric = metric
            if self._is_best_result():
                self.cur_best_results[metric['name']] = self.cur_eval_results[
                    metric['name']]
                self._save()

    def _is_best_result(self):
        metric = self.cur_monitor_metric
        cur_eval_result = self.cur_eval_results[metric['name']]
        cur_best_result = self.cur_best_results[metric['name']]
        if not cur_best_result:
            return True
        elif metric['type'] == 'max':
            return cur_eval_result > cur_best_result
        elif metric['type'] == 'min':
            return cur_eval_result < cur_best_result
        else:
            error_info = 'monitor_metric_type "{}" is invalid.'.format(
                metric['type'])
            raise ValueError(error_info)

    def _switch_mode(self, mode):
        if mode == Trainer.TRAIN:
            self.model.train()
            self.dataset.switch_to_train_data()
        elif mode == Trainer.VALIDATION:
            self.model.eval()
            self.dataset.switch_to_val_data()
            self.num_eval_steps_per_epoch = self.num_valid_steps_per_epoch
        elif mode == Trainer.TEST:
            self.model.eval()
            self.dataset.switch_to_test_data()
            self.num_eval_steps_per_epoch = self.num_test_steps_per_epoch
        else:
            error_info = 'mode "{}" is invalid.'.format(mode)
            raise ValueError(error_info)

    def _save(self):
        torch.save(self.model.state_dict(), self.checkpoint_file_path)
        metric_name = self.cur_monitor_metric['name']
        save_info = '[epoch {}, {} = {}] saving checkpoint into {}'.format(
            self.cur_epoch_id, metric_name,
            self.cur_best_results[metric_name],
            self.checkpoint_file_path)
        self.logger.info(save_info)

    def _load(self):
        state_dict = torch.load(
            self.checkpoint_file_path,
            map_location='cuda:{}'.format(self.gpu))
        self.model.load_state_dict(state_dict)
        load_info = 'loading checkpoint from: {}'.format(
            self.checkpoint_file_path)
        print(load_info)

    def _create_output_dir(self):
        if not os.path.isdir(self.checkpoint_dir_path):
            os.makedirs(self.checkpoint_dir_path)

    def _get_logger(self):
        level_value = logging.getLevelName(self.logging_level)
        log_file_path = os.path.join(self.checkpoint_dir_path,
                                     'training_logs.txt')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(pathname)s[line:%(lineno)d]'
            ' - %(levelname)s: %(message)s')
        file_handler = logging.FileHandler(log_file_path, mode='a')
        file_handler.setFormatter(file_formatter)
        logger = logging.getLogger(__name__)
        logger.setLevel(level=level_value)
        logger.addHandler(file_handler)
        return logger

    def _get_tensorboard_writer(self):
        if os.path.exists(self.tensorboard_log_dir_path):
            shutil.rmtree(self.tensorboard_log_dir_path)
        tensorboard_writer = SummaryWriter(self.tensorboard_log_dir_path)
        return tensorboard_writer

    def _get_feature_visualizer(self):
        class FeatureVisualizer:
            def __init__(self, output_dir_path):
                self.output_dir_path = output_dir_path
                self.reset()

            def add(self, feature: torch.Tensor, label: str):
                feature = feature.view(-1, feature.size(-1)).cpu()
                self.feature_list.append(feature)
                self.feature_label_list.extend([label] * feature.size(0))

            def reset(self):
                self.feature_list = []
                self.feature_label_list = []

            def write(self, global_step, tag):
                features = torch.cat(self.feature_list, dim=0).numpy()
                pca = PCA(n_components=2, whiten=True)
                reduced_features = pca.fit_transform(features)
                data = {
                    'Reduced Dimension 1': reduced_features[:, 0],
                    'Reduced Dimension 2': reduced_features[:, 1],
                    'Level': self.feature_label_list,
                }
                df = pd.DataFrame(data)
                plt.figure()
                sns.set_palette(sns.color_palette())
                sns.scatterplot(data=df,
                                x='Reduced Dimension 1',
                                y='Reduced Dimension 2',
                                hue='Level')
                plt.savefig(
                    os.path.join(self.output_dir_path,
                                 'features_{}_{}.png'.format(tag, global_step)),
                                 bbox_inches='tight', dpi=120)
                plt.close()

        output_dir_path = os.path.join(self.checkpoint_dir_path,
                                       'visualization_results')
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        return FeatureVisualizer(output_dir_path)

    def _get_score_visualizer(self):
        class ScoreVisualizer:
            def __init__(self, output_dir_path):
                self.output_dir_path = output_dir_path
                self.reset()

            def add(self, score: torch.Tensor, label: str):
                cur_score_list = score.view(-1).cpu().tolist()
                self.score_list.extend(cur_score_list)
                self.score_label_list.extend([label] * len(cur_score_list))

            def reset(self):
                self.score_list = []
                self.score_label_list = []

            def write(self, global_step, tag):
                data = {
                    'Score': self.score_list,
                    'Level': self.score_label_list,
                }
                df = pd.DataFrame(data)
                plt.figure()
                sns.set_palette(sns.color_palette())
                sns.violinplot(x='Score', y='Level', data=df)
                plt.savefig(
                    os.path.join(self.output_dir_path,
                                 'scores_{}_{}.png'.format(tag, global_step)),
                                 bbox_inches='tight', dpi=120)
                plt.close()

        output_dir_path = os.path.join(self.checkpoint_dir_path,
                                       'visualization_results')
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        return ScoreVisualizer(output_dir_path)

    def _train_epoch(self):
        raise NotImplementedError

    def _eval_epoch(self, mode: str):
        raise NotImplementedError

    def _is_time_to_display(self, step):
        return (self.display_steps > 0) and (step % self.display_steps == 0)

    @property
    def num_train_steps_per_epoch(self):
        return self.dataset.num_train_batch

    @property
    def num_valid_steps_per_epoch(self):
        return self.dataset.num_val_batch

    @property
    def num_test_steps_per_epoch(self):
        return self.dataset.num_test_batch

    @property
    def num_total_train_steps(self):
        return self.num_train_steps_per_epoch * self.num_epochs

    @property
    def checkpoint_file_path(self):
        checkpoint_file_name = 'model_best_{}.ckpt'.format(
            self.cur_monitor_metric['name'])
        return os.path.join(self.checkpoint_dir_path, checkpoint_file_name)

    @property
    def tensorboard_log_dir_path(self):
        return os.path.join(self.checkpoint_dir_path, 'tensorboard_logs')
