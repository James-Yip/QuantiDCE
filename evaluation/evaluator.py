import os
import logging
from typing import List, Dict
from collections import OrderedDict

import numpy as np
import prettytable as pt
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns

from dataset.human_judgement import HumanJudgementDataset


class Evaluator:
    """#TODO: adds docstring
    """

    def __init__(self,
                 checkpoint_dir_path,
                 eval_data_dir_path='./evaluation/eval_data',
                 result_file_name='human_correlation_results.txt',
                 datasets=['dailydialog_EVAL', 'convai2',
                           'empatheticdialogues'],
                 eval_mode='mix',
                 console_output=True):
        self.checkpoint_dir_path = checkpoint_dir_path
        self.eval_data_dir_path = eval_data_dir_path
        self.datasets = datasets
        self.eval_mode = eval_mode
        self.console_output = console_output
        self.result_file_path = os.path.join(
            checkpoint_dir_path, result_file_name)

        table_header = ['info', 'Pearson', 'Spearman', 'Kendall', 'avg']
        self.table_recorder = TableRecorder(table_header, table_names=datasets)

        self.metric_model = None
        self.additional_eval_info = None

    def evaluate(self, metric_model, additional_eval_info=None):
        self.metric_model = metric_model
        self.additional_eval_info = additional_eval_info
        if self.eval_mode == 'mix':
            self._evaluate_mix()
        elif self.eval_mode == 'separate':
            self._evaluate_sep()

    def _evaluate_mix(self):
        """Uses mix data to evaluate the specified metric model.
        """
        for dataset_name in self.datasets:
            eval_data = HumanJudgementDataset(
                self.eval_data_dir_path, dataset_name)
            predicted_scores, human_scores = self._get_scores(eval_data)
            correlation_results = self._compute_correlation(
                predicted_scores, human_scores)

            result_dict = OrderedDict()
            for meta_metric_name, result in correlation_results.items():
                result_dict[meta_metric_name] = '{}({})'.format(
                    result[0], result[1])
            result_dict['avg'] = np.average(
                [result[0] for result in correlation_results.values()])

            table_row = [self.additional_eval_info] + list(result_dict.values())
            self.table_recorder.add_row(table_row=table_row,
                                        table_name=dataset_name)

            pred_score_dir_path = os.path.join(
                self.checkpoint_dir_path, 'predicted_scores')
            if not os.path.exists(pred_score_dir_path):
                os.makedirs(pred_score_dir_path)

            pred_score_text_name = 'pred_scores_{}_mix_{}.txt'.format(
                dataset_name, self.additional_eval_info)
            pred_score_text_path = os.path.join(
                pred_score_dir_path, pred_score_text_name)
            self._save_txt(predicted_scores, pred_score_text_path)

            pred_score_dist_name = 'pred_score_dist_{}_mix_{}.png'.format(
                dataset_name, self.additional_eval_info)
            pred_score_dist_path = os.path.join(
                pred_score_dir_path, pred_score_dist_name)
            self._save_distribution(predicted_scores, pred_score_dist_path)

        self.table_recorder.save(output_file_path=self.result_file_path)

    def _evaluate_sep(self):
        """Uses separate data to evaluate the specified metric model.
        """
        for dataset_name in self.datasets:
            result_dict = OrderedDict()
            results_of_cur_metric = []
            dialog_model_names = HumanJudgementDataset.DATA_MAP[dataset_name]
            for dialog_model_name in dialog_model_names:
                eval_data = HumanJudgementDataset(
                    self.eval_data_dir_path,
                    dataset_name, dialog_model_name)
                predicted_scores, human_scores = self._get_scores(eval_data)
                correlation_results = self._compute_correlation(
                    predicted_scores, human_scores)

                for meta_metric_name, result in correlation_results.items():
                    key = '{}-{}'.format(
                        dialog_model_name, meta_metric_name)
                    result_dict[key] = '{}({})'.format(result[0], result[1])
                    results_of_cur_metric.append(result[0])

                pred_score_dir_path = os.path.join(
                    self.checkpoint_dir_path, 'predicted_scores')
                if not os.path.exists(pred_score_dir_path):
                    os.makedirs(pred_score_dir_path)

                pred_score_text_name = 'pred_scores_{}_{}_{}.txt'.format(
                    dataset_name, dialog_model_name, self.additional_eval_info)
                pred_score_text_path = os.path.join(
                    pred_score_dir_path, pred_score_text_name)
                self._save_txt(predicted_scores, pred_score_text_path)

                pred_score_dist_name = 'pred_score_dist_{}_{}_{}.png'.format(
                    dataset_name, dialog_model_name, self.additional_eval_info)
                pred_score_dist_path = os.path.join(
                    pred_score_dir_path, pred_score_dist_name)
                self._save_distribution(predicted_scores, pred_score_dist_path)

            result_dict['avg'] = np.average(results_of_cur_metric)
            table_row = [self.additional_eval_info] + list(result_dict.values())
            self.table_recorder.add_row(table_row=table_row,
                                        table_name=dataset_name)

        self.table_recorder.save(output_file_path=self.result_file_path)

    @staticmethod
    def _compute_correlation(predicted_scores: List[float],
                             human_scores: List[float]) -> Dict[str, str]:
        pearson_r, pearson_p = pearsonr(predicted_scores, human_scores)
        spearman_r, spearman_p = spearmanr(predicted_scores, human_scores)
        kendall_tau, kendall_p = kendalltau(predicted_scores, human_scores)
        correlation_results = {
            'Pearson': (round(pearson_r, 3), round(pearson_p, 3)),
            'Spearman': (round(spearman_r, 3), round(spearman_p, 3)),
            'Kendall': (round(kendall_tau, 3), round(kendall_p, 3)),
        }
        return correlation_results

    def _get_scores(self, eval_data):
        predicted_scores = []
        human_scores = []
        for sample in eval_data:
            predicted_score = self.metric_model.get_score(sample)
            human_score = sample['human_score']
            predicted_scores.append(predicted_score)
            human_scores.append(human_score)
        return predicted_scores, human_scores

    def _save_txt(self, content: list, output_file_path):
        if self.console_output:
            print('Saving text into {}...'.format(output_file_path))
        with open(output_file_path, 'w') as f:
            for element in content:
                f.write(str(element) + '\n')

    def _save_distribution(self, content: List[float], output_file_path):
        if self.console_output:
            print('Saving distribution into {}...'.format(output_file_path))
        content = [round(element, 8) for element in content]
        sns.set_style('darkgrid')
        plt.figure()
        ax = sns.histplot(content, kde=True)
        ax.set_xlabel('score')
        ax.set_ylabel('count')
        plt.savefig(output_file_path)


class TableRecorder:

    def __init__(self, table_header: list, table_names: List[str]):
        self.table_dict = {
            name: pt.PrettyTable(table_header) for name in table_names
        }

    def add_row(self, table_row: list, table_name: str):
        self.table_dict[table_name].add_row(table_row)

    def save(self, output_file_path: str):
        with open(output_file_path, 'w') as f:
            for table_name, table in self.table_dict.items():
                f.write(table_name + '\n')
                f.write(str(table) + '\n\n')
