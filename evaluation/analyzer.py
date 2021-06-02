import os
from typing import List
import json

import prettytable as pt
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    import sys
    sys.path.append('..')
from dataset.human_judgement import HumanJudgementDataset


class Analyzer:
    """#TODO: adds docstring
    """

    def __init__(self, datasets: List[str], eval_data_dir_path: str):
        self.datasets = datasets
        self.eval_data_dir_path = eval_data_dir_path

    def analyze(self, mode):
        if mode == 'mix':
            self._analyze_mix()
        elif mode == 'separate':
            self._analyze_sep()

    def _analyze_mix(self):
        """Analyzes the mix data.
        """
        for dataset_name in self.datasets:
            print('\nAnalyzing {}...'.format(dataset_name))
            auto_score_file_dict = {
                'GRADE_score': 'grade_score.txt',
                'QuantiDCE_score': 'mlr_score.txt'
            }
            eval_data = HumanJudgementDataset(
                self.eval_data_dir_path, dataset_name,
                auto_score_file_dict=auto_score_file_dict)

            output_dir_path = os.path.join(
                self.eval_data_dir_path, dataset_name)

            categorized_data_dict = eval_data.categorize_data()
            table = pt.PrettyTable(['Score Range', 'Count'])
            for key, value in categorized_data_dict.items():
                table.add_row([key, len(value)])
            print(table)
            categorized_data_name = 'categorized_{}_mix.json'.format(
                dataset_name)
            categorized_data_path = os.path.join(
                output_dir_path, categorized_data_name)
            self._save_json(categorized_data_dict, categorized_data_path)

            human_scores = eval_data.get_human_scores()
            human_score_dist_name = 'human_score_dist_{}_mix.png'.format(
                dataset_name)
            human_score_dist_path = os.path.join(
                output_dir_path, human_score_dist_name)
            self._save_distribution(human_scores, human_score_dist_path)

    def _analyze_sep(self):
        """Analyzes the separate data.
        """
        for dataset_name in self.datasets:
            dialog_model_names = HumanJudgementDataset.DATA_MAP[dataset_name]
            for dialog_model_name in dialog_model_names:
                print('\nAnalyzing {}-{}...'.format(dataset_name,
                                                  dialog_model_name))
                eval_data = HumanJudgementDataset(
                    self.eval_data_dir_path,
                    dataset_name, dialog_model_name)

                output_dir_path = os.path.join(
                    self.eval_data_dir_path,
                    dataset_name, dialog_model_name)

                categorized_data_dict = eval_data.categorize_data()
                table = pt.PrettyTable(['Score Range', 'Count'])
                for key, value in categorized_data_dict.items():
                    table.add_row([key, len(value)])
                print(table)
                categorized_data_name = 'categorized_{}_{}.json'.format(
                    dataset_name, dialog_model_name)
                categorized_data_path = os.path.join(
                    output_dir_path, categorized_data_name)
                self._save_json(categorized_data_dict, categorized_data_path)

                human_scores = eval_data.get_human_scores()
                human_score_dist_name = 'human_score_dist_{}_{}.png'.format(
                    dataset_name, dialog_model_name)
                human_score_dist_path = os.path.join(
                    output_dir_path, human_score_dist_name)
                self._save_distribution(human_scores, human_score_dist_path)

    def _save_json(self, content: dict, output_file_path):
        print('Saving json into {}...'.format(output_file_path))
        with open(output_file_path, 'w') as f:
            json.dump(content, f, indent=4, ensure_ascii=False)

    def _save_distribution(self, content: List[float], output_file_path):
        print('Saving distribution into {}...'.format(output_file_path))
        sns.set_style('darkgrid')
        plt.figure()
        ax = sns.histplot(content, kde=True)
        ax.set_xlabel('score')
        ax.set_ylabel('count')
        plt.savefig(output_file_path)


if __name__ == '__main__':
    datasets = ['dailydialog_EVAL', 'convai2', 'empatheticdialogues']
    eval_data_dir_path = './eval_data'
    analyzer = Analyzer(datasets, eval_data_dir_path)
    analyzer.analyze(mode='mix')
