import os
from typing import List
import json

from tqdm import tqdm
import texar.torch as tx

from processor_base import DailyDialogPlusPlusProcessor


class DailyDialogPlusPlusMLRLossProcessor(DailyDialogPlusPlusProcessor):
    """Dialog data processor of the DailyDialog++ dataset (MLR Loss version).
    Processes the data for training the model with Multi-Ranking loss.

    The attribute descriptions are in the `DailyDialogPlusPlusProcessor` class.
    """

    def __init__(self,
                 input_dir_path: str,
                 output_dir_path: str,
                 response_types: List[str],
                 max_seq_length: int):
        super().__init__(input_dir_path=input_dir_path,
                         output_dir_path=output_dir_path,
                         response_types=response_types,
                         max_seq_length=max_seq_length)

    def _create_output_dir(self):
        self.maybe_create_dir(self.cur_split_dir_path)
        for pair_path in self.cur_pair_dir_paths.values():
            self.maybe_create_dir(pair_path)

    def _save_dialog_data_in_text_form(self):
        print('\tSaving data in text form...')
        for res_type, pair_path in self.cur_pair_dir_paths.items():
            output_file_path = os.path.join(pair_path,
                                            'context_response.text')
            with open(output_file_path, 'w') as f:
                for dialog in tqdm(self.processed_dialog_data):
                    ctx = dialog['context']
                    responses = dialog[res_type]
                    for res in responses:
                        ctx_res_list = ctx + [res]
                        ctx_res_str = self.dialog2str(ctx_res_list)
                        f.write(ctx_res_str + '\n')

    def _save_dialog_data_in_binary_form(self):
        print('\tSaving data in binary form...')
        for res_type, pair_path in self.cur_pair_dir_paths.items():
            output_file_path = os.path.join(pair_path, 'context_response.pkl')
            self._save_specific_type_of_dialog_data_in_binary_form(
                res_type,
                output_file_path)

    def _save_specific_type_of_dialog_data_in_binary_form(
        self, res_type, output_path):
        with tx.data.RecordData.writer(
            output_path, self.feature_types) as writer:
            for dialog in tqdm(self.processed_dialog_data):
                features = {}
                ctx = dialog['context']
                responses = dialog[res_type]
                for i in range(self.num_res_per_dialog):
                    res = responses[i]
                    output = self.encode_ctx_res_pair(ctx, res)
                    input_ids, token_type_ids, attention_mask = output
                    input_key = 'input_ids_{}'.format(i)
                    token_type_key = 'token_type_ids_{}'.format(i)
                    attention_mask_key = 'attention_mask_{}'.format(i)
                    features[input_key] = input_ids
                    features[token_type_key] = token_type_ids
                    features[attention_mask_key] = attention_mask
                writer.write(features)

    @property
    def cur_pair_dir_paths(self):
        pair_dir_paths = {}
        for idx, res_type in enumerate(self.response_types):
            pair_name = 'pair_{}'.format(idx + 1)
            value = os.path.join(self.cur_split_dir_path, pair_name)
            pair_dir_paths[res_type] = value
        return pair_dir_paths

    @property
    def feature_types(self):
        feature_types = {}
        for i in range(self.num_res_per_dialog):
            input_key = 'input_ids_{}'.format(i)
            token_type_key = 'token_type_ids_{}'.format(i)
            attention_mask_key = 'attention_mask_{}'.format(i)
            value = ['int64', 'stacked_tensor', self.max_seq_length]
            feature_types[input_key] = value
            feature_types[token_type_key] = value
            feature_types[attention_mask_key] = value
        return feature_types


if __name__ == '__main__':
    # test processor
    response_types = [
        'positive_responses',
        'adversarial_negative_responses',
        'random_negative_responses',
    ]
    processor = DailyDialogPlusPlusMLRLossProcessor(
        input_dir_path='./dataset/dailydialog++/',
        output_dir_path='../data/dailydialog_plusplus_mlr',
        response_types=response_types,
        max_seq_length=128)
    processor.analyze()
    processor.prepare()
