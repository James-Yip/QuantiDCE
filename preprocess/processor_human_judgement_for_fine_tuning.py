import os
import sys
import json
from typing import List
from random import shuffle

import texar.torch as tx
from transformers import AutoTokenizer

sys.path.append('..')
from dataset.human_judgement import HumanJudgementDataset


class HumanJudgementForFineTuningProcessor:
    """Data processor of the human judgement dataset.
    """

    TRAIN = 'train'
    VALID = 'validation'

    def __init__(self,
                 input_dir_path: str,
                 output_dir_path: str,
                 dataset_name: str,
                 max_seq_length: int,
                 utterance_separator: str = '|||',
                 pretrained_model_name='bert-base-uncased',
                 train_data_ratio=0.9):
        self.input_dir_path = input_dir_path
        self.output_dir_path = output_dir_path
        self.dataset_name = dataset_name
        self.max_seq_length = max_seq_length
        self.utterance_separator = utterance_separator
        self.train_data_ratio = train_data_ratio

        self.raw_data = []
        self.splited_data = {}

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    def prepare(self) -> None:
        self._load_data()
        self._split_data()
        for split_name in [self.TRAIN, self.VALID]:
            self.cur_split_name = split_name
            self._create_output_dir()
            self._save_data_in_text_form()
            self._save_data_in_binary_form()
        self._save_feature_types()
        print('Done.')

    def _load_data(self):
        dataset = HumanJudgementDataset(
            self.input_dir_path, self.dataset_name)
        self.raw_data = dataset.data_list

    def _split_data(self):
        self.splited_data = {}
        num_train_data = int(len(self.raw_data) * self.train_data_ratio)
        shuffle(self.raw_data)
        train_data = self.raw_data[:num_train_data]
        valid_data = self.raw_data[num_train_data:]
        self.splited_data[self.TRAIN] = train_data
        self.splited_data[self.VALID] = valid_data

    def dialog2str(self, dialog: List[str]) -> str:
        return self.utterance_separator.join(dialog)

    def _create_output_dir(self):
        if not os.path.isdir(self.cur_split_dir_path):
            os.makedirs(self.cur_split_dir_path)

    def _save_data_in_text_form(self):
        output_file_path = os.path.join(self.cur_split_dir_path,
                                        'context_response.text')
        print('Saving text data into {}...'.format(output_file_path))
        with open(output_file_path, 'w') as f:
            for sample in self.splited_data[self.cur_split_name]:
                human_score = sample['human_score']
                ctx_res_list = sample['context'] + [sample['hyp_response']]
                ctx_res_str = self.dialog2str(ctx_res_list)
                f.write('{}\t {}\n'.format(human_score, ctx_res_str))

    def _save_data_in_binary_form(self):
        output_file_path = os.path.join(self.cur_split_dir_path,
                                        'context_response.pkl')
        print('Saving binary data into {}...'.format(output_file_path))
        with tx.data.RecordData.writer(
            output_file_path, self.feature_types) as writer:
            for sample in self.splited_data[self.cur_split_name]:
                ctx = sample['context']
                res = sample['hyp_response']
                output = self.encode_ctx_res_pair(ctx, res)
                input_ids, token_type_ids, attention_mask = output
                human_score = sample['human_score']
                features = {
                    'input_ids': input_ids,
                    'token_type_ids': token_type_ids,
                    'attention_mask': attention_mask,
                    'human_score': human_score,
                }
                writer.write(features)

    def _save_feature_types(self):
        output_file_path = os.path.join(
            self.output_dir_path, 'feature_types.json')
        print('Saving feature types into {}...'.format(output_file_path))
        with open(output_file_path, 'w') as f:
            json.dump(self.feature_types, f, indent=4)

    def encode_ctx_res_pair(self, context: List[str], response: str):
        """Encodes the given context-response pair into ids.
        """
        context = ' '.join(context)
        tokenizer_outputs = self.tokenizer(
            text=context, text_pair=response, truncation=True,
            padding='max_length', max_length=self.max_seq_length)
        input_ids = tokenizer_outputs['input_ids']
        token_type_ids = tokenizer_outputs['token_type_ids']
        attention_mask = tokenizer_outputs['attention_mask']

        # tokenized_text = self.tokenizer.convert_ids_to_tokens(input_ids)
        # print('context: ', context)
        # print('response: ', response)
        # print('tokenizer_outputs: ', tokenizer_outputs)
        # print('tokenized_text: ', ' '.join(tokenized_text))
        # print('length: ', len(tokenized_text))
        # exit()
        return input_ids, token_type_ids, attention_mask

    @property
    def feature_types(self):
        feature_types = {
            'input_ids': ['int64', 'stacked_tensor', self.max_seq_length],
            'token_type_ids': ['int64', 'stacked_tensor', self.max_seq_length],
            'attention_mask': ['int64', 'stacked_tensor', self.max_seq_length],
            'human_score': ['float32', 'stacked_tensor'],
        }
        return feature_types

    @property
    def cur_split_dir_path(self):
        return os.path.join(self.output_dir_path, self.cur_split_name)
