"""Dialog data processor base classes."""

import os
from typing import List, Dict
import json

import numpy as np
from tqdm import tqdm
import prettytable as pt
from transformers import AutoTokenizer


class DialogDataProcessor:
    """Base class for all dialog data processors. A dialog data processor loads
    and processes the dialog data of a specific dialog dataset.

    Attributes:
        dataset_name: The dataset name.
        split_file_paths: A Dict whose keys are split names
            and values are the corresponding split file paths.
        output_dir_path: The output directory path of the processed dialog data.
        max_seq_length: The max sequence length of the context-response pair.
        utterance_separator: A symbol used for separating utterances.
    """

    def __init__(self,
                 dataset_name: str,
                 split_file_paths: Dict[str, str],
                 output_dir_path: str,
                 max_seq_length: int,
                 utterance_separator: str = '|||'):
        self.dataset_name = dataset_name
        self.split_file_paths = split_file_paths
        self.output_dir_path = output_dir_path
        self.max_seq_length = max_seq_length
        self.utterance_separator = utterance_separator
        self.cur_split_name = None
        self.cur_file_path = None

    def prepare(self) -> None:
        for split_name, file_path in self.split_file_paths.items():
            self.cur_split_name = split_name
            self.cur_file_path = file_path
            self._load_data()
            self._process_data()
            self._save_data()

    def analyze(self) -> None:
        for split_name, file_path in self.split_file_paths.items():
            print('------ analyze {} data in {} ------'.format(split_name,
                                                               file_path))
            self.cur_split_name = split_name
            self.cur_file_path = file_path
            self._load_data()
            self._analyze_data()

    def dialog2str(self, dialog: List[str]) -> str:
        return self.utterance_separator.join(dialog)

    def _load_data(self):
        raise NotImplementedError

    def _process_data(self):
        raise NotImplementedError

    def _save_data(self):
        raise NotImplementedError

    def _analyze_data(self):
        raise NotImplementedError

    @property
    def cur_split_dir_path(self):
        return os.path.join(self.output_dir_path, self.cur_split_name)

    @staticmethod
    def load_from_json(data_path: str) -> list:
        """Loads dialog data from json file.

        Args:
            data_path: The file path of the raw dialog data.

        Returns:
            dialog_data: A list containing the raw dialog data.
        """
        try:
            with open(data_path, 'r') as f:
                dialog_data = json.load(f)
        except json.JSONDecodeError:
            # print('Fail to load {} with json.load().'.format(data_path))
            # print('Try to load it line by line with json.loads()...')
            with open(data_path, 'r') as f:
                dialog_data = [json.loads(line) for line in f.readlines()]
        return dialog_data

    @staticmethod
    def maybe_create_dir(dir_path: str) -> bool:
        """Creates directory if it does not exist.

        Args:
            dir_path: The path of the directory needed to be created.

        Returns:
            bool: Whether a new directory is created.
        """
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
            return True
        return False


class DailyDialogPlusPlusProcessor(DialogDataProcessor):
    """Dialog data processor of the DailyDialog++ dataset.

    Attributes:
        input_dir_path: The input directory path of the raw dialog data.
        output_dir_path: The output directory path of the processed dialog data.
        response_types: A list of desired response types. The responses
            whose types are not in this list will be excluded.
        max_seq_length: The max sequence length of the context-response pair.
        pretrained_model_name: A str indicating the pretrained model adopted
            for initializing the tokenizer.
    """

    num_res_per_dialog = 5

    def __init__(self,
                 input_dir_path: str,
                 output_dir_path: str,
                 response_types: List[str],
                 max_seq_length: int,
                 pretrained_model_name='bert-base-uncased'):
        split_file_paths = {
            'train': os.path.join(input_dir_path, 'train.json'),
            'validation': os.path.join(input_dir_path, 'dev.json'),
            'test': os.path.join(input_dir_path, 'test.json'),
        }
        super().__init__(dataset_name='DailyDialog++',
                         split_file_paths=split_file_paths,
                         output_dir_path=output_dir_path,
                         max_seq_length=max_seq_length)

        self.response_types = response_types
        self.raw_dialog_data = None
        self.processed_dialog_data = None
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    def prepare(self):
        super().prepare()
        self._save_feature_types()

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

    def _load_data(self):
        print('Loading data ({})......'.format(self.cur_file_path))
        self.raw_dialog_data = self.load_from_json(self.cur_file_path)

    def _process_data(self):
        print('Processing data......')
        self.processed_dialog_data = []
        for dialog in tqdm(self.raw_dialog_data):
            processed_dialog = {}
            processed_dialog['context'] = dialog['context']
            for res_type in self.response_types:
                processed_dialog[res_type] = dialog[res_type]
            self.processed_dialog_data.append(processed_dialog)

    def _save_data(self):
        print('Saving data......')
        self._create_output_dir()
        self._save_dialog_data_in_text_form()
        self._save_dialog_data_in_binary_form()

    def _analyze_data(self):
        print('Analyzing data......')
        # analyzes contexts
        contexts = [dialog['context'] for dialog in self.raw_dialog_data]
        num_contexts = len(contexts)
        num_utter_in_contexts = [len(context) for context in contexts]
        avg_num_utter_per_ctx = np.mean(num_utter_in_contexts)
        min_num_utter_per_ctx = min(num_utter_in_contexts)
        max_num_utter_per_ctx = max(num_utter_in_contexts)
        table = pt.PrettyTable(['type', 'count'])
        table.add_row(['#contexts', num_contexts])
        table.add_row(['Avg.#utterances_per_context',
                      '%.2f' % avg_num_utter_per_ctx])
        table.add_row(['Min.#utterances_per_context', min_num_utter_per_ctx])
        table.add_row(['Max.#utterances_per_context', max_num_utter_per_ctx])
        print('Statistics (contexts)')
        print(table)
        # analyzes responses
        for res_type in self.response_types:
            responses = [dialog[res_type] for dialog in self.raw_dialog_data]
            num_responses = [len(res) for res in responses]
            avg_num_res_per_dialog = np.mean(num_responses)
            min_num_res_per_dialog = min(num_responses)
            max_num_res_per_dialog = max(num_responses)
            table = pt.PrettyTable(['type', 'count'])
            table.add_row(['Avg.#responses_per_dialog',
                        '%.2f' % avg_num_res_per_dialog])
            table.add_row(['Min.#responses_per_dialog', min_num_res_per_dialog])
            table.add_row(['Max.#responses_per_dialog', max_num_res_per_dialog])
            print('Statistics ({})'.format(res_type))
            print(table)
        # analyzes context-response pairs
        ctx_res_token_list = []
        for dialog in self.raw_dialog_data:
            ctx_tokens = (' '.join(dialog['context'])).split()
            for res_type in self.response_types:
                for res in dialog[res_type]:
                    res_tokens = res.split()
                    ctx_res_tokens = ctx_tokens + res_tokens
                    ctx_res_token_list.append(ctx_res_tokens)
        num_ctx_res_tokens = [len(ctx_res_tokens)
                              for ctx_res_tokens in ctx_res_token_list]
        avg_num_ctx_res_tokens = np.mean(num_ctx_res_tokens)
        min_num_ctx_res_tokens = min(num_ctx_res_tokens)
        max_num_ctx_res_tokens = max(num_ctx_res_tokens)
        num_ctx_res_tokens_larger_than_max_len = len(
            [num for num in num_ctx_res_tokens if num >= self.max_seq_length])
        table = pt.PrettyTable(['type', 'count'])
        table.add_row(['#ctx_res_pairs', len(ctx_res_token_list)])
        table.add_row(['Avg.#ctx_res_tokens',
                    '%.2f' % avg_num_ctx_res_tokens])
        table.add_row(['Min.#ctx_res_tokens', min_num_ctx_res_tokens])
        table.add_row(['Max.#ctx_res_tokens', max_num_ctx_res_tokens])
        table.add_row(['#ctx_res_tokens(len>{})'.format(self.max_seq_length),
                       num_ctx_res_tokens_larger_than_max_len])
        print('Statistics (context-response pairs)')
        print(table)

    def _save_feature_types(self):
        output_feature_type_path = os.path.join(
            self.output_dir_path, 'feature_types.json')
        with open(output_feature_type_path, 'w') as f:
            json.dump(self.feature_types, f, indent=4)

    def _create_output_dir(self):
        raise NotImplementedError

    def _save_dialog_data_in_text_form(self):
        raise NotImplementedError

    def _save_dialog_data_in_binary_form(self):
        raise NotImplementedError

    @property
    def feature_types(self):
        raise NotImplementedError
