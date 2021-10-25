import os
from typing import List

import torch
from torch import nn
import texar.torch as tx
# from texar.torch.modules import BERTEncoder
from transformers import AutoTokenizer, AutoModel

from util.config_base import Config


class BERTMetric(nn.Module):

    NAME = 'bert_metric'

    def __init__(self, args):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name)
        self.max_seq_length = args.max_seq_length

        self.backbone = AutoModel.from_pretrained(args.pretrained_model_name)

        bert_hidden_size = self.backbone.config.hidden_size
        mlp_hidden_size_1 = int(bert_hidden_size / 2)
        mlp_hidden_size_2 = int(mlp_hidden_size_1 / 2)
        self.mlp = nn.Sequential(
            nn.Linear(bert_hidden_size, mlp_hidden_size_1),
            nn.ELU(),
            nn.Linear(mlp_hidden_size_1, mlp_hidden_size_2),
            nn.ELU(),
            nn.Linear(mlp_hidden_size_2, 1),
            nn.Sigmoid())

        if args.gpu:
            self.device = torch.device("cuda:{}".format(args.gpu))
            self.to(self.device)
            map_location = 'cuda:{}'.format(args.gpu)
        else:
            self.device = None
            map_location = None

        if hasattr(args, 'checkpoint_file_name'):
            # loads checkpoint
            checkpoint_file_path = os.path.join(
                args.checkpoint_dir_path, args.checkpoint_file_name)
            state_dict = torch.load(
                checkpoint_file_path,
                map_location=map_location)
            self.load_state_dict(state_dict)
            print('loading checkpoint from: {}'.format(checkpoint_file_path))

        if hasattr(args, 'pretrain_checkpoint_file_name'):
            # loads checkpoint
            checkpoint_file_path = os.path.join(
                args.pretrain_checkpoint_dir_path,
                args.pretrain_checkpoint_file_name)
            state_dict = torch.load(
                checkpoint_file_path,
                map_location=map_location)
            self.load_state_dict(state_dict)
            print('loading checkpoint from: {}'.format(checkpoint_file_path))

    def get_hidden_size(self):
        return self.backbone.config.hidden_size

    def forward(self, input_ids, token_type_ids, attention_mask):
        output_dict = self.backbone(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True)
        pooled_output = output_dict['pooler_output']
        score = self.mlp(pooled_output)
        return output_dict, score

    @torch.no_grad()
    def get_score(self, context: List[str], response: str):
        self.eval()
        input_ids, token_type_ids, attention_mask = self.encode_ctx_res_pair(
            context, response)
        _, score = self.forward(input_ids, token_type_ids, attention_mask)
        return score[0].item()

    def encode_ctx_res_pair(self, context: List[str], response: str):
        """Encodes the given context-response pair into ids.
        """
        context = ' '.join(context)
        tokenizer_outputs = self.tokenizer(
            text=context, text_pair=response,
            return_tensors='pt', truncation=True,
            padding='max_length', max_length=self.max_seq_length)
        input_ids = tokenizer_outputs['input_ids']
        token_type_ids = tokenizer_outputs['token_type_ids']
        attention_mask = tokenizer_outputs['attention_mask']

        input_ids = input_ids.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        assert input_ids.size() == torch.Size([1, self.max_seq_length])
        assert token_type_ids.size() == torch.Size([1, self.max_seq_length])
        assert attention_mask.size() == torch.Size([1, self.max_seq_length])
        # tokenized_text = self.tokenizer.convert_ids_to_tokens(input_ids)
        # print('context: ', context)
        # print('response: ', response)
        # print('tokenizer_outputs: ', tokenizer_outputs)
        # print('tokenized_text: ', ' '.join(tokenized_text))
        # print('length: ', len(tokenized_text))
        # exit()
        return input_ids, token_type_ids, attention_mask
