import os
import json
import functools
import copy
from typing import List, Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm
import texar.torch as tx

from trainer.trainer_base import Trainer


class KnowledgeDistillationFineTuningTrainer(Trainer):
    """#TODO
    """

    def __init__(self, model, data, args):
        super().__init__(model, data, args)
        self.warmup_proportion = args.warmup_proportion
        self.kd_loss_weight = args.kd_loss_weight
        self.finetune_loss_weight = args.finetune_loss_weight

        self.student_model = model
        self.teacher_model = copy.deepcopy(self.student_model)

        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()

        self.mse_criterion = torch.nn.MSELoss()

    def _get_optimizer(self):
        vars_with_decay = []
        vars_without_decay = []
        # for name, param in self.student_model.mlp.named_parameters():
        for name, param in self.student_model.named_parameters():
            if 'layer_norm' in name or name.endswith('bias'):
                vars_without_decay.append(param)
            else:
                vars_with_decay.append(param)

        opt_params = [{
            'params': vars_with_decay,
            'weight_decay': 0.01,
        }, {
            'params': vars_without_decay,
            'weight_decay': 0.0,
        }]

        optimizer = tx.core.BertAdam(
            opt_params,
            betas=(0.9, 0.999),
            eps=1e-6,
            lr=self.learning_rate
        )
        return optimizer

    def _get_scheduler(self):
        def get_lr_multiplier(step: int, total_steps: int, warmup_steps: int):
            """Calculate the learning rate multiplier given current step and the
            number of warm-up steps. The learning rate schedule follows a linear
            warm-up and linear decay.
            """
            step = min(step, total_steps)

            multiplier = (1 - (step-warmup_steps) / (total_steps-warmup_steps))

            if warmup_steps > 0 and step < warmup_steps:
                warmup_percent_done = step / warmup_steps
                multiplier = warmup_percent_done

            return multiplier

        lr_func = functools.partial(get_lr_multiplier,
                                    total_steps=self.num_total_train_steps,
                                    warmup_steps=self.num_warmup_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_func)
        return scheduler

    def _train_epoch(self):
        self._switch_mode(Trainer.TRAIN)
        avg_recorder = tx.utils.AverageRecorder(size=self.display_steps)
        pbar = tqdm(self.dataset, total=self.num_train_steps_per_epoch)
        pbar.set_description('{}-epoch{}'.format(Trainer.TRAIN,
                                                 self.cur_epoch_id))
        for batch in pbar:
            # forward
            student_output_dict, student_score = self.student_model(
                batch['input_ids'],
                batch['token_type_ids'],
                batch['attention_mask'])
            teacher_output_dict, teacher_score = self.teacher_model(
                batch['input_ids'],
                batch['token_type_ids'],
                batch['attention_mask'])
            loss, loss_info_dict = self._compute_kd_finetune_loss(
                student_output_dict=student_output_dict,
                teacher_output_dict=teacher_output_dict,
                student_score=student_score,
                teacher_score=teacher_score,
                target_score=batch['human_score'])
            # backward
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            # updates logging information
            batch_size = self.get_batch_size(batch)
            avg_recorder.add(loss_info_dict, batch_size)
            step = self.global_step[Trainer.TRAIN]
            self.global_step[Trainer.TRAIN] += 1
            if self._is_time_to_display(step):
                avg_loss_info_dict = avg_recorder.avg()
                lr_info_dict = {'lr': self.cur_lr}
                train_info_dict = dict(avg_loss_info_dict, **lr_info_dict)
                train_info = self._get_train_info(train_info_dict)
                self.logger.info(train_info)
            for loss_name, loss_value in loss_info_dict.items():
                self.tensorboard_writer.add_scalar(
                    '{}/train'.format(loss_name), loss_value, step)

    @torch.no_grad()
    def _eval_epoch(self, mode):
        self._switch_mode(mode)
        avg_recorder = tx.utils.AverageRecorder()
        pbar = tqdm(self.dataset, total=self.num_eval_steps_per_epoch)
        pbar.set_description('{}-epoch{}'.format(mode, self.cur_epoch_id))
        for batch in pbar:
            # forward
            student_output_dict, student_score = self.student_model(
                batch['input_ids'],
                batch['token_type_ids'],
                batch['attention_mask'])
            teacher_output_dict, teacher_score = self.teacher_model(
                batch['input_ids'],
                batch['token_type_ids'],
                batch['attention_mask'])
            _, loss_info_dict = self._compute_kd_finetune_loss(
                student_output_dict=student_output_dict,
                teacher_output_dict=teacher_output_dict,
                student_score=student_score,
                teacher_score=teacher_score,
                target_score=batch['human_score'])
            # updates logging information
            batch_size = self.get_batch_size(batch)
            avg_recorder.add(loss_info_dict, batch_size)
            step = self.global_step[mode]
            self.global_step[mode] += 1
            if mode == Trainer.VALIDATION:
                for loss_name, loss_value in loss_info_dict.items():
                    self.tensorboard_writer.add_scalar(
                        '{}/valid'.format(loss_name), loss_value, step)
        eval_results = self._get_metric_results(avg_recorder)
        return eval_results

    def run(self):
        self._before_training()
        self._train()
        self._after_training()

    def _compute_kd_finetune_loss(self,
                                  student_output_dict: dict,
                                  teacher_output_dict: dict,
                                  student_score: torch.Tensor,
                                  teacher_score: torch.Tensor,
                                  target_score: torch.Tensor):
        """#TODO
        """
        kd_loss = self._compute_kd_loss(
            student_output_dict=student_output_dict,
            teacher_output_dict=teacher_output_dict,
            student_score=student_score,
            teacher_score=teacher_score)
        finetune_loss = self._compute_finetune_loss(
            predicted_score=student_score,
            target_score=target_score)
        kd_finetune_loss = self.kd_loss_weight * kd_loss + \
                           self.finetune_loss_weight * finetune_loss
        loss_info_dict = {
            'kd_finetune_loss': kd_finetune_loss.item(),
            'kd_loss': kd_loss.item(),
            'finetune_loss': finetune_loss.item(),
        }
        return kd_finetune_loss, loss_info_dict

    def _compute_kd_loss(self,
                         student_output_dict, teacher_output_dict,
                         student_score, teacher_score):
        # hidden_states: A tuple containing #hidden_layer+1 tensors.
        #   Each tensor's size: [batch_size, #hidden_layer+1, seq_len, hidden_size]
        # attentions: a tuple containing #hidden_layer tensors.
        #   Each tensor's size: [batch_size, #hidden_layer, #heads, seq_len, seq_len]

        hidden_loss = 0.
        s_hidden_states = student_output_dict['hidden_states']
        t_hidden_states = teacher_output_dict['hidden_states']
        for s_hidden_state, t_hidden_state in zip(s_hidden_states,
                                                  t_hidden_states):
            t_hidden_state = t_hidden_state.detach()
            hidden_loss += self.mse_criterion(s_hidden_state, t_hidden_state)

        attention_loss = 0.
        s_attentions = student_output_dict['attentions']
        t_attentions = teacher_output_dict['attentions']
        for s_attention, t_attention in zip(s_attentions, t_attentions):
            s_attention = torch.where(s_attention <= -1e2,
                                      torch.zeros_like(s_attention),
                                      s_attention)
            t_attention = torch.where(t_attention <= -1e2,
                                      torch.zeros_like(t_attention),
                                      t_attention)
            t_attention = t_attention.detach()
            attention_loss += self.mse_criterion(s_attention, t_attention)

        prediction_loss = 0.
        teacher_score = teacher_score.detach()
        prediction_loss += self.mse_criterion(student_score, teacher_score)

        kd_loss = hidden_loss + attention_loss + prediction_loss
        return kd_loss

    def _compute_finetune_loss(self,
                               predicted_score: torch.Tensor,
                               target_score: torch.Tensor):
        predicted_score = predicted_score * 4 + 1
        predicted_score = predicted_score.squeeze(1)
        finetune_loss = self.mse_criterion(predicted_score, target_score)
        return finetune_loss

    def _get_metric_results(self, avg_recorder=None):
        """#TODO
        """
        if avg_recorder:
            avg_dual_fat_loss = avg_recorder.avg('kd_finetune_loss')
            metric_results = {
                'kd_finetune_loss': avg_dual_fat_loss,
            }
        else:
            metric_results = {}
        return metric_results

    def _get_train_info(self, info_dict):
        train_info_head = '[epoch{},step{}] '.format(
            self.cur_epoch_id, self.global_step[Trainer.TRAIN])
        train_info_body = json.dumps(info_dict)
        train_info = train_info_head + train_info_body
        return train_info

    @property
    def cur_lr(self):
        return self.scheduler.optimizer.param_groups[0]['lr']

    @property
    def num_warmup_steps(self):
        return self.num_train_steps_per_epoch * self.warmup_proportion

    @staticmethod
    def get_batch_size(batch: Dict[str, torch.Tensor]):
        tensor = list(batch.values())[0]
        return tensor.size(0)
