"""Provides utility functions for main."""

import os
import sys
import importlib
import random

import torch
import numpy as np


CLASS_NAME_MAP = {
    # Models
    'bert_metric': 'BERTMetric',
    # Datasets
    'dailydialog_plusplus_mlr': 'DailyDialogPlusPlusMLR',
    'human_judgement_for_fine_tuning': 'HumanJudgementForFineTuning',
    # Trainers
    'trainer_mlr_pretrain': 'MultiLevelRankingLossTrainer',
    'trainer_kd_finetune': 'KnowledgeDistillationFineTuningTrainer',
}


def add_module_search_paths(search_paths: list) -> None:
    """Adds paths for searching python modules.
    """
    # TODO: check if the python files in the search paths have the same names
    augmented_search_paths = search_paths
    for path in search_paths:
        for root, dirs, _ in os.walk(path):
            cur_dir_paths = [os.path.join(root, cur_dir) for cur_dir in dirs]
            augmented_search_paths.extend(cur_dir_paths)
    sys.path.extend(augmented_search_paths)

def get_class(module_name):
    class_name = CLASS_NAME_MAP[module_name]
    module = importlib.import_module(module_name)
    Class = getattr(module, class_name)
    return Class

def get_model(args):
    add_module_search_paths(['./model'])
    Model = get_class(args.model)
    model = Model(args)
    return model

def get_dataset(args):
    add_module_search_paths(['./dataset'])
    Dataset = get_class(args.dataset)
    dataset = Dataset(args)
    return dataset

def get_trainer(model, data, args):
    add_module_search_paths(['./trainer'])
    trainer_module_name = 'trainer_{}'.format(args.trainer)
    Trainer = get_class(trainer_module_name)
    trainer = Trainer(model, data, args)
    return trainer

def set_seed(seed):
    """Fixes randomness to enable reproducibility.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
