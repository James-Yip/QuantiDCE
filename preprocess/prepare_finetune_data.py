"""Loads and pre-processes the original dialog data."""

import os
import argparse

from processor_human_judgement_for_fine_tuning import HumanJudgementForFineTuningProcessor


# ==================================================================
# Functions
# ==================================================================
def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data', '-d',
        choices=['dailydialog_EVAL', 'convai2', 'empatheticdialogues'],
        default='dailydialog_EVAL',
        help='The data used for fine-tuning.')
    parser.add_argument(
        '--max_seq_length',
        type=int,
        default=128,
        help='The max sequence length of the context-response pair.')

    args = parser.parse_args()
    return args


# ==================================================================
# Main
# ==================================================================
if __name__ == '__main__':
    args = parse_opt()
    input_dir_path = '../evaluation/eval_data'
    output_dir_path = os.path.join(
        '../data/human_judgement_for_fine_tuning', args.data)
    processor = HumanJudgementForFineTuningProcessor(
        input_dir_path=input_dir_path,
        output_dir_path=output_dir_path,
        dataset_name=args.data,
        max_seq_length=args.max_seq_length)
    processor.prepare()
