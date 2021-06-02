"""Loads and pre-processes the original dialog data."""

import os
import argparse
import importlib


# ==================================================================
# Constants
# ==================================================================
ALL_RESPONSE_TYPES = [
    'positive_responses',
    'adversarial_negative_responses',
    'random_negative_responses',
]

CLASS_NAME_MAP = {
    'processor_dailydialog_plusplus_mlr':
        'DailyDialogPlusPlusMLRLossProcessor',
}


# ==================================================================
# Functions
# ==================================================================
def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--src_dataset', '-sd',
        choices=['dailydialog++'],
        default='dailydialog++',
        help='Source raw dataset name.')
    parser.add_argument(
        '--tgt_dataset', '-td',
        choices=['dailydialog_plusplus_mlr'],
        default='dailydialog_plusplus_mlr',
        help='Target processed dataset name.')
    parser.add_argument(
        '--response_types', '-rt',
        choices=['1+2+3', '1+2', '1+3'],
        default='1+2+3',
        help='Desired response types.')
    parser.add_argument(
        '--max_seq_length',
        type=int,
        default=128,
        help='The max sequence length of the context-response pair.')

    args = parser.parse_args()
    return args

def get_response_types(response_types_str):
    response_types = []
    for type_str in response_types_str.split('+'):
        type_idx = int(type_str) - 1
        response_type = ALL_RESPONSE_TYPES[type_idx]
        response_types.append(response_type)
    return response_types

def get_processor_class(target_dataset_name):
    processor_module_name = 'processor_{}'.format(target_dataset_name)
    processor_class_name = CLASS_NAME_MAP[processor_module_name]
    processor_module = importlib.import_module(processor_module_name)
    Processor = getattr(processor_module, processor_class_name)
    return Processor


# ==================================================================
# Main
# ==================================================================
if __name__ == '__main__':
    args = parse_opt()
    input_dir_path = os.path.join('./dataset', args.src_dataset)
    output_dir_path = os.path.join('../data', args.tgt_dataset)
    response_types = get_response_types(args.response_types)
    Processor = get_processor_class(args.tgt_dataset)
    processor = Processor(
        input_dir_path=input_dir_path,
        output_dir_path=output_dir_path,
        response_types=response_types,
        max_seq_length=args.max_seq_length)
    processor.prepare()
    # processor.analyze()
