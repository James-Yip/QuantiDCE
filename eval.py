"""Startup script for evaluating metric models."""

import util.opt as opt
import util.main_utils as main_utils
from evaluation.evaluator import Evaluator


if __name__ == '__main__':
    args = opt.parse_eval_opt()
    evaluator = Evaluator(
        checkpoint_dir_path=args.checkpoint_dir_path,
        eval_data_dir_path=args.eval_data_dir_path,
        datasets=args.datasets,
        eval_mode=args.eval_mode)
    model = main_utils.get_model(args)
    evaluator.evaluate(model, additional_eval_info=args.additional_eval_info)
