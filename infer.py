"""An example about how to use QuantiDCE."""

import util.opt as opt
import util.main_utils as main_utils
from evaluation.evaluator import Evaluator


if __name__ == '__main__':
    args = opt.parse_eval_opt()
    model = main_utils.get_model(args)
    context = [
        'I need to book a plane ticket to London.',
        'Round-trip or one-way?',
    ]
    response = 'Round trip or one way trip?'
    score = model.get_score(context, response)
    score = round(score * 4 + 1, 2)
    print('context:', context)
    print('response:', response)
    print('QuantiDCE score:', score)
