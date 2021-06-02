"""Startup script for fine-tuning metric models."""

import util.opt as opt
import util.main_utils as main_utils


if __name__ == '__main__':
    args = opt.parse_finetune_opt()
    main_utils.set_seed(args.seed)
    model = main_utils.get_model(args)
    dataset = main_utils.get_dataset(args)
    trainer = main_utils.get_trainer(model, dataset, args)
    trainer.run()
