import argparse


def parse_eval_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--eval_data_dir_path',
        help='The path of the human judgement data directory.')
    parser.add_argument(
        '--eval_mode',
        choices=['mix', 'separate'],
        help='Evaluation mode (currently supports `mix` / `separate`).')
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='Human judgement datasets for evaluating metrics.')
    parser.add_argument(
        '--model',
        choices=['bert_metric'],
        help='Model name (currently supports `bert_metric`).')
    parser.add_argument(
        '--gpu',
        help='the id of GPU for loading model and data.')
    parser.add_argument(
        '--checkpoint_dir_path',
        help='The path of the output checkpoint directory.')
    parser.add_argument(
        '--checkpoint_file_name',
        help='The file name of the checkpoint to be loaded.')
    parser.add_argument(
        '--pretrained_model_name',
        choices=['bert-base-uncased'],
        help='The name of the pretrained model.')
    parser.add_argument(
        '--additional_eval_info',
        help='Additional evaluation information saved into the correlation '
             'result file and used for naming the predicted score files.')
    parser.add_argument(
        '--max_seq_length',
        type=int,
        default=128,
        help='The max sequence length of the context-response pair.')

    args = parser.parse_args()
    return args

def parse_pretrain_opt():
    parser = argparse.ArgumentParser()

    # general settings
    parser.add_argument(
        '--dataset',
        choices=['dailydialog_plusplus_mlr'],
        help='Dataset name (currently supports `dailydialog_plusplus_mlr`).')
    parser.add_argument(
        '--model',
        choices=['bert_metric'],
        help='Model name (currently supports `bert_metric`).')
    parser.add_argument(
        '--trainer',
        choices=['mlr_pretrain'],
        help='Trainer name (currently supports `mlr_pretrain`).')
    parser.add_argument(
        '--mode',
        choices=['train', 'test'],
        help='Runing mode (currently supports `train` / `test`).')
    parser.add_argument(
        '--gpu',
        help='the id of GPU for loading model and data.')
    parser.add_argument(
        '--seed',
        type=int,
        help='The seed for reproducibility (optional).')
    parser.add_argument(
        '--max_seq_length',
        type=int,
        default=128,
        help='The max sequence length of the context-response pair.')

    # data settings
    parser.add_argument(
        '--train_batch_size',
        type=int,
        help='The batch size for training.')
    parser.add_argument(
        '--eval_batch_size',
        type=int,
        help='The batch size for validation.')
    parser.add_argument(
        '--test_batch_size',
        type=int,
        help='The batch size for testing.')

    # trainer settings
    parser.add_argument(
        '--num_epochs',
        type=int,
        help='The number of epochs for training.')
    parser.add_argument(
        '--display_steps',
        type=int,
        help='Print training loss every display_steps.')
    parser.add_argument(
        '--update_steps',
        type=int,
        help='Update centroids every update_steps.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        help='The initial learning rate for training.')
    parser.add_argument(
        '--warmup_proportion',
        type=float,
        help='The warmup proportion of the warmup strategy for LR scheduling.')
    parser.add_argument(
        '--inter_distance_lower_bound',
        type=float,
        help='The lower bound of the inter-cluster distance.')
    parser.add_argument(
        '--intra_distance_upper_bound',
        type=float,
        help='The upper bound of the intra-cluster distance.')
    parser.add_argument(
        '--feature_distance_lower_bound',
        type=float,
        help='The lower bound of the inter-cluster feature distance.'
             '(for dual_mlr_loss computing)')
    parser.add_argument(
        '--feature_distance_upper_bound',
        type=float,
        help='The upper bound of the intra-cluster feature distance.'
             '(for dual_mlr_loss computing)')
    parser.add_argument(
        '--score_distance_lower_bound',
        type=float,
        help='The lower bound of the inter-cluster score distance.'
             '(for dual_mlr_loss computing)')
    parser.add_argument(
        '--score_distance_upper_bound',
        type=float,
        help='The upper bound of the intra-cluster score distance.'
             '(for dual_mlr_loss computing)')
    parser.add_argument(
        '--feature_loss_weight',
        type=float,
        default=1.,
        help='The loss weight for feature_mlr_loss.')
    parser.add_argument(
        '--score_loss_weight',
        type=float,
        default=1.,
        help='The loss weight for score_mlr_loss.')
    parser.add_argument(
        '--bce_loss_weight',
        type=float,
        default=1.,
        help='The loss weight for bce_loss.')
    parser.add_argument(
        '--monitor_metric_name',
        choices=['acc', 'mlr_loss', 'dual_mlr_loss', 'dual_fat_loss',
                 'margin_ranking_loss', 'fat_loss', 'vanilla_mlr_loss'],
        nargs='+',
        help='The metric used as the monitor for saving checkpoints.')
    parser.add_argument(
        '--monitor_metric_type',
        choices=['min', 'max'],
        nargs='+',
        help='The quantified type of the monitor metric.'
             '"min" means lower is better, while "max" means higher is better.')
    parser.add_argument(
        '--checkpoint_dir_path',
        help='The path of the output checkpoint directory.')
    parser.add_argument(
        '--logging_level',
        choices=['INFO', 'DEBUG'],
        help='The output logging level.')
    parser.add_argument(
        '--centroid_mode',
        choices=['mean'],
        help='The mode to compute the centroid feature.')
    parser.add_argument(
        '--distance_mode',
        choices=['cosine'],
        help='The mode to compute the distance between two features.')
    parser.add_argument(
        '--pretrained_model_name',
        choices=['bert-base-uncased'],
        help='The name of the pretrained model.')
    parser.add_argument(
        '--weighted_s_loss',
        action='store_true',
        help='Computing the inter-cluster spearation loss with weights.')
    parser.add_argument(
        '--use_projection_head',
        action='store_true',
        help='Computing the inter-cluster spearation loss with weights.')

    args = parser.parse_args()
    return args

def parse_finetune_opt():
    parser = argparse.ArgumentParser()

    # general settings
    parser.add_argument(
        '--dataset',
        choices=['human_judgement_for_fine_tuning'],
        help='Dataset name.')
    parser.add_argument(
        '--trainer',
        choices=['kd_finetune'],
        help='Trainer name (currently supports `kd_finetune`).')
    parser.add_argument(
        '--model',
        choices=['bert_metric'],
        help='Model name (currently supports `bert_metric`).')
    parser.add_argument(
        '--gpu',
        help='the id of GPU for loading model and data.')
    parser.add_argument(
        '--seed',
        type=int,
        help='The seed for reproducibility (optional).')
    parser.add_argument(
        '--max_seq_length',
        type=int,
        default=128,
        help='The max sequence length of the context-response pair.')

    # data settings
    parser.add_argument(
        '--data',
        choices=['dailydialog_EVAL'],
        help='The data used for fine-tuning.')
    parser.add_argument(
        '--train_batch_size',
        type=int,
        help='The batch size for training.')
    parser.add_argument(
        '--eval_batch_size',
        type=int,
        help='The batch size for validation.')

    # trainer settings
    parser.add_argument(
        '--num_epochs',
        type=int,
        help='The number of epochs for training.')
    parser.add_argument(
        '--display_steps',
        type=int,
        help='Print training loss every display_steps.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        help='The initial learning rate for training.')
    parser.add_argument(
        '--warmup_proportion',
        type=float,
        help='The warmup proportion of the warmup strategy for LR scheduling.')
    parser.add_argument(
        '--kd_loss_weight',
        type=float,
        default=1.,
        help='The loss weight for kd_loss.')
    parser.add_argument(
        '--finetune_loss_weight',
        type=float,
        default=1.,
        help='The loss weight for finetune_loss.')
    parser.add_argument(
        '--monitor_metric_name',
        choices=['kd_finetune_loss'],
        nargs='+',
        help='The metric used as the monitor for saving checkpoints.')
    parser.add_argument(
        '--monitor_metric_type',
        choices=['min', 'max'],
        nargs='+',
        help='The quantified type of the monitor metric.'
             '"min" means lower is better, while "max" means higher is better.')
    parser.add_argument(
        '--checkpoint_dir_path',
        help='The path of the output fine-tuning checkpoint directory.')
    parser.add_argument(
        '--pretrain_checkpoint_dir_path',
        help='The directory path of the checkpoint to be loaded.')
    parser.add_argument(
        '--pretrain_checkpoint_file_name',
        help='The file name of the checkpoint to be loaded.')
    parser.add_argument(
        '--logging_level',
        choices=['INFO', 'DEBUG'],
        help='The output logging level.')
    parser.add_argument(
        '--pretrained_model_name',
        choices=['bert-base-uncased'],
        help='The name of the pretrained model.')

    args = parser.parse_args()
    return args
