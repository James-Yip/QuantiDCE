SEED=71
# EVAL_MODE=separate

# Returns to main directory
cd ../

# Computes human correlations
python infer.py \
    --model bert_metric \
    --checkpoint_dir_path ./output/${SEED}/bert_metric_kd_finetune \
    --checkpoint_file_name model_best_kd_finetune_loss.ckpt \
    --pretrained_model_name bert-base-uncased
