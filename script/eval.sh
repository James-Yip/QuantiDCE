GPU=2
SEED=71
EVAL_MODE=mix
# EVAL_MODE=separate

# Returns to main directory
cd ../

# Computes human correlations
python eval.py \
    --gpu $GPU \
    --datasets convai2 \
    --model bert_metric \
    --eval_data_dir_path ./evaluation/eval_data \
    --eval_mode $EVAL_MODE \
    --checkpoint_dir_path ./output/${SEED}/pretrain/bert_metric_kd_finetune \
    --checkpoint_file_name model_best_kd_finetune_loss.ckpt \
    --pretrained_model_name bert-base-uncased
