# Variables
GPU=0
SEED=71
DATASET=human_judgement_for_fine_tuning
MODEL=bert_metric
TRAINER=kd_finetune

DATA=dailydialog_EVAL
TRAIN_BATCH_SIZE=10
EVAL_BATCH_SIZE=10

NUM_EPOCHS=20
DISPLAY_STEPS=10
LEARNING_RATE=5e-6
WARMUP_PROPORTION=0.2
KD_LOSS_WEIGHT=5.
FINETUNE_LOSS_WEIGHT=1.
MONITOR_METRIC_NAME=kd_finetune_loss
MONITOR_METRIC_TYPE=min
PRETRAIN_CHECKPOINT_DIR_PATH=./output/$SEED/bert_metric_mlr_pretrain
PRETRAIN_CHECKPOINT_FILE_NAME=model_best_dual_mlr_loss.ckpt
CHECKPOINT_DIR_PATH=./output/$SEED/bert_metric_kd_finetune
LOGGING_LEVEL=INFO
PRETRAINED_MODEL_NAME=bert-base-uncased


# Returns to main directory
cd ../

# Train
python finetune.py \
    --dataset $DATASET \
    --data $DATA \
    --model $MODEL \
    --trainer $TRAINER \
    --gpu $GPU \
    --pretrain_checkpoint_dir_path $PRETRAIN_CHECKPOINT_DIR_PATH \
    --pretrain_checkpoint_file_name $PRETRAIN_CHECKPOINT_FILE_NAME \
    --checkpoint_dir_path $CHECKPOINT_DIR_PATH \
    --seed $SEED \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --display_steps $DISPLAY_STEPS \
    --monitor_metric_name $MONITOR_METRIC_NAME \
    --monitor_metric_type $MONITOR_METRIC_TYPE \
    --learning_rate $LEARNING_RATE \
    --warmup_proportion $WARMUP_PROPORTION \
    --logging_level $LOGGING_LEVEL \
    --pretrained_model_name $PRETRAINED_MODEL_NAME \
    --kd_loss_weight $KD_LOSS_WEIGHT \
    --finetune_loss_weight $FINETUNE_LOSS_WEIGHT
