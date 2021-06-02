# Variables
GPU=0
SEED=71
DATASET=dailydialog_plusplus_mlr
MODEL=bert_metric
TRAINER=mlr_pretrain
MODE=train

TRAIN_BATCH_SIZE=3
EVAL_BATCH_SIZE=3
TEST_BATCH_SIZE=3

NUM_EPOCHS=5
DISPLAY_STEPS=50
LEARNING_RATE=2e-5
WARMUP_PROPORTION=0.1
FEATURE_DISTANCE_LOWER_BOUND=1
FEATURE_DISTANCE_UPPER_BOUND=0.1
SCORE_DISTANCE_LOWER_BOUND=0.3
SCORE_DISTANCE_UPPER_BOUND=0.1
FEATURE_LOSS_WEIGHT=0.
SCORE_LOSS_WEIGHT=1.
BCE_LOSS_WEIGHT=0.
MONITOR_METRIC_NAME=dual_mlr_loss
MONITOR_METRIC_TYPE=min
CHECKPOINT_DIR_PATH=./output/$SEED/bert_metric_mlr_pretrain
LOGGING_LEVEL=INFO
CENTROID_MODE=mean
DISTANCE_MODE=cosine
PRETRAINED_MODEL_NAME=bert-base-uncased


# Returns to main directory
cd ../

# Train
python pretrain.py \
    --dataset $DATASET \
    --model $MODEL \
    --trainer $TRAINER \
    --mode $MODE \
    --gpu $GPU \
    --checkpoint_dir_path $CHECKPOINT_DIR_PATH \
    --seed $SEED \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --test_batch_size $TEST_BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --display_steps $DISPLAY_STEPS \
    --monitor_metric_name $MONITOR_METRIC_NAME \
    --monitor_metric_type $MONITOR_METRIC_TYPE \
    --feature_distance_lower_bound $FEATURE_DISTANCE_LOWER_BOUND \
    --feature_distance_upper_bound $FEATURE_DISTANCE_UPPER_BOUND \
    --score_distance_lower_bound $SCORE_DISTANCE_LOWER_BOUND \
    --score_distance_upper_bound $SCORE_DISTANCE_UPPER_BOUND \
    --feature_loss_weight $FEATURE_LOSS_WEIGHT \
    --score_loss_weight $SCORE_LOSS_WEIGHT \
    --bce_loss_weight $BCE_LOSS_WEIGHT \
    --learning_rate $LEARNING_RATE \
    --warmup_proportion $WARMUP_PROPORTION \
    --logging_level $LOGGING_LEVEL \
    --centroid_mode $CENTROID_MODE \
    --distance_mode $DISTANCE_MODE \
    --pretrained_model_name $PRETRAINED_MODEL_NAME \
    --weighted_s_loss
