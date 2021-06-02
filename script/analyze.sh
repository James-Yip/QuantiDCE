EVAL_MODE=mix
# EVAL_MODE=separate

# Returns to main directory
cd ../

# Computes human correlations
python eval.py --datasets dailydialog_EVAL convai2 empatheticdialogues \
               --eval_data_dir_path ./evaluation/eval_data \
               --eval_mode $EVAL_MODE \
               --analyze

