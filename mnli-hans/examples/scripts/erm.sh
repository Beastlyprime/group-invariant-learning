#!/bin/bash
METRICS='{"accuracy":{"type":"categorical_accuracy"}}'
EVAL_FILES='{"path1":"examples/configs/eval/mnli_tev.jsonnet","path2":"examples/configs/eval/hans_oracle.jsonnet"}'
TRAIN_BIAS='{"dataset_name":"mnli_train","file_name":"data/bias/mnli-hans/mind-trade-bias/train.json"}'
PROJECT="HANS-ERM"

for SEED in 54673 # 54324 13214 # 3420 # 28987 37462 # # 
do
    OUTPUT_FOLDER=/mnli/multi_env/erm/basic_bert_lr_5_epoch_3_SEED-${SEED}/
    FILE=$OUTPUT_FOLDER/metrics_epoch_2.json
    if [ ! -f "$FILE" ]; then
        READER_DEBUG=0 python -W ignore::UserWarning __main__.py assemble_debiased_train \
            --train-set-param-path examples/configs/dataset/mnli_train.jsonnet \
            --validation-set-param-path examples/configs/dataset/mnli_dev.jsonnet \
            --training-param-path examples/configs/training/mnli_bucket_loader_training_hans_lr_5.jsonnet \
            --main-model-param-path examples/configs/main_model/basic_bert_classifier.jsonnet \
            --bias-file-args "[${TRAIN_BIAS}]" \
            -s $OUTPUT_FOLDER --force \
            --ebd-mode two_stage \
            --ebd-loss cross_entropy \
            --metrics $METRICS \
            --seed ${SEED}
    fi    
done