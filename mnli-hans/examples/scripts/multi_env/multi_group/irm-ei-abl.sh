#!/bin/bash
PREDEFINED_GROUP_FILE='{"dataset_name":"mnli_train","file_name":"data/bias/mnli-hans/mind-trade-bias/train-eiil-random-1010-new.json"}'
WEIGHTS_FILE="/mnli/multi_env/envs/ei/idx2weight.npy"
METRICS='{"accuracy":{"type":"categorical_accuracy"}}'
EVAL_FILES='{"path1":"examples/configs/eval/mnli_tev.jsonnet","path2":"examples/configs/eval/hans_oracle.jsonnet"}'
PROJECT="HANS-EIIL-tev"

for METHOD in IRMv1
do
    for SEED in 54324 # 13214 54673
    do
        for ADAPT_WEIGHT in 1e-3 1e-4 1e-2
        do
            for ASCEND_RATE in 0.2 0.4 0.6 
            do
                OUTPUT_FOLDER=/mnli/multi_env/eiil-bias-1010/${METHOD}/basic_bert_lr_5_epoch_3_lambda_${ADAPT_WEIGHT}_${ASCEND_RATE}-SEED-${SEED}/
                FILE=$OUTPUT_FOLDER/metrics_epoch_2.json
                if [ ! -f "$FILE" ]; then
                    READER_DEBUG=0 python -W ignore::UserWarning __main__.py multi_environment_train \
                        --train-set-param-path examples/configs/dataset/mnli_train.jsonnet \
                        --validation-set-param-path examples/configs/dataset/mnli_dev.jsonnet \
                        --training-param-path examples/configs/training/mnli_group_base_hans_lr_5.jsonnet \
                        --model-param-path examples/configs/main_model/basic_bert_classifier.jsonnet \
                        --predefined-group-file "[${PREDEFINED_GROUP_FILE}]" \
                        --group-weights-file ${WEIGHTS_FILE} \
                        --multi-env-loss ${METHOD} \
                        --loss-args '{"weight_adapt":'${ADAPT_WEIGHT}'}' \
                        --sampler predefined_group \
                        -s $OUTPUT_FOLDER --force \
                        --metrics $METRICS \
                        --seed ${SEED} \
                        --overrides '{"trainer":{"num_epochs":3,"weight_adapt_ascend_ratio":'${ASCEND_RATE}'}}'
                fi
                    
            done
        done
    done
done