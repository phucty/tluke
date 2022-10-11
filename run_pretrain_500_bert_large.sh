deepspeed \
    --num_gpus=2 \
    luke/pretraining/train.py \
    --output-dir=luke_table_models_bert_large_500 \
    --deepspeed-config-file=pretraining_config/luke_large_stage1.json \
    --dataset-dir=luke_table_pretraining_dataset_bert_500/ \
    --bert-model-name=bert-large-uncased-whole-word-masking \
    --num-epochs=20 \
    --from-tables \
    --fix-bert-weights


deepspeed \
    --num_gpus=2 \
    luke/pretraining/train.py \
    --output-dir=luke_table_models_bert_large_500 \
    --deepspeed-config-file=pretraining_config/luke_large_stage2.json \
    --dataset-dir=luke_table_pretraining_dataset_bert_500/ \
    --bert-model-name=bert-large-uncased-whole-word-masking \
    --num-epochs=20 \
    --from-tables \
    --reset-optimization-states \
    --resume-checkpoint-id=luke_table_models_bert_large_500/checkpoints/epoch20