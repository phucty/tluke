deepspeed \
    --num_gpus=2 \
    luke/pretraining/train.py \
    --output-dir=models/tluke_bert_large_100 \
    --deepspeed-config-file=pretraining_config/tluke_large_stage1.json \
    --dataset-dir=data/tluke_pretraining_bert_large_100/ \
    --bert-model-name=bert-large-uncased-whole-word-masking \
    --num-epochs=5 \
    --masked-lm-prob=0.0 \
    --masked-entity-prob=0.3 \
    --fix-bert-weights \
    --from-tables \
    --reset-optimization-states \
    --resume-checkpoint-id=models/luke_ed_large/pytorch_model.bin


deepspeed \
    --num_gpus=2 \
    luke/pretraining/train.py \
    --output-dir=models/tluke_bert_large_100 \
    --deepspeed-config-file=pretraining_config/tluke_large_stage2.json \
    --dataset-dir=data/tluke_pretraining_bert_large_100/ \
    --bert-model-name=bert-large-uncased-whole-word-masking \
    --num-epochs=5 \
    --masked-lm-prob=0.0 \
    --masked-entity-prob=0.3 \
    --reset-optimization-states \
    --from-tables \
    --resume-checkpoint-id=models/tluke_bert_large_100/checkpoints/epoch5

cp models/tluke_bert_large_100/checkpoints/epoch5/mp_rank_00_model_states.pt models/tluke_ed_large/pytorch_model.bin
cp models/tluke_bert_large_100/metadata.json models/tluke_ed_large/metadata.json

python examples/entity_disambiguation/train.py \
  --model-dir=models/tluke_ed_large/ \
  --dataset-dir=data/entity_disambiguation/ \
  --titles-file=data/entity_disambiguation/enwiki_20181220_titles.txt \
  --redirects-file=data/entity_disambiguation/enwiki_20181220_redirects.tsv \
  --output-dir=models/tluke_ed_large/  \
  --use-tluke


python examples/entity_disambiguation/evaluate.py \
  --model-dir=models/tluke_ed_large/ \
  --dataset-dir=data/entity_disambiguation/ \
  --titles-file=data/entity_disambiguation/enwiki_20181220_titles.txt \
  --redirects-file=data/entity_disambiguation/enwiki_20181220_redirects.tsv \
  --inference-mode=local \
  --document-split-mode=simple \
  --use-tluke
