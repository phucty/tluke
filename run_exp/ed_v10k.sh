VOCAB=10k
DATA="data/tluke_pretraining_bert_large_${VOCAB}/"
GPUS=4
PRETRAIN=models/tluke_bert_large
STAGE1="pretraining_config/tluke_large_${VOCAB}_stage1.json"
STAGE2="pretraining_config/tluke_large_${VOCAB}_stage2.json"
BERT=bert-large-uncased-whole-word-masking
EPOCH1=10
EPOCH2=10
PRETRAIN_INIT=models/luke_ed_large/pytorch_model.bin
PRETRAIN_STAGE1="${PRETRAIN}/checkpoints/epoch${EPOCH1}"
PRETRAIN_STAGE2="${PRETRAIN}/checkpoints/epoch${EPOCH2}/mp_rank_00_model_states.pt"
PRETRAIN_METADATA="${PRETRAIN}/metadata.json"
PRETRAIN_VOCAB="${PRETRAIN}/entity_vocab.jsonl"

ED=models/tluke_ed_large/

deepspeed \
    --num_gpus=$GPUS \
    luke/pretraining/train.py \
    --output-dir=$PRETRAIN \
    --deepspeed-config-file=$STAGE1 \
    --dataset-dir=$DATA \
    --bert-model-name=$BERT \
    --num-epochs=$EPOCH1 \
    --entity-emb-size=1024 \
    --masked-lm-prob=0.0 \
    --masked-entity-prob=0.3 \
    --fix-bert-weights \
    --from-tables \
    --reset-optimization-states \
    --resume-checkpoint-id=$PRETRAIN_INIT


# deepspeed \
#     --num_gpus=$GPUS \
#     luke/pretraining/train.py \
#     --output-dir=$PRETRAIN \
#     --deepspeed-config-file=$STAGE2 \
#     --dataset-dir=$DATA \
#     --bert-model-name=$BERT \
#     --num-epochs=$EPOCH2 \
#     --entity-emb-size=1024 \
#     --masked-lm-prob=0.0 \
#     --masked-entity-prob=0.3 \
#     --reset-optimization-states \
#     --from-tables \
#     --resume-checkpoint-id=$PRETRAIN_STAGE1

python examples/entity_disambiguation/scripts/convert_checkpoint.py \
    --checkpoint-file=$PRETRAIN_STAGE2 \
    --metadata-file=$PRETRAIN_METADATA \
    --entity-vocab-file=$PRETRAIN_VOCAB \
    --output-dir=$ED \
    --use-tluke

python examples/entity_disambiguation/train.py \
  --model-dir=$ED \
  --dataset-dir=data/entity_disambiguation/ \
  --titles-file=data/entity_disambiguation/enwiki_20181220_titles.txt \
  --redirects-file=data/entity_disambiguation/enwiki_20181220_redirects.tsv \
  --output-dir=$ED  \
  --device="cuda:3" \
  --use-tluke


python examples/entity_disambiguation/evaluate.py \
  --model-dir=$ED \
  --dataset-dir=data/entity_disambiguation/ \
  --titles-file=data/entity_disambiguation/enwiki_20181220_titles.txt \
  --redirects-file=data/entity_disambiguation/enwiki_20181220_redirects.tsv \
  --inference-mode=global \
  --document-split-mode=per_mention \
  --device="cuda:3" \
  --use-tluke
  