VOCAB=1k
DATA="data/tluke_pretraining_bert_large_${VOCAB}/"
GPUS=4
PRETRAIN=models/tluke_bert_large
STAGE1="pretraining_config/tluke_large_${VOCAB}_stage1.json"
STAGE2="pretraining_config/tluke_large_${VOCAB}_stage2.json"
BERT=bert-large-uncased-whole-word-masking
EPOCH1=1
EPOCH2=6
PRETRAIN_INIT=models/luke_ed_large/pytorch_model.bin
PRETRAIN_STAGE1="${PRETRAIN}/checkpoints/epoch${EPOCH1}"
PRETRAIN_STAGE2="${PRETRAIN}/checkpoints/epoch${EPOCH2}/mp_rank_00_model_states.pt"
PRETRAIN_METADATA="${PRETRAIN}/metadata.json"

ED=models/tluke_ed_large/
ED_MODEL="${ED}pytorch_model.bin"
ED_METADATA="${ED}metadata.json"


deepspeed \
    --num_gpus=$GPUS \
    luke/pretraining/train.py \
    --output-dir=$PRETRAIN \
    --deepspeed-config-file=$STAGE1 \
    --dataset-dir=$DATA \
    --bert-model-name=$BERT \
    --num-epochs=$EPOCH1 \
    --masked-lm-prob=0.0 \
    --masked-entity-prob=0.3 \
    --fix-bert-weights \
    --from-tables \
    --reset-optimization-states \
    --resume-checkpoint-id=$PRETRAIN_INIT


deepspeed \
    --num_gpus=$GPUS \
    luke/pretraining/train.py \
    --output-dir=$PRETRAIN \
    --deepspeed-config-file=$STAGE2 \
    --dataset-dir=$DATA \
    --bert-model-name=$BERT \
    --num-epochs=$EPOCH2 \
    --masked-lm-prob=0.0 \
    --masked-entity-prob=0.3 \
    --reset-optimization-states \
    --from-tables \
    --resume-checkpoint-id=$PRETRAIN_STAGE1

echo $PRETRAIN_STAGE2 $ED_MODEL
echo $PRETRAIN_METADATA $ED_METADATA
echo $PRETRAIN_VOCAB $ED_VOCAB
cp $PRETRAIN_STAGE2 $ED_MODEL
cp $PRETRAIN_METADATA $ED_METADATA
cp $PRETRAIN_VOCAB $ED_VOCAB

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
  --inference-mode=local \
  --document-split-mode=simple \
  --device="cuda:3" \
  --use-tluke
