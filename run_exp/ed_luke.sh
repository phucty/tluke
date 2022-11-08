python examples/entity_disambiguation/evaluate.py \
  --model-dir=models/luke_ed_large/ \
  --dataset-dir=data/entity_disambiguation/ \
  --titles-file=data/entity_disambiguation/enwiki_20181220_titles.txt \
  --redirects-file=data/entity_disambiguation/enwiki_20181220_redirects.tsv \
  --inference-mode=local \
  --document-split-mode=simple \
  --device="cuda:3" \
