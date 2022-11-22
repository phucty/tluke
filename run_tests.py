import json
from lib2to3.pgen2.token import tok_name

import numpy as np
import torch
import tqdm
from transformers import (LukeForEntityClassification,
                          LukeForEntityPairClassification,
                          LukeForEntitySpanClassification, LukeModel,
                          LukeTokenizer, pipeline)

from luke.pretraining.train import pretrain
from luke.utils.wikipedia_parser import _parse_tables


def main_example():
    model = LukeModel.from_pretrained("studio-ousia/luke-base")
    tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")

    text = "Beyoncé lives in Los Angeles."
    print(f"Input: {text}")
    # tmp_tokens = tokenizer.tokenize(text)
    # tmp_tokens_ids = tokenizer.convert_tokens_to_ids(tmp_tokens)
    # print(f"Tokenization: {tmp_tokens}")
    # print(f"Tokenization IDs: {tmp_tokens_ids}")

    # entity_spans = [(0, 7)]  # character-based entity span corresponding to "Beyoncé"
    # inputs = tokenizer(text, entity_spans=entity_spans, add_prefix_space=True, return_tensors="pt")
    # outputs = model(**inputs)
    # word_last_hidden_state = outputs.last_hidden_state
    # entity_last_hidden_state = outputs.entity_last_hidden_state

    entities = [
        "Beyoncé",
        "[MASK]",
    ]  # Wikipedia entity titles corresponding to the entity mentions "Beyoncé" and "Los Angeles"
    entity_spans = [(0, 7), (17, 28)]  # character-based entity spans corresponding to "Beyoncé" and "Los Angeles"
    inputs = tokenizer(text, entities=entities, entity_spans=entity_spans, add_prefix_space=True, return_tensors="pt")
    outputs = model(**inputs)

    word_last_hidden_state1 = outputs.last_hidden_state
    entity_last_hidden_state1 = outputs.entity_last_hidden_state

    entities = [
        "Beyoncé",
        "[MASK]",
    ]  # Wikipedia entity titles corresponding to the entity mentions "Beyoncé" and "Los Angeles"
    entity_spans = [(0, 7), (17, 28)]  # character-based entity spans corresponding to "Beyoncé" and "Los Angeles"
    inputs = tokenizer(text, entities=entities, entity_spans=entity_spans, add_prefix_space=True, return_tensors="pt")
    outputs = model(**inputs)
    word_last_hidden_state2 = outputs.last_hidden_state
    entity_last_hidden_state2 = outputs.entity_last_hidden_state

    model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
    tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
    entity_spans = [(0, 7), (17, 28)]  # character-based entity spans corresponding to "Beyoncé" and "Los Angeles"
    inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = int(logits[0].argmax())
    print("Predicted class:", model.config.id2label[predicted_class_idx])


def run_luke_entity_classification():
    tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")
    model = LukeForEntityClassification.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")

    text = "Beyoncé lives in Los Angeles."
    entity_spans = [(0, 7)]  # character-based entity span corresponding to "Beyoncé"
    inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])

    # print(torch.cuda.is_available())
    # model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
    # tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
    # text = "Beyoncé lives in Los Angeles."
    # entity_spans = [(0, 7), (17, 28)]  # character-based entity spans corresponding to "Beyoncé" and "Los Angeles"
    # inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
    # outputs = model(**inputs)
    # logits = outputs.logits
    # predicted_class_idx = int(logits[0].argmax())
    # print("Predicted class:", model.config.id2label[predicted_class_idx])
    # # Predicted class: per:cities_of_residence


def run_luke_entity_pair_classification():

    tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
    model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-large-finetuned-tacred")

    text = "Beyoncé lives in Los Angeles."
    entity_spans = [
        (0, 7),
        (17, 28),
    ]  # character-based entity spans corresponding to "Beyoncé" and "Los Angeles"
    inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])


def run_luke_entity_span_classification():
    tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")
    model = LukeForEntitySpanClassification.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")

    text = "Beyoncé lives in Los Angeles"

    word_start_positions = [0, 8, 14, 17, 21]  # character-based start positions of word tokens
    word_end_positions = [7, 13, 16, 20, 28]  # character-based end positions of word tokens
    entity_spans = []
    for i, start_pos in enumerate(word_start_positions):
        for end_pos in word_end_positions[i:]:
            entity_spans.append((start_pos, end_pos))
            print(f"{start_pos}:{end_pos}: {text[start_pos:end_pos]}")

    inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_indices = logits.argmax(-1).squeeze().tolist()
    for span, predicted_class_idx in zip(entity_spans, predicted_class_indices):
        if predicted_class_idx != 0:
            print(text[span[0] : span[1]], model.config.id2label[predicted_class_idx])


def run_notebook_open_entity_typing():
    def load_examples(dataset_file):
        with open(dataset_file, "r") as f:
            data = json.load(f)

        examples = []
        for item in data:
            examples.append(dict(text=item["sent"], entity_spans=[(item["start"], item["end"])], label=item["labels"]))

        return examples

    test_examples = load_examples("data/data/OpenEntity/test.json")
    # Load the model checkpoint
    model = LukeForEntityClassification.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")
    model.eval()
    model.to("cuda")

    # Load the tokenizer
    tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")
    batch_size = 128

    num_predicted = 0
    num_gold = 0
    num_correct = 0

    all_predictions = []
    all_labels = []

    for batch_start_idx in tqdm.trange(0, len(test_examples), batch_size):
        batch_examples = test_examples[batch_start_idx : batch_start_idx + batch_size]
        texts = [example["text"] for example in batch_examples]
        entity_spans = [example["entity_spans"] for example in batch_examples]
        gold_labels = [example["label"] for example in batch_examples]

        inputs = tokenizer(texts, entity_spans=entity_spans, return_tensors="pt", padding=True)
        inputs = inputs.to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)

        num_gold += sum(len(l) for l in gold_labels)
        for logits, labels in zip(outputs.logits, gold_labels):
            for index, logit in enumerate(logits):
                if logit > 0:
                    num_predicted += 1
                    predicted_label = model.config.id2label[index]
                    if predicted_label in labels:
                        num_correct += 1

    precision = num_correct / num_predicted
    recall = num_correct / num_gold
    f1 = 2 * precision * recall / (precision + recall)

    print(f"\n\nprecision: {precision} recall: {recall} f1: {f1}")

    text = "Beyoncé lives in Los Angeles."
    entity_spans = [(0, 7)]  # character-based entity span corresponding to "Beyoncé"

    inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
    inputs.to("cuda")
    outputs = model(**inputs)

    predicted_indices = [index for index, logit in enumerate(outputs.logits[0]) if logit > 0]
    print("Predicted entity type for Beyoncé:", [model.config.id2label[index] for index in predicted_indices])

    entity_spans = [(17, 28)]  # character-based entity span corresponding to "Beyoncé"
    inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
    inputs.to("cuda")
    outputs = model(**inputs)

    predicted_indices = [index for index, logit in enumerate(outputs.logits[0]) if logit > 0]
    print("Predicted entity type for Los Angeles:", [model.config.id2label[index] for index in predicted_indices])


def run_test_transformer():
    summarizer = pipeline("summarization")
    result = summarizer(
        """
        America has changed dramatically during recent years. Not only has the number of 
        graduates in traditional engineering disciplines such as mechanical, civil, 
        electrical, chemical, and aeronautical engineering declined, but in most of 
        the premier American universities engineering curricula now concentrate on 
        and encourage largely the study of engineering science. As a result, there 
        are declining offerings in engineering subjects dealing with infrastructure, 
        the environment, and related issues, and greater concentration on high 
        technology subjects, largely supporting increasingly complex scientific 
        developments. While the latter is important, it should not be at the expense 
        of more traditional engineering.

        Rapidly developing economies such as China and India, as well as other 
        industrial countries in Europe and Asia, continue to encourage and advance 
        the teaching of engineering. Both China and India, respectively, graduate 
        six and eight times as many traditional engineers as does the United States. 
        Other industrial countries at minimum maintain their output, while America 
        suffers an increasingly serious decline in the number of engineering graduates 
        and a lack of well-educated engineers.
    """
    )
    print(result)


def huggingface_tut_using_transformers():
    from accelerate import Accelerator
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import (AdamW, AutoModelForSequenceClassification,
                              AutoTokenizer, DataCollatorWithPadding,
                              get_scheduler)

    raw_datasets = load_dataset("glue", "mrpc")
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator)

    accelerator = Accelerator()
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    train_dl, eval_dl, model, optimizer = accelerator.prepare(train_dataloader, eval_dataloader, model, optimizer)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    progress_bar = tqdm.tqdm(total=num_training_steps)

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dl:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    from datasets import load_metric

    metric = load_metric("glue", "mrpc")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    print(metric.compute())


if __name__ == "__main__":
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-base")

    model = AutoModelForMaskedLM.from_pretrained("studio-ousia/luke-base")
    model.save_pretrained("models/luke-base-1")
    # import debugpy

    # debugpy.listen(5678)
    # print("Wait for client")
    # debugpy.wait_for_client()
    # print("Attached")
    # import tensorflow as tf
    # import torch

    # print(tf.test.is_gpu_available())
    # print(torch.version.cuda)
    # print(torch.cuda.is_available())
    # run_test_transformer()
    # huggingface_tut_using_transformers()
    # main_example()
    # run_luke_entity_classification()
    # run_luke_entity_pair_classification()
    # run_luke_entity_span_classification()
    # run_notebook_open_entity_typing()

    # pretrain()
    """
    tmux capture-pane -pS -1000000 > log/run_pretrain.log
    ps -ef | grep python
    sudo fuser -v /dev/nvidia*
    
    Pharse 1
    deepspeed \
    --num_gpus=1 \
    luke/pretraining/train.py \
    --output-dir=luke_models_base_500k \
    --deepspeed-config-file=pretraining_config/luke_base_stage1.json \
    --dataset-dir=luke_pretraining_dataset_500k/ \
    --bert-model-name=roberta-base \
    --num-epochs=20 \
    --save-interval-steps=1000 \
    --fix-bert-weights \
    --resume-checkpoint-id=luke_models_base_500k/step0002000/mp_rank_00_model_states.pt
    


    Pharse 2
    deepspeed \
    --num_gpus=1 \
    luke/pretraining/train.py \
    --output-dir=luke_models_base_500k \
    --deepspeed-config-file=pretraining_config/luke_base_stage2.json \
    --dataset-dir=luke_pretraining_dataset_500k/ \
    --bert-model-name=roberta-base \
    --num-epochs=20 \
    --reset-optimization-states \
    --resume-checkpoint-id=luke_models_base_500k
```


python -m examples.cli \
    --model-file=models/luke_large_ed.tar.gz \
    entity-disambiguation run \
    --data-dir=data/entity_disambiguation \
    --no-train \
    --do-eval

    """
