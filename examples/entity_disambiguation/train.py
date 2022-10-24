import json
import logging
import os
from collections import OrderedDict
from email.policy import default

import click
import torch
from examples.entity_disambiguation.dataloader import create_dataloader
from examples.entity_disambiguation.dataset import load_dataset
from examples.entity_disambiguation.model import LukeForEntityDisambiguation, TLukeForEntityDisambiguation
from luke.model_table import LukeTableConfig
from luke.utils.entity_vocab import MASK_TOKEN, EntityVocab
from numpy import False_
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

# import debugpy

# debugpy.listen(5678)
# print("Wait for client")
# debugpy.wait_for_client()
# print("Attached")


@click.command()
@click.option("--model-dir", type=click.Path(exists=True), required=True)
@click.option("--dataset-dir", type=click.Path(exists=True), required=True)
@click.option("--output-dir", type=click.Path(), required=True)
@click.option("--titles-file", type=click.Path(exists=True), required=True)
@click.option("--redirects-file", type=click.Path(exists=True), required=True)
@click.option("--batch-size", type=int, default=8)
@click.option("--gradient-accumulation-steps", type=int, default=1)
@click.option("--learning-rate", type=float, default=2e-5)
@click.option("--num-epochs", type=int, default=2)
@click.option("--warmup-ratio", type=float, default=0.1)
@click.option("--weight-decay", type=float, default=0.01)
@click.option("--max-grad-norm", type=float, default=1.0)
@click.option("--masked-entity-prob", type=float, default=0.9)
@click.option("--max-seq-length", type=int, default=512)
@click.option("--max-entity-length", type=int, default=128)
@click.option("--max-candidate-length", type=int, default=30)
@click.option("--max-mention-length", type=int, default=30)
@click.option("--device", type=str, default="cuda")
@click.option("--use-tluke", is_flag=True, default=False)
def train(
    model_dir: str,
    dataset_dir: str,
    output_dir: str,
    titles_file: str,
    redirects_file: str,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    num_epochs: int,
    warmup_ratio: float,
    weight_decay: float,
    max_grad_norm: float,
    masked_entity_prob: float,
    max_seq_length: int,
    max_entity_length: int,
    max_candidate_length: int,
    max_mention_length: int,
    device: str,
    use_tluke: bool,
):
    entity_vocab = EntityVocab(os.path.join(model_dir, "entity_vocab.jsonl"))
    mask_entity_id = entity_vocab[MASK_TOKEN]
    if use_tluke:
        model = TLukeForEntityDisambiguation.from_pretrained(model_dir).train()
        model.to(device)
        model.tluke.entity_embeddings.entity_embeddings.weight.requires_grad = False
    else:
        model = LukeForEntityDisambiguation.from_pretrained(model_dir).train()
        model.to(device)
        model.luke.entity_embeddings.entity_embeddings.weight.requires_grad = False

    assert not model.entity_predictions.decoder.weight.requires_grad
    model.entity_predictions.bias.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    dataset = load_dataset(
        dataset_dir=dataset_dir,
        titles_file=titles_file,
        redirects_file=redirects_file,
    )
    documents = dataset.get_dataset("train")
    dataloader = create_dataloader(
        documents=documents,
        tokenizer=tokenizer,
        entity_vocab=entity_vocab,
        batch_size=batch_size,
        fold="train",
        document_split_mode="simple",
        max_seq_length=max_seq_length,
        max_entity_length=max_entity_length,
        max_candidate_length=max_candidate_length,
        max_mention_length=max_mention_length,
    )
    num_train_steps = len(dataloader) // gradient_accumulation_steps * num_epochs

    optimizer_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in ["bias", "LayerNorm.weight"])
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in ["bias", "LayerNorm.weight"])
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_parameters, lr=learning_rate, eps=1e-6, correct_bias=False)

    warmup_steps = int(num_train_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_train_steps)

    step = 0
    for epoch_num in range(num_epochs):
        with tqdm(dataloader) as pbar:
            for batch in pbar:
                batch["labels"] = batch["entity_ids"].clone()
                for index, entity_length in enumerate(batch["entity_attention_mask"].sum(1).tolist()):
                    masked_entity_length = max(1, round(entity_length * masked_entity_prob))
                    permutated_indices = torch.randperm(entity_length)[:masked_entity_length]
                    batch["entity_ids"][index, permutated_indices[:masked_entity_length]] = mask_entity_id
                    batch["labels"][index, permutated_indices[masked_entity_length:]] = -1
                batch = {k: v.to(device) for k, v in batch.items() if k != "eval_entity_mask"}
                outputs = model(**batch)
                loss = outputs[0]
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    pbar.set_description(f"epoch: {epoch_num} loss: {loss:.7f}")
                step += 1

    os.makedirs(output_dir, exist_ok=True)
    entity_vocab.save(os.path.join(output_dir, "entity_vocab.jsonl"))
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


# @click.command()
# @click.option("--model-dir", type=click.Path(exists=True), required=True)
# @click.option("--dataset-dir", type=click.Path(exists=True), required=True)
# @click.option("--output-dir", type=click.Path(), required=True)
# @click.option("--titles-file", type=click.Path(exists=True), required=True)
# @click.option("--redirects-file", type=click.Path(exists=True), required=True)
# @click.option("--batch-size", type=int, default=8)
# @click.option("--gradient-accumulation-steps", type=int, default=1)
# @click.option("--learning-rate", type=float, default=2e-5)
# @click.option("--num-epochs", type=int, default=2)
# @click.option("--warmup-ratio", type=float, default=0.1)
# @click.option("--weight-decay", type=float, default=0.01)
# @click.option("--max-grad-norm", type=float, default=1.0)
# @click.option("--masked-entity-prob", type=float, default=0.9)
# @click.option("--max-seq-length", type=int, default=512)
# @click.option("--max-entity-length", type=int, default=128)
# @click.option("--max-candidate-length", type=int, default=30)
# @click.option("--max-mention-length", type=int, default=30)
# @click.option("--device", type=str, default="cuda")
# def train_tluke(
#     model_dir: str,
#     dataset_dir: str,
#     output_dir: str,
#     titles_file: str,
#     redirects_file: str,
#     batch_size: int,
#     gradient_accumulation_steps: int,
#     learning_rate: float,
#     num_epochs: int,
#     warmup_ratio: float,
#     weight_decay: float,
#     max_grad_norm: float,
#     masked_entity_prob: float,
#     max_seq_length: int,
#     max_entity_length: int,
#     max_candidate_length: int,
#     max_mention_length: int,
#     device: str,
# ):
#     metadata = json.load(open(os.path.join(model_dir, "config.json")))
#     config = LukeTableConfig(**metadata["model_config"])

#     entity_vocab = EntityVocab(os.path.join(model_dir, "entity_vocab.jsonl"))
#     mask_entity_id = entity_vocab[MASK_TOKEN]

#     state_dict = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location="cpu")
#     if "module" in state_dict:
#         state_dict = state_dict["module"]

#     new_state_dict = OrderedDict()
#     for key, value in state_dict.items():
#         if key.startswith("lm_head") or key.startswith("cls"):
#             continue
#         if key.startswith("entity_predictions"):
#             new_state_dict[key] = value
#         else:
#             new_state_dict[f"tluke.{key}"] = value
#     entity_embeddings = state_dict["entity_embeddings.entity_embeddings.weight"]

#     new_state_dict["tluke.entity_embeddings.mask_embedding"] = entity_embeddings[mask_entity_id]
#     del state_dict
#     model = TLukeForEntityDisambiguation(config=config)
#     missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
#     del new_state_dict
#     assert not missing_keys or missing_keys == ["tluke.embeddings.position_ids"]
#     # assert not unexpected_keys
#     # model.tie_weights()

#     # model = TLukeForEntityDisambiguation.from_pretrained(model_dir).train()
#     model.to(device)
#     model.tluke.entity_embeddings.entity_embeddings.weight.requires_grad = False
#     assert not model.entity_predictions.decoder.weight.requires_grad
#     model.entity_predictions.bias.requires_grad = False

#     tokenizer = AutoTokenizer.from_pretrained(model_dir)

#     dataset = load_dataset(
#         dataset_dir=dataset_dir,
#         titles_file=titles_file,
#         redirects_file=redirects_file,
#     )
#     documents = dataset.get_dataset("train")
#     dataloader = create_dataloader(
#         documents=documents,
#         tokenizer=tokenizer,
#         entity_vocab=entity_vocab,
#         batch_size=batch_size,
#         fold="train",
#         document_split_mode="simple",
#         max_seq_length=max_seq_length,
#         max_entity_length=max_entity_length,
#         max_candidate_length=max_candidate_length,
#         max_mention_length=max_mention_length,
#     )
#     num_train_steps = len(dataloader) // gradient_accumulation_steps * num_epochs

#     optimizer_parameters = [
#         {
#             "params": [
#                 p
#                 for n, p in model.named_parameters()
#                 if p.requires_grad and not any(nd in n for nd in ["bias", "LayerNorm.weight"])
#             ],
#             "weight_decay": weight_decay,
#         },
#         {
#             "params": [
#                 p
#                 for n, p in model.named_parameters()
#                 if p.requires_grad and any(nd in n for nd in ["bias", "LayerNorm.weight"])
#             ],
#             "weight_decay": 0.0,
#         },
#     ]
#     optimizer = AdamW(optimizer_parameters, lr=learning_rate, eps=1e-6, correct_bias=False)

#     warmup_steps = int(num_train_steps * warmup_ratio)
#     scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_train_steps)

#     step = 0
#     for epoch_num in range(num_epochs):
#         with tqdm(dataloader) as pbar:
#             for batch in pbar:
#                 batch["labels"] = batch["entity_ids"].clone()
#                 for index, entity_length in enumerate(batch["entity_attention_mask"].sum(1).tolist()):
#                     masked_entity_length = max(1, round(entity_length * masked_entity_prob))
#                     permutated_indices = torch.randperm(entity_length)[:masked_entity_length]
#                     batch["entity_ids"][index, permutated_indices[:masked_entity_length]] = mask_entity_id
#                     batch["labels"][index, permutated_indices[masked_entity_length:]] = -1
#                 batch = {k: v.to(device) for k, v in batch.items() if k != "eval_entity_mask"}
#                 outputs = model(**batch)
#                 loss = outputs[0]
#                 if gradient_accumulation_steps > 1:
#                     loss = loss / gradient_accumulation_steps
#                 loss.backward()

#                 if (step + 1) % gradient_accumulation_steps == 0:
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
#                     optimizer.step()
#                     scheduler.step()
#                     model.zero_grad()
#                     pbar.set_description(f"epoch: {epoch_num} loss: {loss:.7f}")
#                 step += 1

#     os.makedirs(output_dir, exist_ok=True)
#     entity_vocab.save(os.path.join(output_dir, "entity_vocab.jsonl"))
#     model.save_pretrained(output_dir)
#     tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    train()
    # train_tluke()
