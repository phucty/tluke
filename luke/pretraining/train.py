import datetime
import glob
import json
import logging
import math
import os
import time
from argparse import Namespace
from collections import OrderedDict
from enum import Flag
from typing import List, Tuple, Union

import click
import numpy as np
import tensorflow as tf
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForPreTraining

from luke.model import LukeConfig
from luke.model_table import LukeTableConfig
from luke.pretraining.batch_generator import LukePretrainingBatchGenerator
from luke.pretraining.dataset import WikipediaPretrainingDataset
from luke.pretraining.model import LukePretrainingModel, LukeTablePretrainingModel
from luke.utils.entity_vocab import EntityVocab
from luke.utils.model_utils import ENTITY_VOCAB_FILE

METADATA_FILE = "metadata.json"

logger = None

import debugpy

# debugpy.listen(5678)
# print("Wait for client")
# debugpy.wait_for_client()
# print("Attached")


def save_checkpoint(
    path: str,
    checkpoint_id: str,
    model: Union[LukePretrainingModel, LukeTablePretrainingModel],
    epoch: int,
    global_step: int,
):
    checkpoint_state_dict = dict(epoch=epoch, global_step=global_step)
    model.save_checkpoint(path, checkpoint_id, checkpoint_state_dict, True)


def load_checkpoint(
    path: str,
    checkpoint_id: str,
    model: Union[LukePretrainingModel, LukeTablePretrainingModel],
    entity_vocab=None,
    config=None,
    **kwargs,
):
    if checkpoint_id.endswith(".bin") or checkpoint_id.endswith(".pt"):
        model_weights = torch.load(os.path.join(path, checkpoint_id), map_location="cpu")
        if "module" in model_weights:
            model_weights = model_weights["module"]

        new_model_weights = OrderedDict()
        for key, value in model_weights.items():
            if key.startswith("entity_predictions"):
                new_model_weights[f"module.{key}"] = value
            else:
                if key.startswith("luke."):
                    key = key.replace("luke.", "")
                elif key.startswith("tluke."):
                    key = key.replace("tluke.", "")
                new_model_weights[f"module.{key}"] = value

        if entity_vocab:
            orig_entity_vocab = EntityVocab(os.path.join(path, "entity_vocab.jsonl"))

            if "luke.entity_embeddings.entity_embeddings.weight" in model_weights:
                orig_entity_emb = model_weights["luke.entity_embeddings.entity_embeddings.weight"]
            else:
                orig_entity_emb = model_weights["entity_embeddings.entity_embeddings.weight"]
            if orig_entity_emb.size(0) != len(entity_vocab):  # detect whether the model is fine-tuned
                entity_emb = orig_entity_emb.new_zeros((entity_vocab.size, config.entity_emb_size))
                orig_entity_bias = model_weights["entity_predictions.bias"]
                entity_bias = orig_entity_bias.new_zeros(entity_vocab.size)
                for entity, index in entity_vocab.vocab.items():
                    if entity.title in orig_entity_vocab:
                        orig_index = orig_entity_vocab[entity.title]
                        entity_emb[index] = orig_entity_emb[orig_index]
                        entity_bias[index] = orig_entity_bias[orig_index]
                new_model_weights["module.entity_embeddings.entity_embeddings.weight"] = entity_emb
                new_model_weights["module.entity_embeddings.mask_embedding"] = entity_emb[1].view(1, -1)
                new_model_weights["module.entity_predictions.decoder.weight"] = entity_emb
                new_model_weights["module.entity_predictions.bias"] = entity_bias

                # module.embeddings.word_embeddings
                # module.embeddings.position_embeddings
                # module.entity_embeddings.position_embeddings
                # module.lm_head.bias
                if (
                    model.module.embeddings.word_embeddings.num_embeddings
                    != new_model_weights["module.embeddings.word_embeddings.weight"].shape[0]
                ):
                    new_model_weights["module.embeddings.word_embeddings.weight"] = new_model_weights[
                        "module.embeddings.word_embeddings.weight"
                    ][: model.module.embeddings.word_embeddings.num_embeddings, :]

                if model.module.lm_head.bias.shape[0] != new_model_weights["module.lm_head.bias"].shape[0]:
                    new_model_weights["module.lm_head.bias"] = new_model_weights["module.lm_head.bias"][
                        : model.module.lm_head.bias.shape[0]
                    ]

                if (
                    model.module.entity_embeddings.position_embeddings.num_embeddings
                    != new_model_weights["module.entity_embeddings.position_embeddings.weight"].shape[0]
                ):
                    org_possition_embeddings = new_model_weights["module.entity_embeddings.position_embeddings.weight"]
                    org_possition_embeddings_size = org_possition_embeddings.shape[0]
                    possition_embeddings_size = model.module.entity_embeddings.position_embeddings.num_embeddings
                    possition_embeddings_hiden = model.module.entity_embeddings.position_embeddings.embedding_dim
                    possition_embeddings = org_possition_embeddings.new_zeros(
                        (possition_embeddings_size, possition_embeddings_hiden)
                    )
                    possition_embeddings[:org_possition_embeddings_size, :] = new_model_weights[
                        "module.entity_embeddings.position_embeddings.weight"
                    ]
                    new_model_weights["module.entity_embeddings.position_embeddings.weight"] = possition_embeddings

                if (
                    model.module.embeddings.position_embeddings.num_embeddings
                    != new_model_weights["module.embeddings.position_embeddings.weight"].shape[0]
                ):
                    org_possition_embeddings = new_model_weights["module.embeddings.position_embeddings.weight"]
                    org_possition_embeddings_size = org_possition_embeddings.shape[0]
                    possition_embeddings_size = model.module.embeddings.position_embeddings.num_embeddings
                    possition_embeddings_hiden = model.module.embeddings.position_embeddings.embedding_dim
                    possition_embeddings = org_possition_embeddings.new_zeros(
                        (possition_embeddings_size, possition_embeddings_hiden)
                    )
                    possition_embeddings[:org_possition_embeddings_size, :] = new_model_weights[
                        "module.embeddings.position_embeddings.weight"
                    ]
                    new_model_weights["module.embeddings.position_embeddings.weight"] = possition_embeddings

                del orig_entity_bias, entity_emb, entity_bias
            del orig_entity_emb

        miss_layers, new_layers = model.load_state_dict(new_model_weights, strict=False)
        epoch, global_step = 0, 0
    else:
        _, checkpoint_state_dict = model.load_checkpoint(path, checkpoint_id, **kwargs)
        epoch = checkpoint_state_dict["epoch"]
        global_step = checkpoint_state_dict["global_step"]
    return epoch, global_step


def load_dataset(dataset_dir: str) -> List[WikipediaPretrainingDataset]:
    dataset_dirs = glob.glob(dataset_dir)
    assert len(dataset_dirs) > 0, f"No matching directories with {dataset_dir}"
    datasets = [WikipediaPretrainingDataset(d) for d in dataset_dirs]

    # Check if the attributes are all the same
    # Note that these two checks are not strict, only compare the length of the object
    assert len(frozenset([len(d.tokenizer) for d in datasets])) == 1
    # assert len(frozenset([len(d.entity_vocab) for d in datasets])) == 1
    assert len(frozenset([d.max_seq_length for d in datasets])) == 1
    assert len(frozenset([d.max_entity_length for d in datasets])) == 1
    assert len(frozenset([d.max_mention_length for d in datasets])) == 1

    return datasets


def create_table_model_and_config(
    args: Namespace,
    entity_vocab_size: int,
    local_rank: int,
    max_num_rows: int = 30,
    max_num_cols: int = 20,
    max_cell_value_length: int = 128,
    max_cell_tokens: int = 30,
    reset_position_index_per_cell: bool = True,
) -> Tuple[LukeTablePretrainingModel, LukeTableConfig]:
    bert_config = AutoConfig.from_pretrained(args.bert_model_name)

    config = LukeTableConfig(
        entity_vocab_size=entity_vocab_size,
        bert_model_name=args.bert_model_name,
        entity_emb_size=args.entity_emb_size,
        cls_entity_prediction=args.cls_entity_prediction,
        **bert_config.to_dict(),
        max_num_rows=max_num_rows,
        max_num_cols=max_num_cols,
        max_cell_value_length=max_cell_value_length,
        max_cell_tokens=max_cell_tokens,
        reset_position_index_per_cell=reset_position_index_per_cell,
    )

    model = LukeTablePretrainingModel(config)

    if args.fix_bert_weights:
        for param in model.parameters():
            param.requires_grad = False

        if not args.fix_entity_weights:
            for param in model.entity_embeddings.parameters():
                param.requires_grad = True

            for param in model.entity_predictions.parameters():
                param.requires_grad = True

        # Add relationship prediction parameters
        for param in model.entity_hor_rel_predictions.parameters():
            param.requires_grad = True
        for param in model.entity_ver_rel_predictions.parameters():
            param.requires_grad = True
        for param in model.word_hor_rel_predictions.parameters():
            param.requires_grad = True
        for param in model.word_ver_rel_predictions.parameters():
            param.requires_grad = True

        # col and row embeddings
        for param in model.embeddings.position_row_embeddings.parameters():
            param.requires_grad = True
        for param in model.embeddings.position_col_embeddings.parameters():
            param.requires_grad = True
        for param in model.entity_embeddings.position_row_embeddings.parameters():
            param.requires_grad = True
        for param in model.entity_embeddings.position_col_embeddings.parameters():
            param.requires_grad = True

    if args.word_only_training:
        model.entity_embeddings = None
        model.entity_predictions = None

    if args.disable_entity_prediction_bias:
        model.entity_predictions.bias.requires_grad = False

    if args.freeze_entity_emb:
        model.entity_embeddings.entity_embeddings.weight.requires_grad = False
    print("Train layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    return model, config


def create_model_and_config(
    args: Namespace, entity_vocab_size: int, local_rank: int
) -> Tuple[LukePretrainingModel, LukeConfig]:
    bert_config = AutoConfig.from_pretrained(args.bert_model_name)

    config = LukeConfig(
        entity_vocab_size=entity_vocab_size,
        bert_model_name=args.bert_model_name,
        entity_emb_size=args.entity_emb_size,
        cls_entity_prediction=args.cls_entity_prediction,
        **bert_config.to_dict(),
    )

    model = LukePretrainingModel(config)

    if args.fix_bert_weights:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.entity_embeddings.parameters():
            param.requires_grad = True
        for param in model.entity_predictions.parameters():
            param.requires_grad = True

    if args.word_only_training:
        model.entity_embeddings = None
        model.entity_predictions = None

    if args.disable_entity_prediction_bias:
        model.entity_predictions.bias.requires_grad = False

    if args.freeze_entity_emb:
        model.entity_embeddings.entity_embeddings.weight.requires_grad = False

    return model, config


def create_optimizer_grouped_parameters(
    model: Union[LukePretrainingModel, LukeTablePretrainingModel], weight_decay: float
) -> List[dict]:
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if "pooler" not in n[0] and n[1].requires_grad]
    # parameter names for huggingface transformers
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    return grouped_parameters


def load_state_dict(
    state_dict: dict, model: Union[LukePretrainingModel, LukeTablePretrainingModel], config: LukeConfig
):
    new_state_dict = {}
    for key, param in state_dict.items():
        key = key.replace("gamma", "weight").replace("beta", "bias")
        if key.startswith("roberta."):
            key = key[8:]
        elif key.startswith("bert."):
            key = key[5:]
        elif key.startswith("longformer."):
            key = key[11:]
        new_state_dict[key] = param
    state_dict = new_state_dict

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    assert len(unexpected_keys) == 0
    logger.debug(f"missing keys: {str(missing_keys)}")


@click.command()
@click.option("--dataset-dir", type=click.Path(), required=True)
@click.option("--train-batch-size", type=int, required=True)
@click.option("--num-epochs", type=int, required=True)
def compute_total_training_steps(dataset_dir, train_batch_size, num_epochs):
    dataset_directories = glob.glob(dataset_dir)
    if len(dataset_directories) == 0:
        raise ValueError(f"{dataset_dir} does not match any directory.")

    total_dataset_size = 0
    for dataset_dir in dataset_directories:
        datasets = load_dataset(dataset_dir)
        dataset_size = sum([len(d) for d in datasets])
        total_dataset_size += dataset_size
    train_steps = math.ceil(total_dataset_size / train_batch_size * num_epochs)
    print("Total training steps:", train_steps)


@click.command()
@click.option("--dataset-dir", default="luke_pretraining_dataset", type=click.Path(), required=True)
@click.option("--output-dir", default="luke_models", type=click.Path(), required=True)
@click.option(
    "--deepspeed-config-file",
    default="pretraining_config/luke_base_stage1.json",
    type=click.Path(exists=True),
    required=True,
)
@click.option("--log-dir", type=click.Path())
@click.option("--bert-model-name", default="roberta-base")
@click.option("--cls-entity-prediction", is_flag=True)
@click.option("--disable-entity-prediction-bias", is_flag=True)
@click.option("--entity-emb-size", default=256, type=int)
@click.option("--fix-bert-weights", is_flag=True)
@click.option("--fix-entity-weights", is_flag=True)
@click.option("--local_rank", type=int)  # specified by the deepspeed launcher
@click.option("--mask-words-in-entity-span", is_flag=True)
@click.option("--masked-lm-prob", default=0.15)
@click.option("--masked-entity-prob", default=0.15)
@click.option("--num-epochs", default=20)
@click.option("--random-entity-prob", default=0.0)
@click.option("--random-word-prob", default=0.1)
@click.option("--reset-optimization-states", is_flag=True)
@click.option("--resume-checkpoint-id", type=str, default=None)
@click.option("--resume-hf-checkpoint-file", type=str, default=None)
@click.option("--sampling-smoothing", default=0.7)
@click.option("--save-interval-steps", type=int)
@click.option("--unmasked-entity-prob", default=0.0)
@click.option("--unmasked-word-prob", default=0.1)
@click.option("--whole-word-masking/--subword-masking", default=True)
@click.option("--word-only-training", is_flag=True)
@click.option("--freeze-entity-emb", is_flag=True)
@click.option("--from-tables", is_flag=True)
@click.option("--learn-relations", is_flag=True)
def pretrain(**kwargs):
    global logger

    import deepspeed
    from deepspeed.utils.logging import LoggerFactory

    logger = LoggerFactory.create_logger(__name__)

    args = Namespace(**kwargs)

    deepspeed.init_distributed(dist_backend="nccl")

    tf.config.set_visible_devices([], "GPU")

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    is_master_process = bool(local_rank == -1 or global_rank == 0)
    if is_master_process:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARN)

    if "deepspeed_config_file" in args:
        with open(args.deepspeed_config_file) as f:
            args.deepspeed_config = json.load(f)

    logger.info("Arguments: %s", json.dumps(vars(args), indent=2, sort_keys=True))

    logger.info(f"Output dir: {args.output_dir}")
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info("Preparing datasets...")
    datasets = load_dataset(args.dataset_dir)
    representative_dataset = datasets[0]
    entity_vocab = representative_dataset.entity_vocab

    logger.info("Instantiating the model...")
    if args.from_tables:
        model, config = create_table_model_and_config(
            args,
            entity_vocab_size=entity_vocab.size,
            local_rank=local_rank,
            max_num_rows=representative_dataset.max_num_rows,
            max_num_cols=representative_dataset.max_num_cols,
            max_cell_tokens=representative_dataset.max_cell_tokens,
            reset_position_index_per_cell=representative_dataset.reset_position_index_per_cell,
        )
    else:
        model, config = create_model_and_config(args, entity_vocab_size=entity_vocab.size, local_rank=local_rank)

    logger.info("Model configuration: %s", config)

    logger.info("Initializing Deepspeed...")

    weight_decay = args.deepspeed_config.get("optimizer", {}).get("params", {}).get("weight_decay", 0.01)
    optimizer_parameters = create_optimizer_grouped_parameters(model, weight_decay)
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        config=args.deepspeed_config, model=model, model_parameters=optimizer_parameters
    )
    # lr_scheduler.step()

    epoch = 0
    global_step = 0

    if args.resume_checkpoint_id is None:
        if args.resume_hf_checkpoint_file is None:
            logger.info(f"Initializing the model parameters from pretrained {args.bert_model_name}...")
            state_dict = AutoModelForPreTraining.from_pretrained(args.bert_model_name).state_dict()
        else:
            logger.info(f"Initializing the model parameters from {args.resume_hf_checkpoint_file}...")
            state_dict = torch.load(args.resume_hf_checkpoint_file, map_location="cpu")
        load_state_dict(state_dict, model.module, config)
        state_dict = None

    else:
        logger.info(f"Loading checkpoint: {args.resume_checkpoint_id}")
        target_checkpoint_dir, target_checkpoint_id = os.path.split(args.resume_checkpoint_id)
        if not target_checkpoint_dir and not os.path.exists(target_checkpoint_id):
            target_checkpoint_dir = checkpoint_dir

        if args.reset_optimization_states:
            load_module_strict = True
            if args.cls_entity_prediction:
                load_module_strict = False  # CLSEntityPredictionHead is newly added to the model
            load_module_strict = False

            _, _ = load_checkpoint(
                target_checkpoint_dir,
                target_checkpoint_id,
                model,
                entity_vocab=entity_vocab,
                config=config,
                load_module_strict=load_module_strict,
                load_optimizer_states=False,
                load_lr_scheduler_states=False,
            )
        else:
            epoch, global_step = load_checkpoint(
                target_checkpoint_dir,
                target_checkpoint_id,
                model,
                entity_vocab=entity_vocab,
                config=config,
                load_optimizer_states=True,
                load_lr_scheduler_states=True,
            )
        logger.info(f"epoch: {epoch} global_step: {global_step}")

    if is_master_process:
        logger.info(f"Saving metadata and entity vocabulary files in {args.output_dir}")
        entity_vocab.save(os.path.join(args.output_dir, ENTITY_VOCAB_FILE))
        metadata = dict(
            model_config=config.to_dict(),
            max_seq_length=representative_dataset.max_seq_length,
            max_entity_length=representative_dataset.max_entity_length,
            max_mention_length=representative_dataset.max_mention_length,
            arguments=vars(args),
        )
        with open(os.path.join(args.output_dir, METADATA_FILE), "w") as metadata_file:
            json.dump(metadata, metadata_file, indent=2, sort_keys=True)

    dataset_size = sum([len(d) for d in datasets])
    train_batch_size = args.deepspeed_config["train_batch_size"]
    train_micro_batch_size_per_gpu = args.deepspeed_config["train_micro_batch_size_per_gpu"]
    num_train_steps_per_epoch = math.ceil(dataset_size / train_batch_size)
    num_train_steps = math.ceil(dataset_size / train_batch_size * args.num_epochs)
    gradient_accumulation_steps = model.gradient_accumulation_steps()

    batch_generator = LukePretrainingBatchGenerator(
        datasets,
        batch_size=train_micro_batch_size_per_gpu,
        masked_lm_prob=args.masked_lm_prob,
        masked_entity_prob=args.masked_entity_prob,
        whole_word_masking=args.whole_word_masking,
        unmasked_word_prob=args.unmasked_word_prob,
        random_word_prob=args.random_word_prob,
        unmasked_entity_prob=args.unmasked_entity_prob,
        random_entity_prob=args.random_entity_prob,
        mask_words_in_entity_span=args.mask_words_in_entity_span,
        num_workers=dist.get_world_size(),
        worker_index=global_rank,
        starting_step=int(global_step * train_batch_size),
        cls_entity_prediction=args.cls_entity_prediction,
        word_only=args.word_only_training,
        from_tables=args.from_tables,
    )

    model.train()

    model_outputs = []
    prev_step_time = time.time()
    if is_master_process:
        summary_writer = SummaryWriter(args.log_dir)
        pbar = tqdm(total=num_train_steps, initial=global_step)

    for batch_index, batch in enumerate(batch_generator.generate_batches()):
        batch = {k: torch.from_numpy(v).to(model.device) for k, v in batch.items()}
        model_output = model(**batch)
        loss = model_output["loss"]

        model_outputs.append({k: v.to("cpu").detach().numpy() for k, v in model_output.items()})
        model_output = None

        model.backward(loss)
        model.step()

        if (batch_index + 1) % gradient_accumulation_steps == 0:
            summary = model.module.get_metrics(reset=True)

            summary["learning_rate"] = max(lr_scheduler.get_last_lr())
            summary["loss"] = np.concatenate([r["loss"].flatten() for r in model_outputs]).mean()

            current_time = time.time()
            summary["batch_run_time"] = current_time - prev_step_time
            prev_step_time = current_time

            model_outputs = []

            if is_master_process:
                for name, value in summary.items():
                    summary_writer.add_scalar(name, value, global_step)
                desc = (
                    f"epoch: {int(global_step / num_train_steps_per_epoch)} "
                    f'loss: {summary["loss"]:.4f} '
                    f'time: {datetime.datetime.now().strftime("%H:%M:%S")}'
                )
                pbar.set_description(desc)
                pbar.update()

            global_step += 1

            current_epoch = int(global_step / num_train_steps_per_epoch)
            if global_step == num_train_steps:
                save_checkpoint(checkpoint_dir, f"epoch{args.num_epochs}", model, args.num_epochs, global_step)

            # elif global_step % num_train_steps_per_epoch == 0:
            # save_checkpoint(checkpoint_dir, f"epoch{current_epoch}", model, current_epoch, global_step)

            if args.save_interval_steps and global_step % args.save_interval_steps == 0:
                save_checkpoint(checkpoint_dir, f"step{global_step:07}", model, current_epoch, global_step)

        if global_step == num_train_steps:
            break

    if is_master_process:
        summary_writer.close()


if __name__ == "__main__":
    pretrain()
