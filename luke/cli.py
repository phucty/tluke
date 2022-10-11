import logging
import multiprocessing
import os
import random
from email.policy import default

import click
import numpy as np
import torch
from wikipedia2vec.dump_db import DumpDB
from wikipedia2vec.utils.wiki_dump_reader import WikiDumpReader

from luke.utils.wikipedia_parser import WikiDumpReader, WikipediaDumpDB

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # filter out INFO messages from Tensordflow
try:
    # https://github.com/tensorflow/tensorflow/issues/27023#issuecomment-501419334
    from tensorflow.python.util import deprecation

    deprecation._PRINT_DEPRECATION_WARNINGS = False
except ImportError:
    pass


import debugpy

import luke.pretraining.dataset
import luke.pretraining.train
import luke.utils.convert_luke_to_huggingface_model
import luke.utils.entity_vocab
import luke.utils.interwiki_db
import luke.utils.model_utils

# debugpy.listen(5678)
# print("Wait for client")
# debugpy.wait_for_client()
# print("Attached")


@click.group()
@click.option("--verbose", is_flag=True)
@click.option("--seed", type=int, default=None)
def cli(verbose: bool, seed: int):
    fmt = "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format=fmt)
        logging.getLogger("transformers").setLevel(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO, format=fmt)
        logging.getLogger("transformers").setLevel(level=logging.WARNING)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@cli.command()
@click.argument("dump_file", type=click.Path(exists=True))
@click.argument("out_file", type=click.Path())
@click.option("--pool-size", default=multiprocessing.cpu_count())
@click.option("--chunk-size", type=int, default=100)
def build_dump_db(dump_file: str, out_file: str, **kwargs):
    dump_reader = WikiDumpReader(dump_file)
    DumpDB.build(dump_reader, out_file, **kwargs)


cli.add_command(luke.utils.entity_vocab.build_entity_vocab)
cli.add_command(luke.pretraining.dataset.build_wikipedia_pretraining_dataset)
cli.add_command(luke.pretraining.train.pretrain)
cli.add_command(luke.pretraining.train.compute_total_training_steps)
cli.add_command(luke.utils.interwiki_db.build_interwiki_db)
cli.add_command(luke.utils.entity_vocab.build_multilingual_entity_vocab)
cli.add_command(luke.utils.model_utils.create_model_archive)
cli.add_command(luke.utils.convert_luke_to_huggingface_model.convert_luke_to_huggingface_model)


cli.add_command(luke.utils.wikipedia_parser.build_table_dump_db)
cli.add_command(luke.utils.entity_vocab.build_table_entity_vocab)

if __name__ == "__main__":
    cli()
