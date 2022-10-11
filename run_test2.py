"""
---
title: Fixed Positional Encodings
summary: >
  Implementation with explanation of fixed positional encodings as
  described in paper Attention is All You Need.
---
# Fixed Positional Encodings
The positional encoding encodes the position along the sequence into
 a vector of size `d_model`.
\begin{align}
PE_{p,2i} &= sin\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg) \\
PE_{p,2i + 1} &= cos\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg)
\end{align}
Where $1 \leq 2i, 2i + 1 \leq d_{model}$
 are the feature indexes in the encoding, and $p$ is the position.
"""
import json
import math
import multiprocessing

import numpy as np
import psutil
import torch
import torch.nn as nn
from tqdm import tqdm

from luke.utils.entity_vocab import EntityVocab
from luke.utils.wikipedia_parser import WikiDumpReader, WikipediaDumpDB, show_dump_db_stats


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout_prob: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)

        self.register_buffer("positional_encodings", get_positional_encoding(d_model, max_len), False)

    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[: x.shape[0]].detach().requires_grad_(False)
        x = x + pe
        x = self.dropout(x)
        return x


def get_positional_encoding(d_model: int, max_len: int = 5000):
    # Empty encodings vectors
    encodings = torch.zeros(max_len, d_model)
    # Position indexes
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    # $2 * i$
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32)
    # $10000^{\frac{2i}{d_{model}}}$
    div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))
    # $PE_{p,2i} = sin\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg)$
    encodings[:, 0::2] = torch.sin(position * div_term)
    # $PE_{p,2i + 1} = cos\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg)$
    encodings[:, 1::2] = torch.cos(position * div_term)

    # Add batch dimension
    encodings = encodings.unsqueeze(1).requires_grad_(False)

    return encodings


def _test_positional_encoding():
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5))
    pe = get_positional_encoding(20, 100)
    plt.plot(np.arange(100), pe[:, 0, 4:8].numpy())
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    plt.title("Positional encoding")
    plt.savefig("image_1.png")
    debug = 1


def test_wikitextparser():
    import wikitextparser as wtp

    t = wtp.parse(
        """{| class="wikitable sortable"
        |- 
        ! a !! b !! c 
        |- 
        ! colspan = "2" | [[African American|Black]] || e 
        |- 
        |}"""
    )
    tmp = 1


def check_overlapping():
    e500k = set()
    with open("data/luke_table_entity_vocab_500k.jsonl", "r") as f:
        for line in tqdm(f):
            e500k.add(json.loads(line)["entities"][0][0])
        print(f"500k: {len(e500k):,}")

    with open("data/luke_table_entity_vocab_test.jsonl", "r") as f:
        for line in tqdm(f):
            e500k.add(json.loads(line)["entities"][0][0])
        print(f"500k+128k: {len(e500k):,}")

    eall = {}
    with open("data/luke_table_entity_vocab_all.jsonl", "r") as f:
        for line in tqdm(f):
            eall[json.loads(line)["entities"][0][0]] = json.loads(line)["count"]
        print(f"All: {len(eall):,}")

    with open("data/luke_table_entity_vocab_500k_128k.jsonl", "w") as f:
        for ent_id, (title, count) in enumerate(eall.items()):
            if title not in e500k:
                continue
            json.dump({"id": ent_id, "entities": [[title, "en"]], "count": count}, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    # ps -ef | grep python
    # sudo fuser -v /dev/nvidia*
    # exit()
    check_overlapping()
    exit()
    n_threads = 1
    n_ram_limit = psutil.virtual_memory().total // 4  # 25% of ram
    chunk_size = 100
    out_file = "data/enwiki_table.db"
    dump_file = "data/enwiki-20181220-pages-articles-multistream.xml.bz2"
    # dump_reader = WikiDumpReader(dump_file)

    # print(f"Threads {n_threads} - Ram {n_ram_limit // (1024 **3)} GB")
    # db = WikipediaDumpDB.build(
    #     dump_reader, out_file, pool_size=n_threads, chunk_size=chunk_size, buffer_size=n_ram_limit
    # )
    # db = WikipediaDumpDB(out_file)
    # pagraprahs = db.get_paragraphs("1946 European Athletics Championships – Men's high jump")
    # tables = db.get_tables("1946 European Athletics Championships – Men's high jump")

    show_dump_db_stats(dump_db_file="data/enwiki_table.db", pool_size=multiprocessing.cpu_count(), chunk_size=100)

    print("Done")
