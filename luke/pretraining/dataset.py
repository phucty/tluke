import functools
import itertools
import json
import logging
import multiprocessing
import os
import random
from collections import defaultdict
from contextlib import closing
from multiprocessing.pool import Pool
from operator import invert
from pydoc import describe
from typing import Optional

import click
import tensorflow as tf
import transformers
from tensorflow.io import TFRecordWriter
from tensorflow.train import Int64List
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
from wikipedia2vec.dump_db import DumpDB

from luke.pretraining.tokenization import tokenize, tokenize_segments
from luke.utils.entity_vocab import UNK_TOKEN, EntityVocab
from luke.utils.model_utils import ENTITY_VOCAB_FILE, METADATA_FILE, get_entity_vocab_file_path
from luke.utils.sentence_splitter import SentenceSplitter
from luke.utils.wikipedia_parser import Paragraph, WikiLink, WikipediaDumpDB

logger = logging.getLogger(__name__)

DATASET_FILE = "dataset.tf"

# global variables used in pool workers
_dump_db = _tokenizer = _sentence_splitter = _entity_vocab = _max_num_tokens = _max_entity_length = None
_max_mention_length = _min_sentence_length = _include_sentences_without_entities = _include_unk_entities = None
_abstract_only = _language = None


@click.command()
@click.argument("dump_db_file", type=click.Path(exists=True))
@click.argument("tokenizer_name")
@click.argument("entity_vocab_file", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path(file_okay=False))
@click.option("--sentence-splitter", default="en")
@click.option("--max-seq-length", default=512)
@click.option("--max-entity-length", default=128)
@click.option("--max-mention-length", default=30)
@click.option("--min-sentence-length", default=5)
@click.option("--abstract-only", is_flag=True)
@click.option("--include-sentences-without-entities", is_flag=True)
@click.option("--include-unk-entities/--skip-unk-entities", default=False)
@click.option("--pool-size", default=multiprocessing.cpu_count())
@click.option("--chunk-size", default=100)
@click.option("--max-num-documents", default=None, type=int)
@click.option("--predefined-entities-only", is_flag=True)
@click.option("--from-tables", is_flag=True)
@click.option("--get-metadata", is_flag=True)
@click.option("--get-headers", is_flag=True)
@click.option("--get-outside-text", is_flag=True)
@click.option("--max-num-rows", default=30)
@click.option("--max-num-cols", default=20)
@click.option("--max-cell-tokens", default=30)
@click.option("--max-length-metadata", default=128)
@click.option("--max-header-tokens", default=30)
@click.option("--max-outside-text-tokens", default=128)
@click.option("--reset-position-index-per-cell", is_flag=True)
def build_wikipedia_pretraining_dataset(
    dump_db_file: str, tokenizer_name: str, entity_vocab_file: str, output_dir: str, sentence_splitter: str, **kwargs
):
    dump_db = WikipediaDumpDB(dump_db_file)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    sentence_splitter = SentenceSplitter.from_name(sentence_splitter)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    entity_vocab = EntityVocab(entity_vocab_file)
    WikipediaPretrainingDataset.build(dump_db, tokenizer, sentence_splitter, entity_vocab, output_dir, **kwargs)


class WikipediaPretrainingDataset:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        with open(os.path.join(dataset_dir, METADATA_FILE)) as metadata_file:
            self.metadata = json.load(metadata_file)

    def __len__(self):
        return self.metadata["number_of_items"]

    @property
    def max_seq_length(self):
        return self.metadata["max_seq_length"]

    @property
    def max_entity_length(self):
        return self.metadata["max_entity_length"]

    @property
    def max_mention_length(self):
        return self.metadata["max_mention_length"]

    @property
    def max_num_rows(self):
        return self.metadata["max_num_rows"]

    @property
    def max_num_cols(self):
        return self.metadata["max_num_cols"]

    @property
    def max_length_metadata(self):
        return self.metadata["max_length_metadata"]

    @property
    def max_header_tokens(self):
        return self.metadata["max_header_tokens"]

    @property
    def max_outside_text_tokens(self):
        return self.metadata["max_outside_text_tokens"]

    @property
    def max_cell_tokens(self):
        return self.metadata["max_cell_tokens"]

    @property
    def reset_position_index_per_cell(self):
        return self.metadata["reset_position_index_per_cell"]

    @property
    def language(self):
        return self.metadata.get("language", None)

    @property
    def tokenizer(self):
        tokenizer_class_name = self.metadata.get("tokenizer_class", "")
        tokenizer_class = getattr(transformers, tokenizer_class_name)
        return tokenizer_class.from_pretrained(self.dataset_dir)

    @property
    def entity_vocab(self):
        vocab_file_path = get_entity_vocab_file_path(self.dataset_dir)
        return EntityVocab(vocab_file_path)

    def create_table_iterator(
        self,
        skip: int = 0,
        num_workers: int = 1,
        worker_index: int = 0,
        shuffle_buffer_size: int = 1000,
        shuffle_seed: int = 0,
        num_parallel_reads: int = 10,
        repeat: bool = True,
        shuffle: bool = True,
    ):

        # The TensorFlow 2.0 has enabled eager execution by default.
        # At the starting of algorithm, we need to use this to disable eager execution.
        tf.compat.v1.disable_eager_execution()

        features = dict(
            word_ids=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            word_row_ids=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            word_col_ids=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            entity_ids=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            entity_position_ids=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            entity_position_row_ids=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            entity_position_col_ids=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            page_id=tf.io.FixedLenFeature([1], tf.int64, default_value=[0]),
        )
        dataset = tf.data.TFRecordDataset(
            [os.path.join(self.dataset_dir, DATASET_FILE)],
            compression_type="GZIP",
            num_parallel_reads=num_parallel_reads,
        )
        if repeat:
            dataset = dataset.repeat()
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size, seed=shuffle_seed)
        dataset = dataset.skip(skip)
        dataset = dataset.shard(num_workers, worker_index)
        dataset = dataset.map(functools.partial(tf.io.parse_single_example, features=features))
        it = tf.compat.v1.data.make_one_shot_iterator(dataset)
        it = it.get_next()

        with tf.compat.v1.Session() as sess:
            try:
                while True:
                    obj = sess.run(it)
                    yield dict(
                        page_id=obj["page_id"][0],
                        word_ids=obj["word_ids"],
                        word_row_ids=obj["word_row_ids"],
                        word_col_ids=obj["word_col_ids"],
                        entity_ids=obj["entity_ids"],
                        entity_position_row_ids=obj["entity_position_row_ids"],
                        entity_position_col_ids=obj["entity_position_col_ids"],
                        entity_position_ids=obj["entity_position_ids"].reshape(-1, self.metadata["max_mention_length"]),
                    )
            except tf.errors.OutOfRangeError:
                pass

    def create_iterator(
        self,
        skip: int = 0,
        num_workers: int = 1,
        worker_index: int = 0,
        shuffle_buffer_size: int = 1000,
        shuffle_seed: int = 0,
        num_parallel_reads: int = 10,
        repeat: bool = True,
        shuffle: bool = True,
    ):

        # The TensorFlow 2.0 has enabled eager execution by default.
        # At the starting of algorithm, we need to use this to disable eager execution.
        tf.compat.v1.disable_eager_execution()

        features = dict(
            word_ids=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            entity_ids=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            entity_position_ids=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            page_id=tf.io.FixedLenFeature([1], tf.int64, default_value=[0]),
        )
        dataset = tf.data.TFRecordDataset(
            [os.path.join(self.dataset_dir, DATASET_FILE)],
            compression_type="GZIP",
            num_parallel_reads=num_parallel_reads,
        )
        if repeat:
            dataset = dataset.repeat()
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size, seed=shuffle_seed)
        dataset = dataset.skip(skip)
        dataset = dataset.shard(num_workers, worker_index)
        dataset = dataset.map(functools.partial(tf.io.parse_single_example, features=features))
        it = tf.compat.v1.data.make_one_shot_iterator(dataset)
        it = it.get_next()

        with tf.compat.v1.Session() as sess:
            try:
                while True:
                    obj = sess.run(it)
                    yield dict(
                        page_id=obj["page_id"][0],
                        word_ids=obj["word_ids"],
                        entity_ids=obj["entity_ids"],
                        entity_position_ids=obj["entity_position_ids"].reshape(-1, self.metadata["max_mention_length"]),
                    )
            except tf.errors.OutOfRangeError:
                pass

    @classmethod
    def build(
        cls,
        dump_db: DumpDB,
        tokenizer: PreTrainedTokenizer,
        sentence_splitter: SentenceSplitter,
        entity_vocab: EntityVocab,
        output_dir: str,
        max_seq_length: int,
        max_entity_length: int,
        max_mention_length: int,
        min_sentence_length: int,
        abstract_only: bool,
        include_sentences_without_entities: bool,
        include_unk_entities: bool,
        pool_size: int,
        chunk_size: int,
        max_num_documents: Optional[int],
        predefined_entities_only: bool,
        from_tables: bool,
        get_metadata: bool,
        get_headers: bool,
        get_outside_text: bool,
        max_num_rows: int,
        max_num_cols: int,
        max_cell_tokens: int,
        max_length_metadata: int,
        max_header_tokens: int,
        max_outside_text_tokens: int,
        reset_position_index_per_cell: bool,
    ):

        target_titles = [
            title
            for title in dump_db.titles()
            if not (":" in title and title.lower().split(":")[0] in ("image", "file", "category"))
        ]

        if predefined_entities_only:
            lang = dump_db.language  # None <- entity_vocab の parse に合わせる
            target_titles = [title for title in target_titles if entity_vocab.contains(title, lang)]

        random.shuffle(target_titles)

        if max_num_documents is not None:
            target_titles = target_titles[:max_num_documents]

        max_num_tokens = max_seq_length - 2  # 2 for [CLS] and [SEP]

        tokenizer.save_pretrained(output_dir)

        entity_vocab.save(os.path.join(output_dir, ENTITY_VOCAB_FILE))
        number_of_items = 0
        tf_file = os.path.join(output_dir, DATASET_FILE)
        options = tf.io.TFRecordOptions(tf.compat.v1.io.TFRecordCompressionType.GZIP)

        if from_tables:
            callback_process_page = WikipediaPretrainingDataset._process_page_with_tluke
            obj_name = "Samples"
        else:
            callback_process_page = WikipediaPretrainingDataset._process_page
            obj_name = "Paragraph"

        n_obj, n_examples = 0, 0
        n_text, n_tdata, n_tpheader, n_tpmetadata, n_tptext = 0, 0, 0, 0, 0

        def update_desc():
            if from_tables:
                return f"{obj_name}: {n_obj:,} - {n_text:,}|{n_tdata:,}|{n_tpheader:,}|{n_tpmetadata:,}|{n_tptext:,}"
            else:
                return f"{obj_name}: {n_obj:,} - Examples: {n_examples:,}"

        with TFRecordWriter(tf_file, options=options) as writer:
            pbar = tqdm(total=len(target_titles), desc=update_desc(), mininterval=5)
            initargs = (
                dump_db,
                tokenizer,
                sentence_splitter,
                entity_vocab,
                max_num_tokens,
                max_entity_length,
                max_mention_length,
                min_sentence_length,
                abstract_only,
                include_sentences_without_entities,
                include_unk_entities,
                get_metadata,
                get_headers,
                get_outside_text,
                max_num_rows,
                max_num_cols,
                max_cell_tokens,
                max_length_metadata,
                max_header_tokens,
                max_outside_text_tokens,
                reset_position_index_per_cell,
            )
            # if pool_size == 1:
            # WikipediaPretrainingDataset._initialize_worker(*initargs)
            # for target_title in target_titles:
            # ret = callback_process_page(target_title)
            with closing(
                Pool(pool_size, initializer=WikipediaPretrainingDataset._initialize_worker, initargs=initargs)
            ) as pool:
                for ret in pool.imap(callback_process_page, target_titles, chunksize=chunk_size):
                    n_obj += 1
                    if from_tables:
                        ret_text, ret_tdata, ret_tpheader, ret_tpmetadata, ret_tptext = ret
                        n_text += len(ret_text)
                        n_tdata += len(ret_tdata)
                        n_tpheader += len(ret_tpheader)
                        n_tpmetadata += len(ret_tpmetadata)
                        n_tptext += len(ret_tptext)
                        ret = list(
                            itertools.chain.from_iterable(
                                [ret_text, ret_tdata, ret_tpheader, ret_tpmetadata, ret_tptext]
                            )
                        )
                    else:
                        n_examples += len(ret)

                    for data in ret:
                        writer.write(data)
                        number_of_items += 1
                    # if n_obj and n_obj % 10 == 0:
                    pbar.update()
                    pbar.set_description(update_desc())

        with open(os.path.join(output_dir, METADATA_FILE), "w") as metadata_file:
            json.dump(
                dict(
                    number_of_items=number_of_items,
                    max_seq_length=max_seq_length,
                    max_entity_length=max_entity_length,
                    max_mention_length=max_mention_length,
                    min_sentence_length=min_sentence_length,
                    tokenizer_class=tokenizer.__class__.__name__,
                    language=dump_db.language,
                    max_num_rows=max_num_rows,
                    max_num_cols=max_num_cols,
                    max_cell_tokens=max_cell_tokens,
                    max_length_metadata=max_length_metadata,
                    max_header_tokens=max_header_tokens,
                    max_outside_text_tokens=max_outside_text_tokens,
                    reset_position_index_per_cell=reset_position_index_per_cell,
                ),
                metadata_file,
                indent=2,
            )

    @staticmethod
    def _initialize_worker(
        dump_db: DumpDB,
        tokenizer: PreTrainedTokenizer,
        sentence_splitter: SentenceSplitter,
        entity_vocab: EntityVocab,
        max_num_tokens: int,
        max_entity_length: int,
        max_mention_length: int,
        min_sentence_length: int,
        abstract_only: bool,
        include_sentences_without_entities: bool,
        include_unk_entities: bool,
        get_metadata: bool,
        get_headers: bool,
        get_outside_text: bool,
        max_num_rows: int,
        max_num_cols: int,
        max_cell_tokens: int,
        max_length_metadata: int,
        max_header_tokens: int,
        max_outside_text_tokens: int,
        reset_position_index_per_cell: bool,
    ):
        global _dump_db, _tokenizer, _sentence_splitter, _entity_vocab, _max_num_tokens, _max_entity_length
        global _max_mention_length, _min_sentence_length, _include_sentences_without_entities, _include_unk_entities
        global _abstract_only
        global _language

        global _max_num_rows, _max_num_cols, _max_cell_value_length, _max_cell_tokens, _max_length_metadata, _max_header_tokens, _get_metadata, _reset_position_index_per_cell, _get_headers, _max_outside_text_tokens, _get_outside_text

        _dump_db = dump_db
        _tokenizer = tokenizer
        _sentence_splitter = sentence_splitter
        _entity_vocab = entity_vocab
        _max_num_tokens = max_num_tokens
        _max_entity_length = max_entity_length
        _max_mention_length = max_mention_length
        _min_sentence_length = min_sentence_length
        _include_sentences_without_entities = include_sentences_without_entities
        _include_unk_entities = include_unk_entities
        _abstract_only = abstract_only
        _language = dump_db.language

        _max_num_rows = max_num_rows
        _max_num_cols = max_num_cols
        _max_cell_tokens = max_cell_tokens
        _max_length_metadata = max_length_metadata
        _max_header_tokens = max_header_tokens
        _max_outside_text_tokens = max_outside_text_tokens
        _get_metadata = get_metadata
        _get_headers = get_headers
        _get_outside_text = get_outside_text
        _reset_position_index_per_cell = reset_position_index_per_cell

    @staticmethod
    def _process_page(page_title: str):
        if _entity_vocab.contains(page_title, _language):
            page_id = _entity_vocab.get_id(page_title, _language)
        else:
            page_id = -1

        sentences = []
        try:
            wikipedia_title = _dump_db.get_paragraphs(page_title)
        except Exception as message:
            print(f"Coud not load {page_title}")
            print(message)
            return []

        for paragraph in wikipedia_title:

            if _abstract_only and not paragraph.abstract:
                continue

            paragraph_text = paragraph.text

            # First, get paragraph links.
            # Parapraph links are represented its form (link_title) and the start/end positions of strings
            # (link_start, link_end).
            paragraph_links = []
            for link in paragraph.wiki_links:
                link_title = _dump_db.resolve_redirect(link.title)
                # remove category links
                if link_title.startswith("Category:") and link.text.lower().startswith("category:"):
                    paragraph_text = (
                        paragraph_text[: link.start] + " " * (link.end - link.start) + paragraph_text[link.end :]
                    )
                else:
                    if _entity_vocab.contains(link_title, _language):
                        paragraph_links.append((link_title, link.start, link.end))
                    elif _include_unk_entities:
                        paragraph_links.append((UNK_TOKEN, link.start, link.end))

            sent_spans = _sentence_splitter.get_sentence_spans(paragraph_text.rstrip())
            for sent_start, sent_end in sent_spans:
                cur = sent_start
                sent_words = []
                sent_links = []
                # Look for links that are within the tokenized sentence.
                # If a link is found, we separate the sentences across the link and tokenize them.
                for link_title, link_start, link_end in paragraph_links:
                    if not (sent_start <= link_start < sent_end and link_end <= sent_end):
                        continue
                    entity_id = _entity_vocab.get_id(link_title, _language)

                    sent_tokenized, link_words = tokenize_segments(
                        [paragraph_text[cur:link_start], paragraph_text[link_start:link_end]],
                        tokenizer=_tokenizer,
                        add_prefix_space=cur == 0 or paragraph_text[cur - 1] == " ",
                    )

                    sent_words += sent_tokenized

                    sent_links.append((entity_id, len(sent_words), len(sent_words) + len(link_words)))
                    sent_words += link_words
                    cur = link_end

                sent_words += tokenize(
                    text=paragraph_text[cur:sent_end],
                    tokenizer=_tokenizer,
                    add_prefix_space=cur == 0 or paragraph_text[cur - 1] == " ",
                )

                if len(sent_words) < _min_sentence_length or len(sent_words) > _max_num_tokens:
                    continue
                sentences.append((sent_words, sent_links))

        ret = []
        words = []
        links = []
        for i, (sent_words, sent_links) in enumerate(sentences):
            links += [(id_, start + len(words), end + len(words)) for id_, start, end in sent_links]
            words += sent_words
            if i == len(sentences) - 1 or len(words) + len(sentences[i + 1][0]) > _max_num_tokens:
                if links or _include_sentences_without_entities:
                    links = links[:_max_entity_length]
                    word_ids = _tokenizer.convert_tokens_to_ids(words)
                    assert _min_sentence_length <= len(word_ids) <= _max_num_tokens
                    entity_ids = [id_ for id_, _, _, in links]
                    assert len(entity_ids) <= _max_entity_length
                    entity_position_ids = itertools.chain(
                        *[
                            (list(range(start, end)) + [-1] * (_max_mention_length - end + start))[:_max_mention_length]
                            for _, start, end in links
                        ]
                    )

                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature=dict(
                                page_id=tf.train.Feature(int64_list=tf.train.Int64List(value=[page_id])),
                                word_ids=tf.train.Feature(int64_list=tf.train.Int64List(value=word_ids)),
                                entity_ids=tf.train.Feature(int64_list=tf.train.Int64List(value=entity_ids)),
                                entity_position_ids=tf.train.Feature(int64_list=Int64List(value=entity_position_ids)),
                            )
                        )
                    )
                    ret.append((example.SerializeToString()))

                words = []
                links = []
        return ret

    @staticmethod
    def _process_page_with_tluke(page_title: str):
        ret_text = WikipediaPretrainingDataset._process_page_text(page_title)
        ret_tdata, ret_tpheader, ret_tpmetadata, ret_tptext = WikipediaPretrainingDataset._process_page_with_tables(
            page_title
        )
        return ret_text, ret_tdata, ret_tpheader, ret_tpmetadata, ret_tptext

    @staticmethod
    def _process_page_with_tables(page_title: str):
        if _entity_vocab.contains(page_title, _language):
            page_id = _entity_vocab.get_id(page_title, _language)
        else:
            page_id = -1

        ret_data, ret_header, ret_metadata, ret_text = [], [], [], []
        try:
            tables = _dump_db.get_tables(page_title)
        except Exception as message:
            print(f"Coud not load {page_title}")
            print(message)
            return ret_data, ret_header, ret_metadata, ret_text
        if not tables:
            return ret_data, ret_header, ret_metadata, ret_text

        def resolve_links(paragraphs):
            ret_links = []
            ret_text = ""
            for paragraph in paragraphs:
                ret_text += paragraph.text
                for link in paragraph.wiki_links:
                    link_title = _dump_db.resolve_redirect(link.title)
                    # remove category links or invalid links
                    if link_title.startswith("Category:") and link.text.lower().startswith("category:"):
                        ret_text = ret_text[: link.start] + " " * (link.end - link.start) + ret_text[link.end :]
                    elif not _dump_db.is_wikipedia_page(link_title):
                        continue
                    else:
                        if _entity_vocab.contains(link_title, _language):
                            ret_links.append((link_title, link.start, link.end))
                        elif _include_unk_entities:
                            ret_links.append((UNK_TOKEN, link.start, link.end))
            return ret_text, ret_links

        for table in tables:
            if not table.cells:
                continue
            # Table links are represented its form (link_title) and the start/end positions of strings, and col, and row index
            # Metadata
            # Wikipedia title
            metadata_text, metadata_links = resolve_links(
                [
                    Paragraph(
                        text=table.title,
                        wiki_links=[WikiLink(table.title, table.title, 0, len(table.title))],
                        abstract=False,
                    )
                ]
            )
            # metadata_text = table.title
            # metadata is in col 0, row 0:  # [wikipedia title, row, col, link_start, link_end]
            metadata_links = [[title, 0, 0, start, end] for title, start, end in metadata_links]
            # Wikipedia section
            if table.section:
                # Get top section of Wikipedia
                section_text, _ = resolve_links(
                    [
                        Paragraph(
                            text=table.section[0],
                            wiki_links=[],
                            abstract=False,
                        )
                    ]
                )
                metadata_text += ". "
                metadata_text += section_text
            # Table caption
            if table.caption:
                caption_text, caption_links = resolve_links(table.caption)
                metadata_text += ". "
                offset = len(metadata_text)
                metadata_text += caption_text
                metadata_links += [[title, 0, 0, offset + start, offset + end] for title, start, end in caption_links]

            if table.out_text:
                out_text_text, out_text_links = resolve_links(table.out_text)
                out_text_links = [[title, 0, 0, start, end] for title, start, end in out_text_links]

            # Table headers concate table header rows --> first row: 0
            n_col = _max_num_cols if table.n_col > _max_num_cols else table.n_col
            header_text = ["" for _ in range(n_col)]
            header_links = [[] for _ in range(n_col)]
            header_text_set = [set() for _ in range(n_col)]
            header_rows = []
            if table.headers_cells:
                header_rows = defaultdict(list)
                for r, c in table.headers_cells:
                    header_rows[r].append(c)
                max_header_col = max(len(v) for v in header_rows.values())
                header_rows = {k: v for k, v in header_rows.items() if len(v) == max_header_col}
                # only get first 5 rows as headers
                header_rows = [r_i for r_i in range(5) if r_i in header_rows]
            if header_rows:
                # Go reverse to get more specific information
                for row_i in reversed(header_rows):
                    if row_i >= len(table.cells):
                        break
                    for col_i, cell in enumerate(table.cells[row_i][:_max_num_cols]):
                        cell_text, cell_links = resolve_links(cell.paragraphs)
                        if cell_text in header_text_set[col_i]:
                            continue

                        header_text_set[col_i].add(cell_text)

                        if header_text[col_i]:
                            header_text[col_i] += ". "
                        offset = len(header_text[col_i])
                        header_text[col_i] += cell_text
                        # Header is in the first row - 0
                        header_links[col_i] += [
                            [title, 0, col_i, offset + start, offset + end] for title, start, end in cell_links
                        ]
            # Table cells
            data_text = []
            data_links = []
            row_i = 0
            for i, row in enumerate(table.cells):
                if i in header_rows:
                    continue
                row_text = []
                row_links = []
                for col_i, cell in enumerate(row[:_max_num_cols]):
                    cell_text, cell_links = resolve_links(cell.paragraphs)
                    row_text.append(cell_text)
                    row_links.append([[title, row_i, col_i, start, end] for title, start, end in cell_links])
                row_i += 1
                data_text.append(row_text)
                data_links.append(row_links)

            def get_sentences(obj_text, obj_links, row_i, col_i, max_length):
                sentences = []
                sent_spans = _sentence_splitter.get_sentence_spans(obj_text.rstrip())
                for sent_start, sent_end in sent_spans:
                    cur = sent_start
                    sent_words = []
                    sent_links = []
                    # Look for links that are within the tokenized sentence.
                    # If a link is found, we separate the sentences across the link and tokenize them.
                    sent_tokenized = tokenize(
                        text=obj_text[cur:sent_end],
                        tokenizer=_tokenizer,
                        add_prefix_space=cur == 0 or obj_text[cur - 1] == " ",
                    )
                    if len(sent_words) > max_length:
                        break

                    is_break = False
                    for link_title, _, _, link_start, link_end in obj_links:
                        if not (sent_start <= link_start < sent_end and link_end <= sent_end):
                            continue
                        entity_id = _entity_vocab.get_id(link_title, _language)

                        sent_tokenized, link_words = tokenize_segments(
                            [obj_text[cur:link_start], obj_text[link_start:link_end]],
                            tokenizer=_tokenizer,
                            add_prefix_space=cur == 0 or obj_text[cur - 1] == " ",
                        )
                        if len(sent_words) + len(sent_tokenized) > max_length:
                            is_break = True
                            break

                        sent_words.extend([(token, row_i, col_i) for token in sent_tokenized])
                        sent_links.append((entity_id, row_i, col_i, len(sent_words), len(sent_words) + len(link_words)))
                        sent_words.extend([(token, row_i, col_i) for token in link_words])
                        cur = link_end
                    if is_break:
                        break
                    sent_words.extend([(token, row_i, col_i) for token in sent_tokenized])

                    # ignore min and max sentence length
                    # if len(sent_words) < _min_sentence_length or len(sent_words) > _max_num_tokens:
                    # continue

                    sentences.append((sent_words, sent_links))
                return sentences

            if table.out_text:
                sentences_out_text = get_sentences(
                    out_text_text, out_text_links, row_i=0, col_i=0, max_length=_max_outside_text_tokens
                )
            sentences_metadata = get_sentences(
                metadata_text, metadata_links, row_i=0, col_i=0, max_length=_max_length_metadata
            )
            # Column started at 1 (2nd)
            sentences_headers = [
                get_sentences(header_text[col_i], header_links[col_i], 0, col_i + 1, max_length=_max_header_tokens)
                for col_i in range(len(header_text))
            ]
            # Column started at 1 (2nd), row is started at 1 (2nd)
            sentences_data = []
            for row_i in range(len(data_text)):
                sentences_row = []
                for col_i in range(len(data_text[row_i])):
                    sentences_cell = get_sentences(
                        data_text[row_i][col_i],
                        data_links[row_i][col_i],
                        row_i + 1,
                        col_i + 1,
                        max_length=_max_cell_tokens,
                    )
                    sentences_row.append(sentences_cell)
                sentences_data.append(sentences_row)

            def gen_examples(words, links):
                links = links[:_max_entity_length]
                word_ids = _tokenizer.convert_tokens_to_ids([token for token, _, _ in words])
                word_row_ids = [row_id for _, row_id, _ in words]
                word_col_ids = [col_id for _, _, col_id in words]
                if not _min_sentence_length <= len(word_ids) <= _max_num_tokens:
                    return None

                assert _min_sentence_length <= len(word_ids) <= _max_num_tokens

                entity_ids = [id_ for id_, _, _, _, _, in links]
                assert len(entity_ids) <= _max_entity_length

                entity_position_ids = itertools.chain(
                    *[
                        (list(range(start, end)) + [-1] * (_max_mention_length - end + start))[:_max_mention_length]
                        for _, _, _, start, end in links
                    ]
                )
                entity_position_row_ids = [row_id for _, row_id, _, _, _, in links]
                entity_position_col_ids = [col_id for _, _, col_id, _, _, in links]

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature=dict(
                            page_id=tf.train.Feature(int64_list=tf.train.Int64List(value=[page_id])),
                            word_ids=tf.train.Feature(int64_list=tf.train.Int64List(value=word_ids)),
                            word_row_ids=tf.train.Feature(int64_list=tf.train.Int64List(value=word_row_ids)),
                            word_col_ids=tf.train.Feature(int64_list=tf.train.Int64List(value=word_col_ids)),
                            entity_ids=tf.train.Feature(int64_list=tf.train.Int64List(value=entity_ids)),
                            entity_position_ids=tf.train.Feature(int64_list=Int64List(value=entity_position_ids)),
                            entity_position_row_ids=tf.train.Feature(
                                int64_list=tf.train.Int64List(value=entity_position_row_ids)
                            ),
                            entity_position_col_ids=tf.train.Feature(
                                int64_list=tf.train.Int64List(value=entity_position_col_ids)
                            ),
                        )
                    )
                )
                return example

            def gen_example_with_setting(get_metadata: bool, get_headers: bool, get_text: bool):
                ret = []
                prefix_words = []
                prefix_links = []
                # Add metadata
                if get_metadata:
                    for sent_words, sent_links in sentences_metadata:
                        prefix_links += [
                            (id_, row, col, start + len(prefix_words), end + len(prefix_words))
                            for id_, row, col, start, end in sent_links
                        ]
                        prefix_words += sent_words

                # add outside text
                if get_text and table.out_text:
                    for sent_words, sent_links in sentences_out_text:
                        prefix_links += [
                            (id_, row, col, start + len(prefix_words), end + len(prefix_words))
                            for id_, row, col, start, end in sent_links
                        ]
                        prefix_words += sent_words

                # Add table headers
                if get_headers:
                    for header_cell in sentences_headers:
                        header_cell_text = []
                        header_cell_links = []
                        for sent_words, sent_links in header_cell:
                            if len(header_cell_text) + len(sent_words) > _max_header_tokens:
                                break
                            header_cell_text += sent_words
                            for id_, row, col, start, end in sent_links:
                                if end > len(header_cell_text):
                                    header_cell_text = header_cell_text[:start]
                                    break
                                header_cell_links.append(
                                    (id_, row, col, start + len(header_cell_text), end + len(header_cell_text))
                                )

                        prefix_links += [
                            (id_, row, col, start + len(prefix_words), end + len(prefix_words))
                            for id_, row, col, start, end in header_cell_links
                        ]
                        prefix_words += header_cell_text

                if len(prefix_words) > _max_num_tokens:
                    # logger.warn("Number tokens in table caption, and headers are larger than max_tokens")
                    return ret

                words = prefix_words.copy()
                links = prefix_links.copy()
                row_i = 1
                for sentence_i, sentences_row in enumerate(sentences_data):
                    row_links = []
                    row_words = []
                    for sentences_cell in sentences_row:
                        for sent_words, sent_links in sentences_cell:
                            cell_text = sent_words[:_max_cell_tokens]
                            cell_links = []
                            for id_, row_i_tmp, col_i_tmp, start, end in sent_links:
                                if end > len(cell_text):
                                    cell_text = cell_text[:start]
                                    break
                                cell_links.append(
                                    (id_, row_i_tmp, col_i_tmp, start + len(row_words), end + len(row_words))
                                )

                            row_links += cell_links
                            row_words += cell_text

                    if (
                        len(words) + len(row_words) > _max_num_tokens
                        or (sentence_i + 1 % _max_num_rows) == 0
                        or row_i > _max_num_rows
                    ):
                        if not links and not _include_sentences_without_entities:
                            continue
                        example = gen_examples(words, links)
                        if example:
                            ret.append((example.SerializeToString()))
                        words = prefix_words.copy()
                        links = prefix_links.copy()
                        # if _reset_position_index_per_cell:
                        row_i = 1

                    links += [
                        (token_id, row_i, c_i, start + len(words), end + len(words))
                        for token_id, r_i, c_i, start, end in row_links
                    ]
                    words += [(token_id, row_i, c_i) for token_id, r_i, c_i in row_words]
                    row_i += 1

                if len(prefix_words) < len(words) <= _max_num_tokens and (links or _include_sentences_without_entities):
                    example = gen_examples(words, links)
                    if example:
                        ret.append((example.SerializeToString()))
                return ret

            # Setting:
            # Just data
            ret_data.extend(gen_example_with_setting(get_metadata=False, get_headers=False, get_text=False))
            # with headers
            if _get_headers:
                ret_header.extend(gen_example_with_setting(get_metadata=False, get_headers=True, get_text=False))
            # with headers and metadata
            if _get_metadata:
                ret_metadata.extend(gen_example_with_setting(get_metadata=True, get_headers=True, get_text=False))
            # with text outside table
            if _get_outside_text:
                ret_text.extend(gen_example_with_setting(get_metadata=True, get_headers=True, get_text=True))

        return ret_data, ret_header, ret_metadata, ret_text

    @staticmethod
    def _process_page_text(page_title: str):
        if _entity_vocab.contains(page_title, _language):
            page_id = _entity_vocab.get_id(page_title, _language)
        else:
            page_id = -1

        sentences = []
        try:
            wikipedia_title = _dump_db.get_paragraphs(page_title)
        except Exception as message:
            print(f"Coud not load {page_title}")
            print(message)
            return []

        for paragraph in wikipedia_title:

            if _abstract_only and not paragraph.abstract:
                continue

            paragraph_text = paragraph.text

            # First, get paragraph links.
            # Parapraph links are represented its form (link_title) and the start/end positions of strings
            # (link_start, link_end).
            paragraph_links = []
            for link in paragraph.wiki_links:
                link_title = _dump_db.resolve_redirect(link.title)
                # remove category links
                if link_title.startswith("Category:") and link.text.lower().startswith("category:"):
                    paragraph_text = (
                        paragraph_text[: link.start] + " " * (link.end - link.start) + paragraph_text[link.end :]
                    )
                else:
                    if _entity_vocab.contains(link_title, _language):
                        paragraph_links.append((link_title, link.start, link.end))
                    elif _include_unk_entities:
                        paragraph_links.append((UNK_TOKEN, link.start, link.end))

            sent_spans = _sentence_splitter.get_sentence_spans(paragraph_text.rstrip())
            for sent_start, sent_end in sent_spans:
                cur = sent_start
                sent_words = []
                sent_links = []
                # Look for links that are within the tokenized sentence.
                # If a link is found, we separate the sentences across the link and tokenize them.
                for link_title, link_start, link_end in paragraph_links:
                    if not (sent_start <= link_start < sent_end and link_end <= sent_end):
                        continue
                    entity_id = _entity_vocab.get_id(link_title, _language)

                    sent_tokenized, link_words = tokenize_segments(
                        [paragraph_text[cur:link_start], paragraph_text[link_start:link_end]],
                        tokenizer=_tokenizer,
                        add_prefix_space=cur == 0 or paragraph_text[cur - 1] == " ",
                    )

                    sent_words += sent_tokenized

                    sent_links.append((entity_id, len(sent_words), len(sent_words) + len(link_words)))
                    sent_words += link_words
                    cur = link_end

                sent_words += tokenize(
                    text=paragraph_text[cur:sent_end],
                    tokenizer=_tokenizer,
                    add_prefix_space=cur == 0 or paragraph_text[cur - 1] == " ",
                )

                if len(sent_words) < _min_sentence_length or len(sent_words) > _max_num_tokens:
                    continue
                sentences.append((sent_words, sent_links))

        ret = []
        words = []
        links = []
        for i, (sent_words, sent_links) in enumerate(sentences):
            links += [(id_, start + len(words), end + len(words)) for id_, start, end in sent_links]
            words += sent_words
            if i == len(sentences) - 1 or len(words) + len(sentences[i + 1][0]) > _max_num_tokens:
                if links or _include_sentences_without_entities:
                    links = links[:_max_entity_length]
                    word_ids = _tokenizer.convert_tokens_to_ids(words)
                    assert _min_sentence_length <= len(word_ids) <= _max_num_tokens
                    entity_ids = [id_ for id_, _, _, in links]
                    assert len(entity_ids) <= _max_entity_length
                    entity_position_ids = itertools.chain(
                        *[
                            (list(range(start, end)) + [-1] * (_max_mention_length - end + start))[:_max_mention_length]
                            for _, start, end in links
                        ]
                    )

                    word_row_ids = [0] * len(word_ids)
                    word_col_ids = [0] * len(word_ids)
                    entity_position_row_ids = [0] * len(entity_ids)
                    entity_position_col_ids = [0] * len(entity_ids)

                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature=dict(
                                page_id=tf.train.Feature(int64_list=tf.train.Int64List(value=[page_id])),
                                word_ids=tf.train.Feature(int64_list=tf.train.Int64List(value=word_ids)),
                                word_row_ids=tf.train.Feature(int64_list=tf.train.Int64List(value=word_row_ids)),
                                word_col_ids=tf.train.Feature(int64_list=tf.train.Int64List(value=word_col_ids)),
                                entity_ids=tf.train.Feature(int64_list=tf.train.Int64List(value=entity_ids)),
                                entity_position_ids=tf.train.Feature(int64_list=Int64List(value=entity_position_ids)),
                                entity_position_row_ids=tf.train.Feature(
                                    int64_list=tf.train.Int64List(value=entity_position_row_ids)
                                ),
                                entity_position_col_ids=tf.train.Feature(
                                    int64_list=tf.train.Int64List(value=entity_position_col_ids)
                                ),
                            )
                        )
                    )
                    ret.append((example.SerializeToString()))

                words = []
                links = []
        return ret
