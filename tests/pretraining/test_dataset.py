import tempfile
from dataclasses import dataclass
from typing import List

from luke.pretraining.dataset import WikipediaPretrainingDataset
from luke.utils.entity_vocab import EntityVocab
from luke.utils.sentence_splitter import SentenceSplitter
from transformers import AutoTokenizer
from wikipedia2vec.dump_db import DumpDB


@dataclass
class WikiLink:
    text: str
    title: str
    start: int
    end: int


@dataclass
class Paragraph:
    text: str
    wiki_links: List[WikiLink] = None
    abstract: str = None


SAMPLE_PARAGRAPHS = {
    "Japan": [
        Paragraph(
            "Japan is an island country in East Asia. It is situated in the northwest Pacific Ocean.",
            wiki_links=[
                WikiLink("Japan", "Japan", 0, 5),
                WikiLink("East Asia", "East Asia", 30, 39),
                WikiLink("Pacific Ocean", "Pacific Ocean", 73, 86),
            ],
        )
    ],
    "Studio Ousia": [
        Paragraph(
            "Studio Ousia develops advanced multilingual natural language AI.",
            wiki_links=[
                WikiLink("Studio Ousia", "Studio Ousia", 0, 12),
                WikiLink("AI", "Artificial Intelligence", 61, 63),
            ],
        ),
        Paragraph(
            "Our award-winning AI will accelerate your business.",
            wiki_links=[
                WikiLink("AI", "Artificial Intelligence", 18, 20),
            ],
        ),
    ],
}


class DummyDumpDB(DumpDB):

    language = None

    def __init__(self):
        pass

    def get_paragraphs(self, page_title: str):
        return SAMPLE_PARAGRAPHS[page_title]

    def is_disambiguation(self, title: str):
        return False

    def is_redirect(self, title: str):
        return False

    def resolve_redirect(self, title: str):
        return title

    def titles(self):
        return list(SAMPLE_PARAGRAPHS.keys())



def test_build_and_read_dataset():
    dummy_dump_db = DummyDumpDB()

    tokenizer_name = "roberta-base"
    sentence_tokenizer = "icu"
    entity_vocab_file = "/home/phuc/git/luke/tests/fixtures/dummy_entity_vocab.tsv"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    sentence_tokenizer = SentenceSplitter.from_name(sentence_tokenizer)

    entity_vocab = EntityVocab(entity_vocab_file)

    with tempfile.TemporaryDirectory() as temp_directory_path:

        WikipediaPretrainingDataset.build(
            dummy_dump_db,
            tokenizer,
            sentence_tokenizer,
            entity_vocab,
            temp_directory_path,
            max_seq_length=512,
            max_entity_length=128,
            max_mention_length=30,
            min_sentence_length=5,
            abstract_only=False,
            include_sentences_without_entities=True,
            include_unk_entities=True,
            pool_size=1,
            chunk_size=1,
            max_num_documents=None,
            predefined_entities_only=False,
        )

        dataset = WikipediaPretrainingDataset(temp_directory_path)
        entity_vocab = dataset.entity_vocab
        tokenizer = dataset.tokenizer
        items = [item for item in dataset.create_iterator(shuffle=False, repeat=False)]
        # the order of the items seems to be stochastic due to multiprocessing in build
        # so we sort the items here
        items = sorted(items, key=lambda x: x["page_id"])

        item = items[0]
        assert entity_vocab.get_title_by_id(item["page_id"]) == "Japan"
        assert (
            tokenizer.decode(item["word_ids"]).strip()
            == "Japan is an island country in East Asia. It is situated in the northwest Pacific Ocean."
        )
        for entity_id, entity_position_ids, expected_title, expected_mention in zip(
            item["entity_ids"],
            item["entity_position_ids"],
            ["Japan", "East Asia", "Pacific Ocean"],
            ["Japan", "East Asia", "Pacific Ocean"],
        ):
            assert entity_vocab.get_title_by_id(entity_id) == expected_title
            assert (
                tokenizer.decode(item["word_ids"][[i for i in entity_position_ids if i > -1]]).strip()
                == expected_mention
            )

        item = items[1]
        assert entity_vocab.get_title_by_id(item["page_id"]) == "Studio Ousia"
        assert (
            tokenizer.decode(item["word_ids"]).strip()
            == "Studio Ousia develops advanced multilingual natural language AI. Our award-winning AI will accelerate your business."
        )
        for entity_id, entity_position_ids, expected_title, expected_mention in zip(
            item["entity_ids"],
            item["entity_position_ids"],
            ["Studio Ousia", "Artificial Intelligence", "Artificial Intelligence"],
            ["Studio Ousia", "AI", "AI"],
        ):
            assert entity_vocab.get_title_by_id(entity_id) == expected_title
            assert (
                tokenizer.decode(item["word_ids"][[i for i in entity_position_ids if i > -1]]).strip()
                == expected_mention
            )
