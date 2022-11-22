import bz2
import csv
import logging
import multiprocessing
import pickle
import re
import zlib
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass
from functools import partial
from multiprocessing.pool import Pool
from turtle import title
from typing import List, TextIO, Tuple
from uuid import uuid1
from xml.etree.ElementTree import iterparse

import click
import lmdb
import mwparserfromhell
import pandas as pd
import pkg_resources
import psutil
import six
import wikitextparser as wtp
from lz4 import frame
from matplotlib.pyplot import table
from tqdm import tqdm

logger = logging.getLogger(__name__)

STYLE_RE = re.compile("'''*")
DISAMBI_REGEXP = re.compile(r"{{\s*(disambiguation|disambig|disamb|dab|geodis)\s*(\||})", re.IGNORECASE)

DEFAULT_IGNORED_NS = (
    "wikipedia:",
    "file:",
    "portal:",
    "template:",
    "mediawiki:",
    "user:",
    "help:",
    "book:",
    "draft:",
    "module:",
    "timedtext:",
)
IGNORED_LINKS = [
    "File:",
]
NAMESPACE_RE = re.compile(r"^{(.*?)}")
CLEANR = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")


def remove_html_tags(raw_html):
    cleantext = re.sub(CLEANR, " ", raw_html)
    return cleantext


@dataclass
class WikiLink:
    title: str
    text: str
    start: int
    end: int

    @property
    def span(self):
        return (self.start, self.end)

    def __repr__(self):
        text = "<WikiLink %s->%s>" % (self.text, self.title)
        if six.PY2:
            return (text).encode("utf-8")
        else:
            return text

    def __reduce__(self):
        return (self.__class__, (self.title, self.text, self.start, self.end))


@dataclass
class Paragraph:
    text: str
    wiki_links: List[WikiLink]
    abstract: bool

    def __repr__(self):
        text = "<Paragraph %s>" % (self.text[:50] + "...")
        if six.PY2:
            return (text).encode("utf-8")
        else:
            return text

    def __reduce__(self):
        return (self.__class__, (self.text, self.wiki_links, self.abstract))


@dataclass
class TableCell:
    # text: str
    # wiki_links: List[WikiLink]
    paragraphs: List[Paragraph]
    is_header: bool

    def __repr__(self):
        total_links = 0
        text = ""
        for paragraph in self.paragraphs:
            total_links += len(paragraph.wiki_links)
            if paragraph.text:
                text += paragraph.text

        if total_links:
            text += f" ({total_links} links)"

        if six.PY2:
            return (text).encode("utf-8")
        else:
            return text

    @property
    def text(self):
        return " ".join([paragraph.text for paragraph in self.paragraphs])

    def __reduce__(self):
        return (self.__class__, (self.paragraphs, self.is_header))


@dataclass
class WikiTable:
    title: str
    section: List[str]
    table_index: int
    cells: List[TableCell]
    headers_rows: List[int]
    headers_cells: List[Tuple[int, ...]]
    caption: List[Paragraph]
    n_row: int
    n_col: int
    out_text: List[Paragraph]
    # out_text_index: int

    def __repr__(self):
        text = [f"{self.title}.{self.table_index}"]
        if self.caption:
            caption_text = " ".join([i.text for i in self.caption])
            text.append(f"C:{caption_text}")
        if self.section:
            text.append("S:" + "/".join(self.section))

        text = " | ".join(text)[:50]

        if six.PY2:
            return (text).encode("utf-8")
        else:
            return text

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.title,
                self.section,
                self.table_index,
                self.cells,
                self.headers_rows,
                self.headers_cells,
                self.caption,
                self.n_row,
                self.n_col,
                self.out_text,
            ),
        )

    def to_pandas(self):
        table = []
        for rows in self.cells:
            table.append([col.text for col in rows])
        return pd.DataFrame(table)

    def get_outside_text(self):
        return self.out_text.text

    def to_text(self):
        table_text = ""
        if self.caption:
            caption_text = " ".join([i.text for i in self.caption])
            table_text += f"\nCaption: {caption_text}\n"
        table_text += self.to_pandas().to_string() + "\n"
        if self.out_text:
            out_text = " ".join([i.text for i in self.out_text])
            table_text = out_text + "\n" + table_text

        if self.section:
            table_text = "\n".join(self.section) + table_text
        return f"{self.title}\n{table_text}"


@dataclass
class WikiPage:
    title: str
    language: str
    wiki_text: str
    redirect: str

    def __repr__(self):
        if six.PY2:
            return b"<WikiPage %s>" % self.title.encode("utf-8")
        else:
            return "<WikiPage %s>" % self.title

    def __reduce__(self):
        return (self.__class__, (self.title, self.language, self.wiki_text, self.redirect))

    @property
    def is_redirect(self):
        return bool(self.redirect)

    @property
    def is_disambiguation(self):
        return bool(DISAMBI_REGEXP.search(self.wiki_text))


class WikiDumpReader(object):
    def __init__(self, dump_file, ignored_ns=DEFAULT_IGNORED_NS):
        self._dump_file = dump_file
        self._ignored_ns = ignored_ns

        with bz2.BZ2File(self._dump_file) as f:
            self._language = re.search(r'xml:lang="(.*)"', six.text_type(f.readline())).group(1)

    @property
    def dump_file(self):
        return self._dump_file

    @property
    def language(self):
        return self._language

    def __iter__(self):
        with bz2.BZ2File(self._dump_file) as f:
            c = 0
            for (title, wiki_text, redirect) in _extract_pages(f):
                lower_title = title.lower()
                if any([lower_title.startswith(ns) for ns in self._ignored_ns]):
                    continue
                c += 1

                yield WikiPage(title, self._language, wiki_text, redirect)

                if c % 100000 == 0:
                    logger.info("Processed: %d pages", c)


# obtained from https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/corpora/wikicorpus.py
def _extract_pages(in_file):
    elems = (elem for (_, elem) in iterparse(in_file, events=(b"end",)))
    elem = next(elems)

    tag = six.text_type(elem.tag)
    namespace = _get_namespace(tag)
    page_tag = "{%s}page" % namespace
    text_path = "./{%s}revision/{%s}text" % (namespace, namespace)
    title_path = "./{%s}title" % namespace
    redirect_path = "./{%s}redirect" % namespace

    for elem in elems:
        if elem.tag == page_tag:
            title = elem.find(title_path).text
            text = elem.find(text_path).text or ""
            redirect = elem.find(redirect_path)
            if redirect is not None:
                redirect = _normalize_title(_to_unicode(redirect.attrib["title"]))

            yield _to_unicode(title), _to_unicode(text), redirect

            elem.clear()


# obtained from https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/corpora/wikicorpus.py
def _get_namespace(tag):
    match_obj = NAMESPACE_RE.match(tag)
    if match_obj:
        namespace = match_obj.group(1)
        if not namespace.startswith("http://www.mediawiki.org/xml/export-"):
            raise ValueError("%s not recognized as MediaWiki dump namespace" % namespace)
        return namespace
    else:
        return ""


def _to_unicode(s):
    if isinstance(s, str):
        return s
    return s.decode("utf-8")


def _normalize_title(title: str):
    return (title[0].upper() + title[1:]).replace("_", " ")


class WikipediaDumpDB:
    def __init__(self, db_file: str):
        self._db_file = db_file
        self._env = lmdb.open(
            self._db_file,
            readonly=True,
            subdir=False,
            lock=False,
            max_dbs=4,
        )
        self._meta_db = self._env.open_db(b"__meta__")
        self._page_db = self._env.open_db(b"__page__")
        self._table_db = self._env.open_db(b"__table__")
        self._redirect_db = self._env.open_db(b"__redirect__")

    def __reduce__(self):
        return (self.__class__, (self._db_file,))

    @property
    def uuid(self):
        with self._env.begin(db=self._meta_db) as txn:
            return txn.get(b"id").decode("utf-8")

    @property
    def dump_file(self):
        with self._env.begin(db=self._meta_db) as txn:
            return txn.get(b"dump_file").decode("utf-8")

    @property
    def language(self):
        with self._env.begin(db=self._meta_db) as txn:
            return txn.get(b"language").decode("utf-8")

    def page_size(self):
        with self._env.begin(db=self._page_db) as txn:
            return txn.stat()["entries"]

    def page_size_table(self):
        with self._env.begin(db=self._table_db) as txn:
            return txn.stat()["entries"]

    def titles(self):
        with self._env.begin(db=self._page_db) as txn:
            cur = txn.cursor()
            for key in cur.iternext(values=False):
                yield key.decode("utf-8")

    def redirects(self):
        with self._env.begin(db=self._redirect_db) as txn:
            cur = txn.cursor()
            for (key, value) in iter(cur):
                yield (key.decode("utf-8"), value.decode("utf-8"))

    def resolve_redirect(self, title: str):
        with self._env.begin(db=self._redirect_db) as txn:
            value = txn.get(title.encode("utf-8"))
            if value:
                return value.decode("utf-8")
            else:
                return title

    def is_redirect(self, title: str):
        with self._env.begin(db=self._redirect_db) as txn:
            value = txn.get(title.encode("utf-8"))

        return bool(value)

    def is_wikipedia_page(self, title):
        with self._env.begin(db=self._page_db) as txn:
            value = txn.get(title.encode("utf-8"))
            if value:
                return True
        return False

    def is_disambiguation(self, title: str):
        with self._env.begin(db=self._page_db) as txn:
            value = txn.get(title.encode("utf-8"))

        if not value:
            return False

        return pickle.loads(frame.decompress(value))[1]

    def get_paragraphs(self, key: str):
        with self._env.begin(db=self._page_db) as txn:
            value = txn.get(key.encode("utf-8"))
            if not value:
                raise KeyError(key)

        return self._deserialize_paragraphs(value)

    def _deserialize_paragraphs(self, value: bytes):
        ret = []
        for obj in pickle.loads(frame.decompress(value))[0]:
            wiki_links = [WikiLink(*args) for args in obj[1]]
            ret.append(Paragraph(obj[0], wiki_links, obj[2]))
        return ret

    def get_tables(self, key: str):
        with self._env.begin(db=self._table_db) as txn:
            value = txn.get(key.encode("utf-8"))
            if not value:
                raise KeyError(key)

        return self._deserialize_tables(value)

    def _deserialize_tables(self, value: bytes):
        ret = pickle.loads(frame.decompress(value))
        return ret

    @staticmethod
    def build(
        dump_reader,
        out_file,
        pool_size,
        chunk_size,
        preprocess_func=None,
        init_map_size=1073741824 * 20,
        buffer_size=1073741824,
    ):
        with closing(lmdb.open(out_file, subdir=False, map_async=True, map_size=init_map_size, max_dbs=4)) as env:
            map_size = [init_map_size]
            meta_db = env.open_db(b"__meta__")
            with env.begin(db=meta_db, write=True) as txn:
                txn.put(b"id", six.text_type(uuid1().hex).encode("utf-8"))
                txn.put(b"dump_file", dump_reader.dump_file.encode("utf-8"))
                txn.put(b"language", dump_reader.language.encode("utf-8"))
                txn.put(
                    b"version", six.text_type(pkg_resources.get_distribution("wikipedia2vec").version).encode("utf-8")
                )

            page_db = env.open_db(b"__page__")
            table_db = env.open_db(b"__table__")
            redirect_db = env.open_db(b"__redirect__")

            def write_db(db, data):
                # sort make it write faster
                data = sorted(data, key=lambda x: x[1])

                try:
                    with env.begin(db=db, write=True) as txn:
                        txn.cursor().putmulti(data)

                except lmdb.MapFullError:
                    map_size[0] *= 2
                    env.set_mapsize(map_size[0])
                    write_db(db, data)

            with closing(Pool(pool_size)) as pool:
                page_buf = []
                table_buf = []
                redirect_buf = []
                buff_size = 0
                f = partial(_parse, preprocess_func=preprocess_func)
                for ret in tqdm(pool.imap_unordered(f, dump_reader, chunksize=chunk_size), mininterval=5):
                    # if pool_size == 1:

                    # for title in tqdm(dump_reader, mininterval=5):
                    ret = _parse(title, preprocess_func)
                    if ret:
                        if ret[0] == "page":
                            buff_size += len(ret[1][0]) + len(ret[1][1]) + len(ret[1][2])
                            page_buf.append((ret[1][0], ret[1][1]))
                            table_buf.append((ret[1][0], ret[1][2]))

                        else:  # redirect
                            buff_size += len(ret[1][0]) + len(ret[1][1])
                            redirect_buf.append(ret[1])

                    if buff_size > buffer_size:
                        print("Saved")
                        write_db(page_db, page_buf)
                        page_buf = []
                        write_db(redirect_db, redirect_buf)
                        redirect_buf = []
                        write_db(table_db, table_buf)
                        table_buf = []
                        buff_size = 0
                    # if buff_size > 1000000:
                    # break
                    # if ret[1][0].decode("utf-8") == "1946 European Athletics Championships – Men's high jump":
                    #     break
                    # if len(page_buf) > 100000:
                    #     break

                if page_buf:
                    write_db(page_db, page_buf)

                if redirect_buf:
                    write_db(redirect_db, redirect_buf)

                if table_buf:
                    write_db(table_db, table_buf)


def _parse_text_with_wikilinks(wikitext: str):
    wikitext = remove_html_tags(wikitext)
    wikitext = wtp.parse(wikitext).plain_text(replace_wikilinks=False).strip()

    link_parser = wtp.parse(wikitext)
    wiki_links = []
    wikitext = link_parser.string
    if link_parser.wikilinks:
        for link_obj in link_parser.wikilinks:
            if not link_obj.title:
                continue
            start, end = link_obj.span
            text = link_obj.text if link_obj.text else link_obj.title
            title = _normalize_title(link_obj.title)
            wikilink_obj = WikiLink(title=title, text=text, start=start, end=end)
            if wikilink_obj:
                wiki_links.append(wikilink_obj)

        wiki_links.sort(key=lambda x: x.start)
        i, j = 0, 0
        wikitext = ""
        while i < len(link_parser.string) and j < len(wiki_links):
            wikitext += link_parser.string[i : wiki_links[j].start] + wiki_links[j].text

            i = wiki_links[j].end
            j += 1
        wikitext += link_parser.string[i:]
        wikitext = " ".join(wikitext.split(" ")).strip()

        # Remove Files link
        wiki_links = [l for l in wiki_links if l.title and not any([l.title.startswith(ns) for ns in IGNORED_LINKS])]

        last_position = 0
        for link in wiki_links:
            start = wikitext.find(link.text, last_position, len(wikitext))
            if start >= last_position:
                link.start = start
                link.end = start + len(link.text)
                last_position = link.end
    return wikitext, wiki_links


def _parse_tables(wikipedia_title, wiki_text: str):
    try:
        parsed_page_obj = wtp.parse(wiki_text)
        if not parsed_page_obj.tables:
            return []
    except Exception:
        logger.warn(f"Failed to parse wikitables: {wikipedia_title}")
        return []

    tables = []
    section_hierarchy = defaultdict(list)
    section_text_wo_table = {}
    # section_index_table = {}
    for section in parsed_page_obj.sections:
        if not section.tables:
            continue
        # Remove other subsections text
        # section
        # | text     : keep
        # | table    : remove
        # | sub-section  : remove
        section_text = wtp.parse(section.contents).plain_text(replace_wikilinks=False).strip()

        # remove other sub-section
        for subsection in section.sections:
            if not subsection.contents or subsection.span == section.span:
                continue
            subsection_text = wtp.parse(subsection.contents).plain_text(replace_wikilinks=False).strip()
            try:
                subsection_start = section_text.index(subsection_text)
                section_text = (
                    section_text[0:subsection_start] + section_text[subsection_start + len(subsection_text) :]
                ).strip()
            except ValueError:
                pass

        for table in section.tables:
            if section.title:
                section_title = section.title.strip()
                section_hierarchy[table.span].append(section_title)

            # remove table text
            table_text = wtp.parse(table.string).plain_text(replace_wikilinks=False).strip()
            try:
                table_start = section_text.index(table_text)
                section_text = (section_text[0:table_start] + section_text[table_start + len(table_text) :]).strip()
                # section_index_table[table.span] = table_start
            except ValueError:
                pass

        if section_text:
            # remove extra spaces, enter, tab
            # section_text, section_links = _parse_text_with_wikilinks(section_text)

            for table in section.tables:
                section_paragraphs = _parse_wikitext_with_parser_from_hell(section_text)
                section_paragraphs = [
                    [args[0], [WikiLink(*i) for i in args[1]], args[2]] for args in section_paragraphs
                ]
                section_paragraphs = [Paragraph(*args) for args in section_paragraphs]
                section_text_wo_table[table.span] = section_paragraphs
                # section_text_wo_table[table.span] = Paragraph(
                # text=section_text, wiki_links=section_links, abstract=False
                # )

    for table_index, parsed_table in enumerate(parsed_page_obj.tables):
        if parsed_table.tables:
            continue
        try:
            table_section = section_hierarchy.get(parsed_table.span)

            table_out_text = section_text_wo_table.get(parsed_table.span)
            # table_index = section_index_table.get(parsed_table.span)

            cells, headers_rows, header_cells = [], [], []
            n_col, n_row = 0, 0
            caption = None

            if parsed_table.caption:
                caption_paragraphs = _parse_wikitext_with_parser_from_hell(parsed_table.caption.strip())
                caption_paragraphs = [
                    [args[0], [WikiLink(*i) for i in args[1]], args[2]] for args in caption_paragraphs
                ]
                caption_paragraphs = [Paragraph(*args) for args in caption_paragraphs]
                caption = caption_paragraphs

                # caption = _parse_wikitext_with_parser_from_hell(parsed_table.caption)
                # caption_text, caption_links = _parse_text_with_wikilinks(parsed_table.caption)
                # caption = Paragraph(text=caption_text, wiki_links=caption_links, abstract=False)

            for r, row_obj in enumerate(parsed_table.cells()):
                results_row = []
                n_col = max(len(row_obj), n_col)

                c_header_cell = 0
                is_null_row = True
                for c, cell in enumerate(row_obj):
                    cell_text = ""
                    is_header = False
                    if cell:
                        cell_text = cell.value
                        is_header = cell.is_header
                    if is_header:
                        header_cells.append([r, c])
                        c_header_cell += 1

                    # cell_paragraphs = _parse_wikitext_with_parser_from_hell(cell_text)

                    cell_paragraphs = _parse_wikitext_with_parser_from_hell(cell_text.strip())
                    cell_paragraphs = [[args[0], [WikiLink(*i) for i in args[1]], args[2]] for args in cell_paragraphs]
                    cell_paragraphs = [Paragraph(*args) for args in cell_paragraphs]

                    # cell_text, cell_links = _parse_text_with_wikilinks(cell_text)
                    if cell_paragraphs:
                        is_null_row = False
                    results_row.append(TableCell(cell_paragraphs, is_header))
                if is_null_row:
                    continue
                if c_header_cell >= len(row_obj) // 2:
                    headers_rows.append(r)

                cells.append(results_row)
                n_row += 1

            tables.append(
                WikiTable(
                    title=wikipedia_title,
                    section=table_section,
                    table_index=table_index,
                    cells=cells,
                    headers_rows=headers_rows,
                    headers_cells=header_cells,
                    caption=caption,
                    n_row=n_row,
                    n_col=n_col,
                    out_text=table_out_text,
                    # out_text_index=table_index,
                )
            )
        except Exception:
            logger.warn(
                f"Failed to parse wikitables: {wikipedia_title}- index {table_index}",
            )
            with open(f"log/{wikipedia_title}.{table_index}.txt", "w") as f:
                f.write(parsed_table.string)

    return tables


def _parse_wikitext_with_parser_from_hell(wiki_text, preprocess_func=None, title=None):
    # remove style tags to reduce parsing errors
    wiki_text = STYLE_RE.sub("", wiki_text)
    try:
        parsed = mwparserfromhell.parse(wiki_text)
    except Exception:
        logger.warn("Failed to parse wiki text: %s", title)
        return None

    paragraphs = []
    cur_text = ""
    cur_links = []
    abstract = True

    if preprocess_func is None:
        preprocess_func = lambda x: x

    for node in parsed.nodes:
        if isinstance(node, mwparserfromhell.nodes.Text):
            for (n, text) in enumerate(six.text_type(node).split("\n")):
                if n == 0:
                    cur_text += preprocess_func(text)
                else:
                    if cur_text and not cur_text.isspace():
                        paragraphs.append([cur_text, cur_links, abstract])

                    cur_text = preprocess_func(text)
                    cur_links = []

        elif isinstance(node, mwparserfromhell.nodes.Wikilink):
            title = node.title.strip_code().strip(" ")
            if title.startswith(":"):
                title = title[1:]
            if not title:
                continue
            title = _normalize_title(title)
            # (title[0].upper() + title[1:]).replace("_", " ")

            if node.text:
                text = node.text.strip_code()
                # dealing with extended image syntax: https://en.wikipedia.org/wiki/Wikipedia:Extended_image_syntax
                if title.lower().startswith("file:") or title.lower().startswith("image:"):
                    text = text.split("|")[-1]
            else:
                text = node.title.strip_code()

            text = preprocess_func(text)
            start = len(cur_text)
            cur_text += text
            end = len(cur_text)
            cur_links.append((title, text, start, end))

        elif isinstance(node, mwparserfromhell.nodes.ExternalLink):
            if not node.title:
                continue

            text = node.title.strip_code()
            cur_text += preprocess_func(text)

        elif isinstance(node, mwparserfromhell.nodes.Tag):
            if node.tag not in ("b", "i", "u"):
                continue
            if not node.contents:
                continue

            text = node.contents.strip_code()
            cur_text += preprocess_func(text)

        elif isinstance(node, mwparserfromhell.nodes.Heading):
            abstract = False

    if cur_text and not cur_text.isspace():
        paragraphs.append([cur_text, cur_links, abstract])
    return paragraphs


def _parse(page: WikiPage, preprocess_func):
    if page.is_redirect:
        return ("redirect", (page.title.encode("utf-8"), page.redirect.encode("utf-8")))

    paragraphs = _parse_wikitext_with_parser_from_hell(page.wiki_text, preprocess_func, page.title)

    ret = [paragraphs, page.is_disambiguation]

    encoded_title = page.title.encode("utf-8")
    encoded_paragraphs = frame.compress(pickle.dumps(ret, protocol=-1))

    # Parse table content using wikitextparser
    tables = _parse_tables(page.title, page.wiki_text)

    # if tables:
    # print(tables[0].to_text())

    encoded_tables = frame.compress(pickle.dumps(tables, protocol=-1))
    return ("page", (encoded_title, encoded_paragraphs, encoded_tables))


def _initialize_worker(dump_db):
    global _dump_db
    _dump_db = dump_db


def count_paragraph_entity_vocab(title):
    counter = Counter()
    try:
        paragraphs = _dump_db.get_paragraphs(title)
    except Exception as message:
        print(f"Count not parse: {title}")
        print(message)
        return title, counter, 0
    for paragraph in paragraphs:
        for wiki_link in paragraph.wiki_links:
            title = _dump_db.resolve_redirect(wiki_link.title)
            if title.startswith("Category:") or not _dump_db.is_wikipedia_page(title):
                continue
            counter[title] += 1
    return title, counter, len(paragraphs)


def count_table_entity_vocab(title):
    counter = Counter()
    try:
        tables = _dump_db.get_tables(title)
    except Exception as message:
        print(f"Count not parse: {title}")
        print(message)
        return title, counter, 0
    for table in tables:
        if table.caption:
            for wiki_link in table.caption.wiki_links:
                title = _dump_db.resolve_redirect(wiki_link.title)
                if title.startswith("Category:") or not _dump_db.is_wikipedia_page(title):
                    continue
                counter[title] += 1

        for row in table.cells:
            for cell in row:
                for wiki_link in cell.wiki_links:
                    title = _dump_db.resolve_redirect(wiki_link.title)
                    if title.startswith("Category:") or not _dump_db.is_wikipedia_page(title):
                        continue
                    counter[title] += 1
    return title, counter, len(tables)


# @click.command()
# @click.argument("dump_db_file", type=click.Path())
# @click.option("--pool-size", default=multiprocessing.cpu_count())
# @click.option("--chunk-size", type=int, default=100)
def show_dump_db_stats(dump_db_file: str, pool_size: int, chunk_size: int):
    dump_db = WikipediaDumpDB(dump_db_file)
    target_titles = [
        title
        for title in dump_db.titles()
        if not (":" in title and title.lower().split(":")[0] in ("image", "file", "category"))
    ]

    def run_counter(callback_func):
        n_mention, n_obj = 0, 0
        counter = Counter()
        articles = set()
        p_bar = tqdm(total=len(target_titles), mininterval=1)
        with closing(Pool(pool_size, initializer=_initialize_worker, initargs=(dump_db,))) as pool:
            for title, ret, n_obj_i in pool.imap_unordered(callback_func, target_titles, chunksize=chunk_size):
                counter.update(ret)
                n_mention += sum(ret.values())
                n_obj += n_obj_i
                p_bar.update()
                if n_obj_i:
                    articles.add(title)
                # if len(counter) > 1000:
                # break
        p_bar.close()
        return articles, counter, n_mention, n_obj

    paragraph_articles, paragraph_counter, paragraph_mention, n_paragraph = run_counter(count_paragraph_entity_vocab)
    table_articles, table_counter, table_mention, n_table = run_counter(count_table_entity_vocab)

    all_counter = Counter()
    all_counter.update(paragraph_counter)
    all_counter.update(table_counter)
    all_mention = table_mention + paragraph_mention
    all_articles = set()
    all_articles.update(paragraph_articles)
    all_articles.update(table_articles)

    print(f"   Obj    \tPages\t#Obj\tMention\tEntity")
    print(
        f"Paragraphs\t{len(paragraph_articles):,}\t{n_paragraph:,}\t{paragraph_mention:,}\t{len(paragraph_counter):,}"
    )
    print(f"Tables    \t{len(table_articles):,}\t{n_table:,}\t{table_mention:,}\t{len(table_counter):,}")
    print(f"All       \t{len(all_articles):,}\t  -  \t{all_mention:,}\t{len(all_counter):,}")

    # def save_vocab(out_file, vocab):
    #     with open(out_file, "w") as f:
    #         writer = csv.writer(f, delimiter="\t")
    #         for title, title_fre in vocab.most_common():
    #             writer.writerow([title, title_fre])

    # save_vocab("data/entity_vocab_20181220_paragraph.tsv", paragraph_counter)
    # save_vocab("data/entity_vocab_20181220_table.tsv", table_counter)
    # save_vocab("data/entity_vocab_20181220_all.tsv", all_counter)


@click.command()
@click.argument("dump_file", type=click.Path(exists=True))
@click.argument("out_file", type=click.Path())
@click.option("--pool-size", default=multiprocessing.cpu_count())
@click.option("--chunk-size", type=int, default=100)
@click.option("--memory", type=int, default=20)  # 25% of ram
def build_table_dump_db(dump_file: str, out_file: str, memory: int, **kwargs):
    memory = psutil.virtual_memory().total * memory // 100
    dump_reader = WikiDumpReader(dump_file)
    WikipediaDumpDB.build(dump_reader, out_file, buffer_size=memory, **kwargs)
