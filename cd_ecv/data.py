
from pathlib import Path
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from .io_utils import find_required_files, read_jsonl


def load_scifact_dataset(base_dir: Path) -> Tuple[List[dict], List[dict]]:
    corpus_path, claims_path = find_required_files(base_dir)
    corpus = read_jsonl(corpus_path)
    claims = read_jsonl(claims_path)
    return corpus, claims


def build_corpus(base_dir: Path):
    corpus_records, _ = load_scifact_dataset(base_dir)

    corpus_texts = []
    corpus_srcs = []
    corpus_pmids = []

    for record in corpus_records:
        title = record.get("title", "")
        abstract = record.get("abstract", [])

        if isinstance(abstract, list):
            abstract_text = " ".join(str(x) for x in abstract)
        else:
            abstract_text = str(abstract)

        full_text = f"{title} {abstract_text}".strip()

        corpus_texts.append(full_text)
        corpus_srcs.append(record)
        corpus_pmids.append(record["doc_id"])

    tokenized_corpus = [text.lower().split() for text in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    return corpus_texts, corpus_srcs, corpus_pmids, bm25


def load_claims(base_dir: Path) -> List[dict]:
    _, claims = load_scifact_dataset(base_dir)
    return claims
