from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CDConfig:
    sci_dir: Path = field(
        default_factory=lambda: Path(
            "/content/drive/MyDrive/cd_ecv_modular_v2/data/scifact/data"
        )
    )
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))

    bm25_top_docs: int = 150
    dense_top_docs: int = 25
    dense_prefilter_k: int = 25
    doc_dense_thresh: float = 0.20
    sent_rerank_k: int = 200
    reranker_top_k: int = 200

    min_sent_words: int = 6
    min_focus: float = 0.20
    conf_thresh: float = 0.35
    min_filtered_sents: int = 1
    focus_sent_threshold: int = 1
    fallback_top_n: int = 5

    top_evidence: int = 5
    max_evidence: int = 1
    nli_top_ev: int = 3
    evidence_top_k: int = 5
    nli_conf_thresh: float = 0.45
    nli_batch_size: int = 16

    nei_lex_thresh: float = 0.06

    idx_contra: int = 0
    idx_neutral: int = 1
    idx_entail: int = 2

    label_support: str = "SUPPORT"
    label_contradict: str = "CONTRADICT"
    label_nei: str = "NOT_ENOUGH_INFO"
    all_labels: tuple = ("SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO")

    use_focus_filter: bool = True
    use_reranker: bool = True
    use_dense_prefilter: bool = True

    negation_vocab: set = field(
        default_factory=lambda: {
            "not", "no", "nor", "fail", "fails", "failed", "lack", "lacks", "lacking",
            "absent", "neither", "unable", "without", "inhibit", "inhibits", "inhibited",
            "reduce", "reduces", "reduced", "decrease", "decreases", "decreased",
            "prevent", "prevents", "prevented", "block", "blocks", "blocked",
            "suppress", "suppresses", "suppressed", "abolish", "abolishes", "abolished",
            "attenuate", "attenuates", "attenuated", "impair", "impairs", "impaired",
            "loss", "lose", "loses", "lost", "deficient", "deficiency", "null",
            "nullify", "reverse", "reversed", "contrary", "opposite", "contradict",
            "contradicts", "disagree", "inconsistent", "incompatible", "refute",
            "refutes", "refuted"
        }
    )

    stop_words: set = field(
        default_factory=lambda: {
            "a", "an", "the", "and", "or", "but", "if", "then", "else", "for", "to",
            "of", "in", "on", "at", "by", "with", "without", "as", "is", "are", "was",
            "were", "be", "been", "being", "this", "that", "these", "those", "it",
            "its", "from", "into", "over", "under", "than", "such", "so", "not", "no",
            "yes"
        }
    )

    random_seed: int = 42
