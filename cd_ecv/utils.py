import re
import numpy as np
from sentence_transformers import util

from .config import CDConfig

try:
    from scipy.special import softmax as _softmax
except ImportError:
    def _softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()


_WORD_RE = re.compile(r'[A-Za-z][A-Za-z\-]+')


def content_words(text: str, cfg: CDConfig) -> list[str]:
    return [
        w.lower()
        for w in _WORD_RE.findall(text.lower())
        if w.lower() not in cfg.stop_words and len(w) > 2
    ]


def sent_overlap(sent: str, query: str, cfg: CDConfig) -> bool:
    qset = set(content_words(query, cfg))
    return bool(set(_WORD_RE.findall(sent.lower())) & qset) if qset else True


def lexical_overlap(claim: str, evidence: str, cfg: CDConfig) -> float:
    qw = set(content_words(claim, cfg))
    ew = set(content_words(evidence, cfg))
    if not qw or not ew:
        return 0.0
    return len(qw & ew) / max(1.0, (len(qw) * len(ew)) ** 0.5)


def focus_score(query: str, sent: str, embedder, cfg: CDConfig) -> float:
    qw, sw = set(content_words(query, cfg)), set(content_words(sent, cfg))
    lex = len(qw & sw) / max(1.0, (len(qw) * len(sw)) ** 0.5) if qw and sw else 0.0
    qv = embedder.encode([query], convert_to_tensor=True, normalize_embeddings=True)
    sv = embedder.encode([sent], convert_to_tensor=True, normalize_embeddings=True)
    sem = (float(util.cos_sim(qv, sv)[0][0]) + 1.0) / 2.0
    return 0.55 * sem + 0.45 * lex


def batch_focus_scores(query: str, sentences: list[str], embedder, cfg: CDConfig) -> np.ndarray:
    if not sentences:
        return np.array([])
    qw = set(content_words(query, cfg))
    lex = np.array([
        len(qw & set(content_words(s, cfg))) /
        max(1.0, (len(qw) * max(1, len(set(content_words(s, cfg))))) ** 0.5)
        if qw else 0.0
        for s in sentences
    ])
    qv = embedder.encode([query], convert_to_tensor=True, normalize_embeddings=True)
    sv = embedder.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
    sem = (util.cos_sim(qv, sv)[0].detach().cpu().numpy() + 1.0) / 2.0
    return 0.55 * sem + 0.45 * lex
