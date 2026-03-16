import numpy as np
from nltk.tokenize import sent_tokenize
from scipy.special import expit as sigmoid
from sentence_transformers import util

from .config import CDConfig
from .utils import batch_focus_scores


def dense_rerank_docs(query, doc_indices, corpus_texts, embedder, cfg: CDConfig):
    proxies = [' '.join(corpus_texts[i].split()[:64]) for i in doc_indices]
    qv = embedder.encode([query], convert_to_tensor=True, normalize_embeddings=True)
    dv = embedder.encode(proxies, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(qv, dv)[0].detach().cpu().numpy()
    order = np.argsort(-sims)
    kept = [i for i in order if sims[i] >= cfg.doc_dense_thresh]
    if len(kept) < 5:
        kept = list(order[:5])
    kept = kept[:cfg.dense_top_docs]
    return [doc_indices[i] for i in kept], sims[np.array([order[i] for i in range(len(kept))])]


def retrieve_and_rerank(
    query,
    corpus_texts,
    corpus_srcs,
    bm25,
    embedder,
    cross_encoder,
    cfg: CDConfig,
    use_focus_filter=True,
    use_reranker=True,
    use_dense_prefilter=True,
):
    bm_scores = bm25.get_scores(query.split())
    top_idxs = list(np.argsort(-bm_scores)[:cfg.bm25_top_docs])

    doc_score_map = {}
    if use_dense_prefilter and use_reranker:
        top_idxs, dense_sims = dense_rerank_docs(query, top_idxs, corpus_texts, embedder, cfg)
        doc_score_map = {int(idx): float(sc) for idx, sc in zip(top_idxs, dense_sims)}

    candidates, seen = [], set()
    for doc_idx in top_idxs:
        for sent in sent_tokenize(corpus_texts[int(doc_idx)]):
            sent = sent.strip()
            if not sent or len(sent.split()) < cfg.min_sent_words or sent.lower() in seen:
                continue
            seen.add(sent.lower())
            candidates.append((sent, int(doc_idx), corpus_srcs[int(doc_idx)]))

    if not candidates:
        return [], {'filtered_sentences': 0, 'fallback_used': True, 'doc_scores': {}}

    if use_reranker and cross_encoder:
        if len(candidates) > cfg.sent_rerank_k:
            qe = embedder.encode([query], convert_to_tensor=True, normalize_embeddings=True)
            se = embedder.encode(
                [s for s, _, _ in candidates],
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            order = np.argsort(-util.cos_sim(qe, se)[0].detach().cpu().numpy())[:cfg.sent_rerank_k]
            candidates = [candidates[i] for i in order]
        try:
            raw_scores = cross_encoder.predict([(query, s) for s, _, _ in candidates], show_progress_bar=False)
        except Exception:
            qe = embedder.encode([query], convert_to_tensor=True, normalize_embeddings=True)
            se = embedder.encode([s for s, _, _ in candidates], convert_to_tensor=True, normalize_embeddings=True)
            raw_scores = util.cos_sim(qe, se)[0].detach().cpu().numpy()
    else:
        qe = embedder.encode([query], convert_to_tensor=True, normalize_embeddings=True)
        se = embedder.encode([s for s, _, _ in candidates], convert_to_tensor=True, normalize_embeddings=True)
        raw_scores = util.cos_sim(qe, se)[0].detach().cpu().numpy()

    scored = [(s, idx, float(sc), src) for (s, idx, src), sc in zip(candidates, raw_scores)]

    if use_focus_filter:
        sents_only = [s for s, _, _, _ in scored]
        focus_vals = batch_focus_scores(query, sents_only, embedder, cfg)
        filtered = [
            item for item, fv in zip(scored, focus_vals)
            if item[2] >= cfg.conf_thresh and fv >= cfg.min_focus
        ]
        fallback_used = not filtered
        if fallback_used:
            filtered = sorted(scored, key=lambda x: -x[2])[:cfg.fallback_top_n]
    else:
        filtered = sorted(scored, key=lambda x: -x[2])[:cfg.fallback_top_n]
        fallback_used = False

    doc_best = {}
    for item in sorted(filtered, key=lambda x: -x[2]):
        if item[1] not in doc_best:
            doc_best[item[1]] = item

    def joint_score(idx):
        sent_score = float(sigmoid(doc_best[idx][2])) if use_reranker else doc_best[idx][2]
        doc_score = doc_score_map.get(idx, 0.3)
        return float(np.sqrt(max(sent_score, 1e-9) * max(doc_score, 1e-9)))

    ranked = sorted(doc_best.keys(), key=joint_score, reverse=True)
    unique = [doc_best[i] for i in ranked]
    return unique, {
        'filtered_sentences': int(len(filtered)),
        'fallback_used': fallback_used,
        'doc_scores': {int(k): float(v) for k, v in doc_score_map.items()},
    }
