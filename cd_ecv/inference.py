import numpy as np

from .config import CDConfig
from .utils import _softmax, lexical_overlap, sent_overlap


def construct_answer(query, reranked, corpus_srcs, corpus_pmids, cfg: CDConfig):
    spans, labels, pmids, seen = [], [], [], set()
    for item in reranked:
        if len(item) == 4:
            sent, idx, score, src = item
        elif len(item) == 3:
            sent, idx, score = item
            src = corpus_srcs[idx] if idx < len(corpus_srcs) else 'Unknown'
        elif len(item) == 2:
            sent, idx = item
            score = 0.0
            src = corpus_srcs[idx] if idx < len(corpus_srcs) else 'Unknown'
        else:
            continue

        if sent_overlap(sent, query, cfg) and sent.lower() not in seen:
            spans.append((sent, idx, score))
            labels.append(src)
            if idx < len(corpus_pmids):
                pmids.append(corpus_pmids[idx])
            seen.add(sent.lower())
        if len(spans) >= cfg.top_evidence:
            break

    return ' '.join(s for s, _, _ in spans), spans, labels, pmids


def predict_label(claim, spans, nli_model, cfg: CDConfig, n_filtered_sents=999, use_nli=True):
    if not use_nli:
        if nli_model is None or not spans:
            return 'NOT_ENOUGH_INFO'
        sorted_spans = sorted(spans, key=lambda x: -x[2]) if len(spans[0]) == 3 else spans
        sentences = [s for s, *_ in sorted_spans[:1]]
        if not sentences:
            return 'NOT_ENOUGH_INFO'
        try:
            raw = np.array(nli_model.predict([(claim, sentences[0])]))
            probs = _softmax(raw[0])
            winner = int(np.argmax(probs))
            if winner == cfg.idx_entail:
                return 'SUPPORT'
            if winner == cfg.idx_contra:
                return 'CONTRADICT'
            return 'NOT_ENOUGH_INFO'
        except Exception:
            return 'NOT_ENOUGH_INFO'

    if n_filtered_sents < cfg.min_filtered_sents:
        return 'NOT_ENOUGH_INFO'
    if nli_model is None or not spans:
        return 'NOT_ENOUGH_INFO'

    sorted_spans = sorted(spans, key=lambda x: -x[2]) if len(spans[0]) == 3 else spans
    if sorted_spans:
        top_sent = sorted_spans[0][0]
        if lexical_overlap(claim, top_sent, cfg) < cfg.nei_lex_thresh:
            return 'NOT_ENOUGH_INFO'

    sentences = [s for s, *_ in sorted_spans[:cfg.nli_top_ev]]
    if not sentences:
        return 'NOT_ENOUGH_INFO'

    try:
        raw = np.array(nli_model.predict([(claim, s) for s in sentences]))
        probs = np.array([_softmax(row) for row in raw])

        avg_c = float(probs[:, cfg.idx_contra].mean())
        avg_n = float(probs[:, cfg.idx_neutral].mean())
        avg_e = float(probs[:, cfg.idx_entail].mean())

        class_scores = {
            'CONTRADICT': avg_c,
            'NOT_ENOUGH_INFO': avg_n,
            'SUPPORT': avg_e,
        }
        pred_label = max(class_scores, key=class_scores.get)
        pred_conf = class_scores[pred_label]

        if pred_label != 'NOT_ENOUGH_INFO' and pred_conf < cfg.nli_conf_thresh:
            return 'NOT_ENOUGH_INFO'
        return pred_label
    except Exception:
        return 'NOT_ENOUGH_INFO'
