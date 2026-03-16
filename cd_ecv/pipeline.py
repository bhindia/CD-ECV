import logging

from .config import CDConfig
from .inference import construct_answer, predict_label
from .metrics import CDECVMetrics, gold_evidence_ids, gold_label_from_rec
from .retrieval import retrieve_and_rerank

log = logging.getLogger('cd_ecv')


def run_pipeline(
    claims_data,
    corpus_texts,
    corpus_srcs,
    corpus_pmids,
    bm25,
    embedder,
    cross_encoder,
    nli_model,
    cfg: CDConfig,
    *,
    use_reranker=True,
    use_focus_filter=True,
    use_nli=True,
    use_dense_prefilter=True,
    label='CD-ECV Full',
):
    metrics = CDECVMetrics(cfg)
    results_out, fallback_count = [], 0

    for i, claim_rec in enumerate(claims_data):
        claim_text = claim_rec['claim']
        reranked, diag = retrieve_and_rerank(
            claim_text,
            corpus_texts,
            corpus_srcs,
            bm25,
            embedder,
            cross_encoder,
            cfg,
            use_focus_filter=use_focus_filter,
            use_reranker=use_reranker,
            use_dense_prefilter=use_dense_prefilter,
        )
        fallback_used = diag.get('fallback_used', False)
        n_filtered = diag.get('filtered_sentences', 0)
        if fallback_used:
            fallback_count += 1

        answer, spans, cite_labels, pmids = construct_answer(
            claim_text,
            reranked,
            corpus_srcs,
            corpus_pmids,
            cfg,
        )
        pred_label = predict_label(
            claim_text,
            spans,
            nli_model,
            cfg,
            n_filtered_sents=0 if fallback_used else n_filtered,
            use_nli=use_nli,
        )

        gold_ids = gold_evidence_ids(claim_rec)
        gold_label = gold_label_from_rec(claim_rec)
        reported_pmids = [] if (fallback_used or pred_label == 'NOT_ENOUGH_INFO') else list(dict.fromkeys(pmids))[:cfg.max_evidence]

        metrics.update(pred_label, gold_label, set(reported_pmids), gold_ids)
        results_out.append({
            'claim_id': claim_rec.get('id'),
            'claim': claim_text,
            'predicted_label': pred_label,
            'gold_label': gold_label,
            'answer': answer,
            'predicted_pmids': reported_pmids,
            'gold_pmids': list(gold_ids),
            'diagnostics': {**diag, 'n_filtered_sents': n_filtered},
        })

        if (i + 1) % 25 == 0:
            log.info('  [%s] %s/%s', label, i + 1, len(claims_data))

    log.info('[%s] Fallback: %s/%s', label, fallback_count, len(claims_data))
    metrics.print_summary(label=label)
    return metrics, results_out


def run_all_baselines(claims_data, corpus_texts, corpus_srcs, corpus_pmids, bm25, embedder, cross_encoder, nli_model, cfg: CDConfig):
    systems = [
        {
            'name': 'BL1 (BM25 only)',
            'kwargs': dict(use_reranker=False, use_focus_filter=False, use_nli=False, use_dense_prefilter=False),
            'tag': 'bl1',
        },
        {
            'name': 'BL2 (BM25 + Reranker)',
            'kwargs': dict(use_reranker=True, use_focus_filter=False, use_nli=False, use_dense_prefilter=False),
            'tag': 'bl2',
        },
        {
            'name': 'BL3 / CD-ECV v9',
            'kwargs': dict(use_reranker=True, use_focus_filter=True, use_nli=True, use_dense_prefilter=True),
            'tag': 'bl3',
        },
    ]

    outputs = {}
    for system in systems:
        print('\n' + '=' * 60)
        print(f"Running {system['name']}")
        print('=' * 60)
        metrics, results = run_pipeline(
            claims_data,
            corpus_texts,
            corpus_srcs,
            corpus_pmids,
            bm25,
            embedder,
            cross_encoder,
            nli_model,
            cfg,
            label=system['name'],
            **system['kwargs'],
        )
        outputs[system['tag']] = {'metrics': metrics, 'results': results, 'name': system['name']}

    return outputs
