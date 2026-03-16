import json
import logging
from pathlib import Path

from cd_ecv.config import CDConfig
from cd_ecv.data import build_corpus, load_claims
from cd_ecv.io_utils import find_required_files
from cd_ecv.metrics import NumpyEncoder
from cd_ecv.models import load_models
from cd_ecv.pipeline import run_all_baselines

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cd_ecv")


def print_summary_table(outputs):
    print("\n" + "=" * 60)
    print("SUMMARY: Baseline Comparison (300 SciFact dev claims)")
    print("=" * 60)
    print(f"  {'System':<30} {'LabelAcc':>9} {'Abst':>7} {'EP':>7} {'ER':>7} {'EF1':>7} {'Macro':>7}")
    print("  " + "-" * 78)
    for item in outputs.values():
        result = item["metrics"].compute()
        print(
            f"  {item['name']:<30} {result['Label Accuracy']:>9.4f} {result['Abstention Rate']:>7.4f} "
            f"{result['Evidence Precision']:>7.4f} {result['Evidence Recall']:>7.4f} "
            f"{result['Evidence F1']:>7.4f} {result['Macro Accuracy']:>7.4f}"
        )


def save_outputs(outputs, output_dir: str | Path = "."):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for tag, item in outputs.items():
        with open(output_dir / f"results_{tag}.json", "w", encoding="utf-8") as f:
            json.dump(item["results"], f, indent=2, cls=NumpyEncoder)

        with open(output_dir / f"metrics_{tag}.json", "w", encoding="utf-8") as f:
            json.dump(item["metrics"].compute(), f, indent=2, cls=NumpyEncoder)

    print("\nSaved: results_bl1/2/3.json, metrics_bl1/2/3.json")


def main():
    cfg = CDConfig()
    print("Using sci_dir:", cfg.sci_dir)

    corpus_jsonl, claims_jsonl = find_required_files(cfg.sci_dir)

    corpus_texts, corpus_srcs, corpus_pmids, bm25 = build_corpus(cfg.sci_dir)
    log.info("Corpus: %s docs", len(corpus_texts))

    embedder, cross_encoder, nli_model = load_models()
    claims_data = load_claims(cfg.sci_dir)
    log.info("Claims: %s", len(claims_data))

    outputs = run_all_baselines(
        claims_data,
        corpus_texts,
        corpus_srcs,
        corpus_pmids,
        bm25,
        embedder,
        cross_encoder,
        nli_model,
        cfg,
    )

    print_summary_table(outputs)
    save_outputs(outputs, cfg.output_dir)


if __name__ == "__main__":
    main()
