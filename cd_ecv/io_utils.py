import json
import logging
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Tuple

log = logging.getLogger("cd_ecv.io_utils")


def download_url(url: str, save_path: Path, retries: int = 3) -> Path:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists():
        log.info(f"File already exists: {save_path}")
        return save_path

    last_error = None
    for attempt in range(retries):
        try:
            log.info(f"Downloading {url} (attempt {attempt + 1}/{retries})")
            urllib.request.urlretrieve(url, save_path)
            log.info(f"Download complete: {save_path}")
            return save_path
        except Exception as e:
            last_error = e
            log.warning(f"Attempt {attempt + 1} failed: {e}")

    raise RuntimeError(f"Failed to download {url}") from last_error


def safe_extract_tar_gz(tar_path: Path, out_dir: Path) -> None:
    log.info(f"Extracting {tar_path} -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(out_dir)

    log.info("Extraction complete")


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning(f"Skipping invalid JSON on line {line_num} in {path}: {e}")

    return records


def find_required_files(base_dir: Path) -> Tuple[Path, Path]:
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    corpus_jsonl = next(base_dir.rglob("corpus.jsonl"), None)
    claims_jsonl = next(base_dir.rglob("claims_dev.jsonl"), None)

    if corpus_jsonl is None:
        raise FileNotFoundError(f"corpus.jsonl not found inside {base_dir}")

    if claims_jsonl is None:
        raise FileNotFoundError(f"claims_dev.jsonl not found inside {base_dir}")

    return corpus_jsonl, claims_jsonl
