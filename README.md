# CD-ECV modular refactor (updated)

This update fixes the weakest part of the first refactor: file and dataset handling.

## What changed

- Added `cd_ecv/io_utils.py` for:
  - `download_url(...)`
  - `safe_extract_tar_gz(...)`
  - `read_jsonl(...)`
  - `find_required_files(...)`
- Moved file discovery out of `main.py`
- Updated `data.py` to use `read_jsonl(...)`
- Changed `CDConfig.sci_dir` default to `/content/scifact/data`
- Replaced brittle assertion-based file discovery with proper `FileNotFoundError`

## Folder layout

```text
cd_ecv_modular_v2/
├── cd_ecv/
│   ├── __init__.py
│   ├── config.py
│   ├── io_utils.py
│   ├── data.py
│   ├── inference.py
│   ├── metrics.py
│   ├── models.py
│   ├── pipeline.py
│   ├── retrieval.py
│   └── utils.py
├── main.py
└── README.md
```

## Expected dataset location

By default, the code now looks under:

```python
Path('/content/scifact/data')
```

So these files should exist there:
- `/content/scifact/data/corpus.jsonl`
- `/content/scifact/data/claims_dev.jsonl`

If your files live elsewhere, change `sci_dir` in `cd_ecv/config.py`.
