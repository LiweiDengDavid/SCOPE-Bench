# Contributing

Contributions that improve reproducibility, add a well-referenced baseline, fix a model/data contract, or extend cognitive-depth evaluation are welcome.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[multimodal,scoring,hpo,dev]"
```

## Before opening a pull request

```bash
python -m compileall -q core models scripts scoring main.py
python scripts/validate_models.py
pytest -q
```

If local data is available, also run:

```bash
python scripts/validate_short_video_bundle.py
```

New recommendation models must include a model implementation, a matching `configs/models/<ModelName>.yaml`, a paper/source entry in `docs/models.md`, and a minimal validation note. Do not commit raw datasets, generated features, model weights, full scoring outputs, credentials, or machine-specific absolute paths.

Please keep benchmark-affecting changes explicit: describe the data split, negative sampling, evaluation mask, tuned parameters, random seeds, and any deviation from the cited implementation.

