# Repository Guidelines

## Project Structure & Module Organization
Core reusable code lives in `ppmat/`, which hosts datasets, models (e.g., `models/infgcn`), and utilities used across tasks. Task pipelines sit in top-level folders: `interatomic_potentials/`, `property_prediction/`, `structure_generation/`, `spectrum_elucidation/`, and `electronic_structure/` (training entry points inside module-specific `train.py`). Documentation and hardware details are under `docs/`, while generated artifacts should stay inside `output/`. Keep experiments or one-off analyses outside this repo or clearly labeled to avoid coupling with production code.

## Build, Test, and Development Commands
- `python -m pip install -r requirements.txt` — install PaddlePaddle, pytest, and scientific dependencies.
- `python setup.py build_ext --inplace` — build the `ppmat.models.mattersim` Cython extension before running structure-generation workloads.
- `python -m pip install -e .` — develop against the latest `ppmat` package without repeated reinstalls.
- `pytest ppmat -k <module>` — run targeted unit tests for updated subsystems.
- `python property_prediction/predict.py --model_name=megnet_mp2018_train_60k_e_form --weights_name=best.pdparams --cif_file_path=property_prediction/example_data/cifs --save_path=output/result.csv` — regression check for property prediction pipelines.

## Coding Style & Naming Conventions
Use 4-space indentation, snake_case modules, CamelCase classes, and SCREAMING_SNAKE constants. Run `pre-commit run --all-files` to apply `black`, `isort`, `ruff`, Markdown tab guards, and `.clang_format` for CUDA/C++ bindings. Keep configs in lowercase, hyphen-free YAML (see `electronic_structure/configs`). Prefer typed function signatures inside `ppmat/` and mirror PaddlePaddle-style docstrings for public APIs.

## Testing Guidelines
Pytest is the default harness; tests live beside their feature modules (e.g., `ppmat/models/**/tests`) with names like `test_<feature>.py`. New behavior should include deterministic fixtures or synthetic CIFs stored with the test module. Gate changes with `pytest ppmat -k <module>` plus relevant task suites, and aim for ≥80% line coverage on new files. For scripts, add CLI smoke tests that assert file creation without downloading large checkpoints.

## Commit & Pull Request Guidelines
Recent history favors short, imperative subjects scoped to the touched subsystem (e.g., `update infgcn & fig diffnmr bugs`). Follow the same style, mention task tags (`MLIP`, `PP`, etc.), and reference an issue ID when applicable. PRs should describe architecture impact, list commands executed (build + pytest), call out new assets or configs, and attach screenshots/plots for UI or scientific metrics. When adding pretrained weights, link to external storage instead of committing binaries.

## Configuration & Assets
Store environment-specific overrides in dedicated YAML under each task rather than hard-coding paths. Secrets and proprietary data must stay outside the repo; use `.env` files ignored by Git. Generated checkpoints and CSV outputs belong in `output/` with descriptive filenames (`<task>_<dataset>_<date>.pdparams`) so they can be pruned safely.
