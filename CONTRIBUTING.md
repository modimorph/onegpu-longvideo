# Contributing

Thank you for helping.

1. Fork, branch, and keep each pull request to one topic.
2. `pip install -e ".[dev]"`, then `ruff check src tests` and `pytest -q` must pass. CI runs both on CPU.
3. Add a unit test for any pure-Python change. GPU paths need a short note with the command you ran and its output.
4. Experiments: add a row to `docs/EXPERIMENTS.md` with the setting, n, the measured number and the GPU. Failures are as useful as wins.
5. Do not add model weights, datasets, secrets, machine-specific paths, or any real person's likeness or voice without their written consent.

By contributing you agree that your contribution is licensed under the Apache License 2.0.
