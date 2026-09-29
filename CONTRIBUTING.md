# Contributing

Contributions that improve reproducibility, portability, documentation, or test coverage are welcome.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

Run the complete local check before opening a pull request:

```bash
pytest -q
paired-eval verify-release
```

## Pull requests

- Keep experimental definitions and reported values synchronized with `results/paper_results.json`.
- Add or update synthetic tests when changing split logic, offsets, metrics, or artifact validation.
- Do not commit dataset files or participant-level data.
- Keep preprocessing within the relevant training partition.
- Document changes to CLI inputs or output contracts.

Please keep each pull request focused and describe the motivation, implementation, and validation performed.
