# Contributing

## Development setup

```bash
git clone https://github.com/waxmorph/waxmorph.git
cd waxmorph
pip install -e ".[all]"
pre-commit install
```

## Running tests

```bash
pytest tests/
```

## Code style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.
Pre-commit hooks enforce style automatically:

```bash
pre-commit run --all-files
```

## Pull requests

1. Create a feature branch from `main`
2. Make your changes and add tests
3. Run `pytest` and `ruff check` locally
4. Open a PR against `main`
