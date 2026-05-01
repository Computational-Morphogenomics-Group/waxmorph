# Contributing

WaxMorph is a scientific package, so contributions should make both the code
and the modelling assumptions easier to inspect. A useful pull request should
answer three questions clearly: what biological or computational behavior is
being changed, where that behavior lives in the package, and how a user can
verify it.

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

The default pytest configuration writes terminal and XML coverage reports for
the `waxmorph` package. The full test suite includes both PyTorch and JAX
coverage, so development environments should install `.[all]` or at least
`.[learning,simulation,jax]` before running it.

## Code style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and
[black](https://black.readthedocs.io/) for formatting.
Pre-commit hooks enforce style automatically:

```bash
pre-commit run --all-files
```

## Pull requests

1. Create a feature branch from `main`
2. Make your changes and add tests
3. Run `pytest`, `ruff check waxmorph tests`, and `black --check waxmorph tests` locally
4. Open a PR against `main`

## Documentation expectations

Public functions should explain the biological object being represented, the
array shapes expected by the implementation, and whether gradients are meant to
flow through the operation. Avoid vague phrases such as "processes data" or
"handles simulation"; name the concrete state variables, for example positions
`X`, polarities `P`, radii `R`, gene state `G`, cell types `CT`, or contact
edges.

When a modelling choice is an approximation, say so directly. For example,
graph topology is rebuilt from state snapshots and treated as fixed within a
rollout step, while continuous node and edge features can remain
differentiable. That distinction is important for users interpreting learned
biophysical rules.
