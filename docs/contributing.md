# Contributing

A good pull request answers three questions: what biological or computational
behavior changes, where that behavior lives in the package, and how the change
is verified.

## Development setup

```bash
git clone https://github.com/Computational-Morphogenomics-Group/waxmorph.git
cd waxmorph
pip install -e ".[all]"
pre-commit install
```

## Running tests

```bash
pytest tests/
```

The default pytest configuration writes terminal and XML coverage reports for
the `waxmorph` package. The full suite exercises both the PyTorch and JAX
backends, so a development environment should install `.[all]`, or at minimum
`.[learning,simulation,jax]`, before running it.

## Code style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and
[black](https://black.readthedocs.io/) for formatting.
Pre-commit hooks enforce style automatically:

```bash
pre-commit run --all-files
```

## Pull requests

1. Create a feature branch from `main`.
2. Make the changes and add tests.
3. Run `pytest`, `ruff check waxmorph tests`, and `black --check waxmorph tests` locally.
4. Open a pull request against `main`.

When Torch and JAX share a behavioral contract, update both implementations and
their agreement tests. Preserve documented differences in graph tuples,
distance stabilization, loss families, PRNG, checkpoints, and Warp bridges.

## Documentation expectations

Public functions should state which biological object they represent, the array
shapes they expect, and whether gradients flow through the operation. Use the
concrete state variables and shapes catalogued in
{doc}`explanation/cell_state`, together with the contact edges induced by
spatial proximity. By convention, the per-cell molecular state is denoted `c`
and the predicted increments are `dX`, `dP`, and `dc`.

When a modelling choice is an approximation, say so directly. For example, the
contact graph is rebuilt from a detached state snapshot and treated as fixed
within a rollout step, while the continuous node and edge features remain
differentiable. That distinction matters when interpreting the learned
biophysical update rules.

### Docstring conventions

Docstrings follow the Google style that Napoleon renders:

- **Open with an imperative one-line summary.** Use "Build the contact graph
  ...". Focus the summary on the cell-state transition.
- **Let type hints carry types.** Argument descriptions name the biological role
  and array shape, such as `X (shape [N, 3])`;
  `autodoc_typehints = "description"` renders the type.
- **Express physics with `.. math::` blocks.** When a function implements a
  governing equation (the soft-sphere force, the graph Laplacian, a reaction
  term, a Hill function), include that equation in a `.. math::` block so the
  rendered docs match the notation `X`, `P`, `R`, `c`, `dX`, `dP`, `dc`.
- **Cross-link backend counterparts with `See Also`.** A docstring on
  `waxmorph.torch.<thing>` should point to `waxmorph.jax.<thing>` and vice
  versa when both exist. State relevant interface or gradient differences.
