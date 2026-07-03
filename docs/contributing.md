# Contributing

waxMorph is a scientific package, so contributions should keep both the code
and the underlying modelling assumptions open to inspection. A well-formed
pull request answers three questions clearly: which biological or computational
behavior is changed, where that behavior resides in the package, and how the
change can be verified.

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

Because the PyTorch and JAX backends are kept at parity, a change to graph
construction, losses, training, or the simulator and emulator should be mirrored
across both backends and their tests, and the assertions guarding
PyTorch-versus-JAX agreement should not be weakened.

## Documentation expectations

Public functions should state the biological object being represented, the
array shapes expected by the implementation, and whether gradients are intended
to flow through the operation. Vague phrasing such as "processes data" or
"handles simulation" should be avoided in favor of naming the concrete state
variables: positions `X` (shape `[N, 3]`), polarities `P` (shape `[N, 3]`),
radii `R` (shape `[N]`), signaling-molecule concentrations `c`
(shape `[N, num_molecules]`), cell types `CT`, and the contact edges induced by
spatial proximity. By convention, the per-cell molecular state is denoted `c`
and the predicted increments are `dX`, `dP`, and `dc`.

When a modelling choice is an approximation, it should be stated directly. For
example, the contact graph is rebuilt from a detached state snapshot and treated
as fixed within a rollout step, while the continuous node and edge features
remain differentiable. That distinction matters when interpreting the learned
biophysical update rules.

### Docstring conventions

Docstrings follow the Google style that Napoleon renders:

- **Open with an imperative one-line summary.** Write "Build the contact graph
  …", not "This function builds …" or "Builds …". The summary states what the
  call does to the cell state, not what kind of object it is.
- **Do not restate types in the prose.** Type hints already carry the type, and
  `autodoc_typehints = "description"` renders them, so the argument
  descriptions should name the biological role and the array shape — for
  example `X (shape [N, 3])` — rather than repeating `np.ndarray`.
- **Express physics with `.. math::` blocks.** When a function implements a
  governing equation (the soft-sphere force, the graph Laplacian, a reaction
  term, a Hill function), include that equation in a `.. math::` block so the
  rendered docs match the notation `X`, `P`, `R`, `c`, `dX`, `dP`, `dc`.
- **Cross-link the torch and jax counterparts with `See Also`.** Because the
  two backends are kept at parity, a docstring on
  `waxmorph.torch.<thing>` should point to `waxmorph.jax.<thing>` and vice
  versa through a `See Also` section, so a reader of one backend can find its
  twin. Note in the docstring whether gradients are intended to flow through
  the operation.
