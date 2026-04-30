"""Sphinx configuration for the waxmorph documentation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC if SRC.exists() else ROOT))

project = "waxmorph"
copyright = "2026, WaxMorph Contributors"
author = "WaxMorph Contributors"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = False

# -- General -----------------------------------------------------------------

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
nitpicky = False
suppress_warnings = ["myst.header"]
nitpick_ignore_regex = [
    # Local aliases and runtime-only annotation strings below are emitted by
    # autodoc from import-time objects rather than from explicit docstring
    # references. Keep package API docstrings linked through intersphinx
    # mappings instead of adding package-wide ignores here.
    (r"py:class", r"[Oo]ptional"),
    (r"py:class", r"GNS"),
    (r"py:class", r"JaxGNS"),
    (r"py:class", r"Path"),
    (r"py:class", r"TrainConfig"),
    (r"py:class", r"TrainResult"),
    (r"py:class", r"callable"),
    (r"py:class", r"SamplesLoss"),
    (r"py:class", r"optax\.OptState"),
    (r"py:class", r"jnp\.ndarray"),
    (r"py:class", r"np\.ndarray"),
    (r"py:class", r"eqx\..*"),
    (r"py:class", r"array"),
    (r"py:class", r"ndim=.*"),
    (r"py:class", r"dtype=.*"),
    (r"py:class", r"dtype=wp\..*"),
    (r"py:class", r"same shape as G"),
    (r"py:class", r"warp\._src\.types\.HashGrid"),
    (r"py:class", r"warp\._src\.types\.array"),
    (r"py:class", r"warp\._src\..*"),
    (r"py:class", r"warp\.Tape"),
    (r"py:class", r"wp\..*"),
]

# -- MyST / notebooks --------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
]
nb_execution_mode = "off"

# -- Intersphinx --------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "equinox": ("https://docs.kidger.site/equinox/", None),
    "optax": ("https://optax.readthedocs.io/en/latest/", None),
    "trimesh": ("https://trimesh.org/", None),
    "pyvista": ("https://docs.pyvista.org/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "imageio": ("https://imageio.readthedocs.io/en/stable/", None),
    "warp": ("https://nvidia.github.io/warp/", None),
    "geomloss": ("https://www.kernel-operations.io/geomloss/", None),
}

# -- Bibliography -------------------------------------------------------------

bibtex_bibfiles = ["references.bib"]

# -- HTML output --------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/waxmorph/waxmorph",
    "use_repository_button": True,
}
