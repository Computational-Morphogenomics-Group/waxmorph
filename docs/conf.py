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
    # Third-party runtime annotation names below do not have stable Sphinx
    # intersphinx inventories or are imported under local aliases in docstrings.
    (r"py:class", r"[Oo]ptional"),
    (r"py:class", r"GNS"),
    (r"py:class", r"JaxGNS"),
    (r"py:class", r"Path"),
    (r"py:class", r"TrainConfig"),
    (r"py:class", r"TrainResult"),
    (r"py:class", r"callable"),
    (r"py:class", r"jax\.Array"),
    (r"py:class", r"jnp\.ndarray"),
    (r"py:class", r"np\.ndarray"),
    (r"py:class", r"optax\..*"),
    (r"py:class", r"equinox\..*"),
    (r"py:class", r"eqx\..*"),
    (r"py:class", r"trimesh\..*"),
    (r"py:func", r"trimesh\..*"),
    (r"py:class", r"geomloss\..*"),
    (r"py:class", r"SamplesLoss"),
    (r"py:class", r"pyvista\..*"),
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
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}

# -- Bibliography -------------------------------------------------------------

bibtex_bibfiles = ["references.bib"]

# -- HTML output --------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/waxmorph/waxmorph",
    "use_repository_button": True,
}
