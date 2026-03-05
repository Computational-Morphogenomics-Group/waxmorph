"""Sphinx configuration for WaxMorph documentation."""

project = "WaxMorph"
copyright = "2026, WaxMorph Contributors"
author = "WaxMorph Contributors"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.bibtex",
]

# -- General -----------------------------------------------------------------

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
nitpicky = True
suppress_warnings = ["myst.header"]

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
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# -- Bibliography -------------------------------------------------------------

bibtex_bibfiles = ["references.bib"]

# -- HTML output --------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_theme_options = {
    "repository_url": "https://github.com/waxmorph/waxmorph",
    "use_repository_button": True,
}
