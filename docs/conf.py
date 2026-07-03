"""Sphinx configuration for the waxMorph documentation."""

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC if SRC.exists() else ROOT))

# Force-tracked root notebooks rendered as tutorials. The copies land under
# docs/tutorials/ and are picked up by the existing *.ipynb gitignore rule, so
# they never enter version control. The source notebooks are left untouched.
TUTORIAL_NOTEBOOKS = {
    "simulation_with_autodiff.ipynb": "forward_simulator.ipynb",
    "shape_assembly.ipynb": "learned_emulator.ipynb",
    "shape_assembly_jax.ipynb": "learned_emulator_jax.ipynb",
}


def _copy_tutorial_notebooks(app):
    """Copy the tracked root notebooks into docs/tutorials/ before the build."""
    dest_dir = HERE / "tutorials"
    dest_dir.mkdir(exist_ok=True)
    for source_name, dest_name in TUTORIAL_NOTEBOOKS.items():
        source = ROOT / source_name
        if source.exists():
            shutil.copy(source, dest_dir / dest_name)


def setup(app):
    """Register the notebook-copy hook on builder startup."""
    app.connect("builder-inited", _copy_tutorial_notebooks)

project = "waxMorph"
copyright = "2026, waxMorph Contributors"
author = "waxMorph Contributors"
# Sidebar/tab title: just the brand, not "waxMorph <version> documentation".
html_title = "waxMorph"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx_design",
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
    # NVIDIA Warp publishes no intersphinx inventory (objects.inv); omit it so the
    # build does not emit a fetch warning. ``warp.*`` cross-refs stay unlinked.
    "geomloss": ("https://www.kernel-operations.io/geomloss/", None),
}

# -- Bibliography -------------------------------------------------------------

bibtex_bibfiles = ["references.bib"]

# -- HTML output --------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/Computational-Morphogenomics-Group/waxmorph",
    "use_repository_button": True,
    # Brand mark at the top of the left sidebar in place of the text title.
    # The theme swaps the light/dark variant with the color mode (the dark
    # file is shadow-lifted so the navy ribbon separates from near-black pages).
    "logo": {
        "image_light": "_static/logo-512w.png",
        "image_dark": "_static/logo-dark-512w.png",
        "alt_text": "waxMorph",
        "text": "",
    },
}

# Browser favicon from the logo-icon package. ``html_title`` still names the
# browser tab. The extra icon links (svg/apple-touch/manifest) are wired in
# ``_templates/layout.html``.
html_favicon = "_static/favicon.ico"

# Landing-page hero, card grid, and sidebar-logo styling.
html_css_files = ["css/waxmorph.css", "css/gallery.css"]

# Static assets (images, CSS, JS) live here and are copied verbatim into the
# built site under ``_build/html/_static/``. Drop documentation images in
# ``_static/images/`` and reference them with a source-root-relative path,
# e.g. ``.. figure:: /_static/images/example.png``.
html_static_path = ["_static"]
