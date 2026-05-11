# Configuration file for the Sphinx documentation builder.

import os
import sys

# -- Path setup --------------------------------------------------------------

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../phringe"))
os.environ["PYTHONPATH"] = os.path.abspath("..")

# -- Project information -----------------------------------------------------

project = "PHRINGE"
author = "Philipp A. Huber"
copyright = "2024, Philipp A. Huber"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "recommonmark",
    "sphinx_contributors",
]

master_doc = "index"

# -- Notebook execution ------------------------------------------------------

nbsphinx_execute = "never"
nb_execution_mode = "off"

# -- HTML output -------------------------------------------------------------

html_theme = "shibuya"
html_title = "PHRINGE Docs"
html_static_path = ["_static"]

html_theme_options = {
    "light_logo": "_static/phringe2_light.png",
    "dark_logo": "_static/phringe2_dark.png",
    "github_url": "https://github.com/pahuber/PHRINGE",
    "globaltoc_expand_depth": 1,
    "accent_color": "blue",
}

# Shows GitHub repo stats, but no edit-page link because html_sidebars below
# does not include "sidebars/edit-this-page.html".
html_context = {
    "source_type": "github",
    "source_user": "pahuber",
    "source_repo": "PHRINGE",
    "source_version": "main",
    "source_docs_path": "/docs/",
}

html_sidebars = {
    "**": [
        "sidebars/localtoc.html",
        "sidebars/repo-stats.html",
    ]
}

html_css_files = [
    "custom.css",
    # "_static/custom.css",
]

pygments_style = "monokai"
pygments_dark_style = "monokai"

# -- Matplotlib configuration ------------------------------------------------

try:
    import matplotlib as mpl

    mpl.rcParams["text.usetex"] = False
except ImportError:
    pass
