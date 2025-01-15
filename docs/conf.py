# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------

project = 'PHRINGE'
author = 'Philipp A. Huber'
copyright = f'2024, Philipp A. Huber'
# html_theme_options = {
#     "logo_light": "_static/phringe2.png",
#     "logo_dark": "_static/phringe.png"
# }
html_static_path = ['_static']
html_logo = "_static/phringe_logo.png"
html_title = " "
html_theme_options = {
    "logo_only": True,  # Show only the logo, not the project name
}
html_context = {
    "display_github": False,  # Example of context variable
    "title": ""
}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = ['sphinx_copybutton',
              'sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'nbsphinx',
              'recommonmark']

# -- Options for HTML output -------------------------------------------------

master_doc = 'index'
html_theme = "furo"
html_static_path = ['_static']

import os
import sys

sys.path.insert(0, os.path.abspath('../phringe'))
