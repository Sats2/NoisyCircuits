# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'NoisyCircuits'
copyright = '2025, Sathyamurthy Hegde'
author = 'Sathyamurthy Hegde'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
    'sphinxcontrib.bibtex',
    'nbsphinx'
]

nbsphinx_execute = "never"

templates_path = ['_templates']
exclude_patterns = []
bibtex_bibfiles = ['references.bib']
bibtex_default_style = "unsrt"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_context = {
    "display_github": True,
    "github_user": "Sats2",
    "github_repo": "NoisyCircuits",
    "github_version": "main",
    "conf_py_path": "/README.md",
}