# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sphinx_rtd_theme

import fluidml

project = fluidml.__name__
copyright = fluidml.__copyright__
author = fluidml.__author__
release = fluidml.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
    # "sphinx.ext.autosectionlabel",
    # "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
]

source_suffix = [".rst", ".md"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ["_static"]

intersphinx_mapping = {"python": ("http://docs.python.org/3", None)}

autodoc_typehints = "description"
autodoc_mock_imports = ["mongoengine"]
