# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
# without */docs/source/ folder
sys.path.insert(0, os.path.abspath(".."))
# with */docs/source/ folder
# sys.path.insert(0, os.path.abspath("../src"))

project = "PyDASA"
copyright = "2025, @SFAM, Uniandes, DISC, Bogotá D.C. Colombia"
author = "@SFAM"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions to use
# install them with: pip install...
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "autoapi.extension",
    "sphinx.ext.todo",
    "sphinx.ext.intersphinx",
    "sphinx.ext.graphviz",
    "sphinx.ext.duration",
    "sphinx.ext.inheritance_diagram",
    "sphinx_markdown_builder",
    "sphinx_copybutton",
    "sphinx_favicon",
    "sphinx_gitstamp",
    "sphinx-prompt",
    "myst_parser",
    # "docxsphinx",
    # "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store"
]

# internationalization options
language = "en"
locale_dirs = ["locale/"]
gettext_compact = False
languages = ["es", "ja", "de"]

# autoapi configuration
autoapi_dirs = [
    # without */docs/source/ folder
    os.path.join(os.path.dirname(__file__), "..", "src"),
    # with */docs/source/ folder
    # os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")),
]

# myst_parser configuration
source_suffix = {
    ".rst": "restructuredtext",
    # ".txt": "markdown",
    # ".md": "markdown",
}

# myst extensions to enable
# intall them with: pip install...
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]


# favicon configuration
favicons = [
    {
        "sizes": "16x16",
        "href": "https://secure.example.com/favicon/favicon-16x16.png",
    },
    {
        "sizes": "32x32",
        "href": "https://secure.example.com/favicon/favicon-32x32.png",
    },
    {
        "rel": "apple-touch-icon",
        "sizes": "180x180",
        "href": "apple-touch-icon-180x180.png",  # use a local file in _static
    },
]


# -- Options for HTML output -------------------------------------------------
# gitstamp configuration, URL: https://github.com/jdillard/sphinx-gitstamp
# Date format for git timestamps
gitstamp_fmt = "%Y-%m-%d %H:%M:%S %z"

# docxsphinx configuration
# "docx_template" need *.docx or *.dotx template file name. default is None.
# docx_template = "template.docx"

# Language to be used for generating the HTML full-text search index.
# Sphinx supports the following languages:
#   "da", "de", "en", "es", "fi", "fr", "hu", "it", "ja"
#   "nl", "no", "pt", "ro", "ru", "sv", "tr", "zh"
#
html_search_language = "en"

# A dictionary with options for the search language support, empty by default.
# "ja" uses this config value.
# "zh" user can custom change `jieba` dictionary path.
#
html_search_options = {"type": "default"}

# The name of a javascript file (relative to the configuration directory) that
# implements a search results scorer. If empty, the default will be used.
#
html_search_scorer = ""


# configuration for sphinx inheritance diagram
inheritance_graph_attrs = dict(rankdir="LR", size='"6.0, 8.0"',
                               fontsize=14, ratio="compress")

inheritance_node_attrs = dict(shape="ellipse", fontsize=14, height=0.75,
                              color="dodgerblue1", style="filled")


# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# prefered theme
# html_theme = "cloud"
# html_theme = "renku"
# html_theme = "sphinx_rtd_theme"
# html_theme = "sphinx_book_theme"
# https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/install.html#
html_theme = "pydata_sphinx_theme"

# theme options
html_theme_options = {
}

# multi-language and version configuration
html_context = {
    "default_mode": "dark",
    "current_version": release,
    "versions": [["1.0", "link to 1.0"], ["2.0", "link to 2.0"]],
    "current_language": "en",
    "languages": [["en", "link to en"], ["de", "link to de"]]
}

html_static_path = ["_static"]
