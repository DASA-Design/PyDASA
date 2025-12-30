# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os
from pydasa._version import __version__

# without */docs/source/ folder
sys.path.insert(0, os.path.abspath("../../src"))

project = "PyDASA"
copyright = "2025, sa-artea, Uniandes, DISC, Bogotá D.C. Colombia"
author = "sa-artea"

# Get version from package
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions to use
# install them with: pip install...
extensions = [
    "autoapi.extension",
    "sphinx.ext.todo",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.graphviz",
    "sphinx.ext.duration",
    "sphinx.ext.inheritance_diagram",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx_design",  # For grid cards like NumPy
    "sphinx_markdown_builder",
    "sphinx_copybutton",
    "sphinx_favicon",
    "sphinx_gitstamp",
    "sphinx-prompt",
]

# Napoleon settings (Google/NumPy docstring style)
# napoleon_google_docstring = True
# napoleon_numpy_docstring = True

# AutoDoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**/__pycache__",
    "**/test_*.py",
]

# internationalization options
language = "en"
locale_dirs = ["locale/"]
gettext_compact = False
languages = ["es", "ja", "de"]

# autoapi configuration
autoapi_dirs = [
    os.path.join(os.path.dirname(__file__), "..", "..", "src"),
]

autoapi_type = "python"
autoapi_template_dir = "_templates/autoapi"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

autoapi_python_use_implicit_namespaces = True
autoapi_python_class_content = "both"

autoapi_ignore = [
    "*/tests/*",
    "*test_*.py",
    "*/__pycache__/*",
]
autoapi_add_toctree_entry = True
autoapi_keep_files = False  # Clean up generated files after build

# myst_parser configuration
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

# myst extensions to enable
# install them with: pip install...
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
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# gitstamp configuration, URL: https://github.com/jdillard/sphinx-gitstamp
# Date format for git timestamps
gitstamp_fmt = "%Y-%m-%d %H:%M:%S %z"

# Language to be used for generating the HTML full-text search index.
html_search_language = "en"

# A dictionary with options for the search language support, empty by default.
html_search_options = {"type": "default"}

# The name of a javascript file (relative to the configuration directory) that
# implements a search results scorer. If empty, the default will be used.
html_search_scorer = ""

# configuration for sphinx inheritance diagram
inheritance_graph_attrs = dict(
    rankdir="LR",
    size="6.0, 8.0",
    fontsize=14,
    ratio="compress"
)

inheritance_node_attrs = dict(
    shape="ellipse",
    fontsize=14,
    height=0.75,
    color="dodgerblue1",
    style="filled"
)

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
    # TODO uncomment when ready
    # "logo": {
    #     "image_light": "_static/logo.png",
    #     "image_dark": "_static/logo.png",
    # },
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_prev_next": True,
}

# multi-language and version configuration
html_context = {
    "default_mode": "default",
    "current_version": release,
    "versions": [["1.0", "link to 1.0"], ["2.0", "link to 2.0"]],
    "current_language": "en",
    "languages": [["en", "link to en"], ["de", "link to de"]]
}

html_static_path = ["_static"]
html_logo = "_static/logo.png"

# viewcode specific config
# viewcode_import = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}
