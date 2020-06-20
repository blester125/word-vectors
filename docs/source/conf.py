# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import word_vectors


# -- Project information -----------------------------------------------------

project = "word-vectors"
copyright = "2020, Brian Lester"
author = "Brian Lester"

version = word_vectors.__version__
# The full version, including alpha/beta/rc tags
release = word_vectors.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.autosectionlabel",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
# autodoc_typehints = "signature"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Fiddling with the object to force sphinx to follow __wrapped__
# Everything I use with `@file_or_name` uses `@functools.wraps` correctly but
# sphinx doesn't actually use `inspect.signature` or follows the `__wrapped__`
# attribute. By pretending that we are a context manager we force sphinx to
# follow the `__wrapped__`. We do this by adding a hook to the
# `autodoc-before-process-signature` event. In this hook we then we fiddle with
# the `__name__` and `__file__` attributes so that spinx things we are a contexmanager
# I figured out how to trigger this by debugging into
# `sphinx.utils.inspect._should_unwrap`. This is an internal function and could
# change we we should pin our sphinx version.
import contextlib


def munge_sig(app, obj, bound_method):
    obj.__globals__["__name__"] = "contextlib"
    obj.__globals__["__file__"] = contextlib.__file__


def setup(app):
    app.connect("autodoc-before-process-signature", munge_sig)
