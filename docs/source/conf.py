# Copyright (c) 2025-2026 Ian Galloway, Prashant K. Jha
# SPDX-License-Identifier: GPL-3.0-or-later
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "MatTO"
copyright = "2025-2026, Ian Galloway and Prashant K. Jha"
author = "Ian Galloway, Prashant K. Jha"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

# FEniCSx (dolfinx, basix, ufl, petsc4py, mpi4py) is an MPI-linked conda
# dependency (see environment.yml) that is not installed where the docs
# are built. Mocking it lets autodoc import the modules and read their
# docstrings and signatures without a real dolfinx.
autodoc_mock_imports = ["dolfinx", "basix", "ufl", "petsc4py", "mpi4py"]
autodoc_member_order = "bysource"

napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/CEADpx/matto",
}
html_static_path = []
