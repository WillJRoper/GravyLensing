"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import re
from datetime import datetime
from pathlib import Path

# Single source of truth for the version is the top-level CMakeLists.txt.
_cmakelists = (Path(__file__).parents[2] / "CMakeLists.txt").read_text()
_match = re.search(r'set\(GRAVY_VERSION\s+"([^"]+)"', _cmakelists)

project = "GravyLensing"
copyright = f"{datetime.now().year}, Will Roper"
author = "Will Roper"
release = _match.group(1) if _match else "0.0.0"
version = release

extensions = [
    "sphinx_copybutton",
]

exclude_patterns = []
master_doc = "index"

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = "GravyLensing"
html_show_sourcelink = False
html_static_path = ["_static"]
html_logo = "_static/gravylensing-logo.png"
html_theme_options = {
    "source_repository": "https://github.com/WillJRoper/gravy-lensing/",
    "source_branch": "main",
    "source_directory": "docs/source/",
}
