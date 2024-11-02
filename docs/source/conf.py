# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from importlib.metadata import version as get_version
from packaging.version import Version

sys.path.insert(0, os.path.abspath("../.."))


project = "regularizepsf"
copyright = "2024, J. Marcus Hughes and the PUNCH Science Operations Center"
author = "J. Marcus Hughes and the PUNCH Science Operations Center"

release: str = get_version("regularizepsf")
version: str = release
_version = Version(release)
if _version.is_devrelease:
    version = release = f"{_version.base_version}.dev{_version.dev}"
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["autoapi.extension",
              "sphinx.ext.autodoc",
              "sphinx.ext.napoleon",
              "nbsphinx",
              "IPython.sphinxext.ipython_console_highlighting"]

templates_path = ["_templates"]
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_show_sourcelink = False
html_static_path = ["_static"]
html_theme_options = {
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/punch-mission/regularizepsf",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
    "show_nav_level": 1,
    "show_toc_level": 3,
}
html_context = {
    # "github_url": "https://github.com", # or your GitHub Enterprise site
    "github_user": "punch-mission",
    "github_repo": "regularizepsf",
    "github_version": "main",
    "doc_path": "docs/source/",
}


autoapi_dirs = ["../../regularizepsf"]
autoapi_python_class_content = "both"
