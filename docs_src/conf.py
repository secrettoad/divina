import sys
import pathlib

sys.path.insert(0, pathlib.Path(__file__).parent.parent)

extensions = ["sphinx_click", "sphinx.ext.autodoc", "sphinx-jsonschema"]

project = "divina"

autodoc_member_order = "bysource"

html_static_path = ["_static"]

html_css_files = [
    "css/docs.css",
]

html_js_files = [
    "js/docs.js",
]
