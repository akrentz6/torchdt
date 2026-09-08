from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs" / "_ext"))
project = "TorchDT"
author = "Alex Krentz"
release = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
version = release
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode", "registered_functions"]
autodoc_inherit_docstrings = False
autodoc_member_order = "bysource"
autodoc_typehints = "none"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "alabaster"
html_title = f"TorchDT {release}"
html_theme_options = {"description": "Custom number formats for PyTorch", "fixed_sidebar": True}
html_sidebars = {"**": ["about.html", "navigation.html", "searchfield.html"]}
