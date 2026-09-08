"""Render public PyTorch aliases and real wrapper signatures from registrations."""
import ast
from pathlib import Path

from docutils import nodes
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective


class RegisteredFunctions(SphinxDirective):
    required_arguments = 1

    def run(self):
        module = self.arguments[0]
        source = Path(self.env.srcdir).parent / "torchdt" / "funcs" / f"{module}.py"
        self.env.note_dependency(str(source))
        tree = ast.parse(source.read_text())
        lines = []
        for function in tree.body:
            if not isinstance(function, ast.FunctionDef):
                continue
            aliases = []
            for decorator in function.decorator_list:
                if (isinstance(decorator, ast.Call)
                        and ast.unparse(decorator.func) == "DType.register_func"):
                    aliases.extend(ast.unparse(arg) for arg in decorator.args)
            if not aliases:
                continue
            # Use the public alias as the heading; autodoc supplies the wrapper
            # signature underneath without inventing a native torch.dtype API.
            title = aliases[0]
            lines.extend([title, "~" * len(title), "",
                          "Entry points: " + ", ".join(f"``{a}``" for a in aliases) + ".", "",
                          f".. autofunction:: torchdt.funcs.{module}.{function.name}", ""])
        container = nodes.section()
        container.document = self.state.document
        self.state.nested_parse(StringList(lines, source=str(source)), 0, container,
                                match_titles=True)
        return container.children


def setup(app):
    app.add_directive("torchdt-functions", RegisteredFunctions)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
