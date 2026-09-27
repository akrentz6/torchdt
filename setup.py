from pathlib import Path
import runpy
from setuptools import Extension, setup
from setuptools.command.build_py import build_py

build = runpy.run_path(str(Path(__file__).parent / "torchdt" / "_build_ext.py"))
no_cpp = build["enabled"]("TORCHDT_NO_CPP")


class BuildPython(build_py):
    def run(self):
        super().run()
        if no_cpp:
            # A prior native build may share this build directory, we don't
            # want to ship its extension in a Python-only wheel
            package = Path(self.build_lib) / "torchdt"
            for pattern in ("_C*.so", "_C*.pyd", "_C*.dylib"):
                for artifact in package.glob(pattern):
                    artifact.unlink()


setup(
    ext_modules=[] if no_cpp else [Extension(
        "torchdt._C", sources=["torchdt/csrc/" + s for s in build["CPU_SOURCES"]],
        language="c++", optional=not build["enabled"]("TORCHDT_STRICT_CPP"),
    )],
    cmdclass={"build_ext": build["BuildCxxExtension"], "build_py": BuildPython},
)
