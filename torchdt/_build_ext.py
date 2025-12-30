from torch.utils.cpp_extension import BuildExtension, include_paths
from pathlib import Path
import os

class BuildCxxExtension(BuildExtension):
    """
    Build all .cpp files in csrc/ and include Torch headers.
    Fails hard if compilation fails.
    """

    user_options = BuildExtension.user_options + [
        ("no-cpp", None, "Do not build the C++ extension"),
    ]
    boolean_options = getattr(BuildExtension, "boolean_options", []) + ["no-cpp"]

    def initialize_options(self):
        super().initialize_options()
        self.no_cpp = False

    def finalize_options(self):
        super().finalize_options()
        if os.environ.get("TORCHDT_NO_CPP", "").lower() in {"1", "true", "yes", "on"}:
            self.no_cpp = True

    def build_extensions(self):
        if self.no_cpp:
            self.extensions = []
            return

        src_dir = Path(__file__).resolve().parent / "csrc"
        include_dir = Path(__file__).resolve().parent / "include"
        cpp_files = [str(p) for p in src_dir.glob("*.cpp")]

        torch_includes = include_paths()

        for ext in self.extensions:
            if ext.name == "torchdt._C":
                ext.sources = cpp_files
            ext.include_dirs = list(ext.include_dirs or [])
            ext.include_dirs += torch_includes
            ext.include_dirs += [str(include_dir)]

        super().build_extensions()
