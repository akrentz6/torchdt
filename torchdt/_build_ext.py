from pathlib import Path
import os
import sys
import warnings
from torch.utils.cpp_extension import BuildExtension, include_paths, library_paths

CPU_SOURCES = ("extension.cpp", "registry.cpp", "lns/cpu.cpp")
CUDA_SOURCES = ("lns/cuda/compile_probe.cu",)


def enabled(name):
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


class BuildCxxExtension(BuildExtension):
    user_options = BuildExtension.user_options + [("no-cpp", None, "Skip the native extension")]
    boolean_options = getattr(BuildExtension, "boolean_options", []) + ["no-cpp"]

    def initialize_options(self):
        super().initialize_options()
        self.no_cpp = False

    def finalize_options(self):
        super().finalize_options()
        self.no_cpp = self.no_cpp or enabled("TORCHDT_NO_CPP")
        self.strict = enabled("TORCHDT_STRICT_CPP")
        self.with_cuda = enabled("TORCHDT_BUILD_CUDA")
        if self.no_cpp:
            self.extensions = []
            self.distribution.ext_modules = []
            return
        root = Path(__file__).resolve().parent
        for ext in self.extensions:
            if ext.name != "torchdt._C":
                continue
            ext.optional = not self.strict
            ext.sources = [os.path.relpath(root / "csrc" / s) for s in CPU_SOURCES]
            if self.with_cuda:
                ext.sources += [os.path.relpath(root / "csrc" / s) for s in CUDA_SOURCES]
            # Both CPU and CUDA paths link against the installed PyTorch.
            # Accept the older API spelling used by supported PyTorch releases.
            try:
                includes = include_paths(device_type="cuda" if self.with_cuda else "cpu")
                libraries = library_paths(device_type="cuda" if self.with_cuda else "cpu")
            except TypeError:
                includes = include_paths(cuda=self.with_cuda)
                libraries = library_paths(cuda=self.with_cuda)
            ext.include_dirs = list(ext.include_dirs or []) + includes + [str(root / "include")]
            ext.library_dirs = list(ext.library_dirs or []) + libraries
            ext.libraries = ["c10", "torch", "torch_cpu", "torch_python"]
            if self.with_cuda:
                ext.libraries += ["cudart", "c10_cuda", "torch_cuda"]
            cxx_flags = ["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"]
            ext.extra_compile_args = {"cxx": cxx_flags, "nvcc": ["-O3", "-std=c++17"]}
            # ATen's inline parallel_for is silently serial without _OPENMP
            # when the installed PyTorch uses its OpenMP backend.
            config_header = Path(includes[0]) / "ATen" / "Config.h"
            if "#define AT_PARALLEL_OPENMP 1" in config_header.read_text():
                if os.name == "nt":
                    cxx_flags += ["/openmp"]
                elif sys.platform == "darwin":
                    cxx_flags += ["-Xpreprocessor", "-fopenmp"]
                    ext.libraries += ["omp"]
                else:
                    cxx_flags += ["-fopenmp"]
                    ext.extra_link_args = list(ext.extra_link_args or []) + ["-fopenmp"]
            if os.name != "nt":
                ext.runtime_library_dirs = libraries
            if enabled("TORCHDT_SANITIZE"):
                ext.extra_compile_args["cxx"] += ["-fsanitize=undefined", "-fno-sanitize-recover=all"]
                ext.extra_link_args = list(ext.extra_link_args or []) + ["-fsanitize=undefined"]

    def run(self):
        if self.no_cpp:
            return
        if self.strict:
            return super().run()
        try:
            super().run()
        except Exception as exc:
            warnings.warn(f"C++ backend was not built: {exc}. "
                          "Set TORCHDT_STRICT_CPP=1 to make native build failures fatal.", RuntimeWarning)
