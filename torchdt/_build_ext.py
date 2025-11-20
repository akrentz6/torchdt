from torch.utils.cpp_extension import BuildExtension, include_paths
from pathlib import Path
import shutil

def copy_headers(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for header in src_dir.glob("*.h"):
        shutil.copy(header, dst_dir)

class BuildCxxExtension(BuildExtension):
    """
    Build all .cpp files in csrc/ and include Torch headers.
    Fails hard if compilation fails.
    """
    def build_extensions(self):
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
