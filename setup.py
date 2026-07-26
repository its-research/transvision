import os
from pathlib import Path

from setuptools import Distribution, find_packages, setup
from setuptools.command.build_ext import build_ext


PACKAGE_NAME = "transvision"
ROOT = Path(__file__).resolve().parent
VERSION_FILE = ROOT / "transvision" / "version.py"
CONSTRAINTS_FILE = ROOT / "environments" / "resilient_v2x" / "constraints.txt"


def read_version() -> str:
    namespace: dict[str, str] = {}
    exec(VERSION_FILE.read_text(), namespace)
    return namespace["__version__"]


def parse_requirements(fname: Path = CONSTRAINTS_FILE) -> list[str]:
    return [
        line.strip()
        for line in fname.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def make_cuda_ext(name, module, sources, sources_cuda=(), extra_args=(), extra_include_path=()):
    import torch
    from torch.utils.cpp_extension import CppExtension, CUDAExtension

    define_macros = []
    extra_compile_args = {"cxx": list(extra_args)}
    extension = CppExtension

    if torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1":
        define_macros.append(("WITH_CUDA", None))
        extension = CUDAExtension
        extra_compile_args["nvcc"] = list(extra_args) + [
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-gencode=arch=compute_70,code=sm_70",
            "-gencode=arch=compute_75,code=sm_75",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_86,code=sm_86",
        ]
        sources = list(sources) + list(sources_cuda)
    else:
        print("Compiling {} without CUDA".format(name))

    return extension(
        name="{}.{}".format(module, name),
        sources=[os.path.join(*module.split("."), source) for source in sources],
        include_dirs=list(extra_include_path),
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


def build_extensions():
    return [
        make_cuda_ext(
            name="voxel_layer",
            module="transvision.models.voxel",
            sources=(
                "src/voxelization.cpp",
                "src/scatter_points_cpu.cpp",
                "src/scatter_points_cuda.cu",
                "src/voxelization_cpu.cpp",
                "src/voxelization_cuda.cu",
            ),
        ),
        make_cuda_ext(
            name="bev_pool_ext",
            module="transvision.models.bev_pool",
            sources=("src/bev_pool.cpp", "src/bev_pool_cuda.cu"),
        ),
    ]


class LazyDistribution(Distribution):
    def has_ext_modules(self):
        return True


class LazyBuildExtension(build_ext):
    def build_extensions(self):
        from torch.utils.cpp_extension import BuildExtension

        self.extensions = build_extensions()
        self.distribution.ext_modules = self.extensions
        return BuildExtension.build_extensions(self)


setup(
    name=PACKAGE_NAME,
    version=read_version(),
    url="",
    description=PACKAGE_NAME,
    license="None",
    packages=find_packages(exclude=("configs", "tests")),
    install_requires=parse_requirements(),
    python_requires=">=3.10,<3.11",
    ext_modules=[],
    cmdclass={"build_ext": LazyBuildExtension},
    distclass=LazyDistribution,
    zip_safe=False,
)
