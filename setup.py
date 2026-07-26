import os
from pathlib import Path

from setuptools import Extension, find_packages, setup
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


def native_extension_metadata():
    return [
        Extension(
            name="transvision.models.voxel.voxel_layer",
            sources=[
                "transvision/models/voxel/src/voxelization.cpp",
                "transvision/models/voxel/src/scatter_points_cpu.cpp",
                "transvision/models/voxel/src/scatter_points_cuda.cu",
                "transvision/models/voxel/src/voxelization_cpu.cpp",
                "transvision/models/voxel/src/voxelization_cuda.cu",
            ],
        ),
        Extension(
            name="transvision.models.bev_pool.bev_pool_ext",
            sources=[
                "transvision/models/bev_pool/src/bev_pool.cpp",
                "transvision/models/bev_pool/src/bev_pool_cuda.cu",
            ],
        ),
    ]


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


class LazyBuildExtension(build_ext):
    def finalize_options(self):
        self._delegate_options = {}
        for option, _, _ in self.user_options:
            name = option.rstrip("=").replace("-", "_")
            value = getattr(self, name, None)
            if value is not None:
                self._delegate_options[name] = value
        super().finalize_options()

    def run(self):
        from torch.utils.cpp_extension import BuildExtension

        self.distribution.ext_modules = build_extensions()
        delegate = BuildExtension(self.distribution)
        for name, value in self._delegate_options.items():
            setattr(delegate, name, value)
        delegate.ensure_finalized()
        delegate.extensions = self.distribution.ext_modules
        return delegate.run()


setup(
    name=PACKAGE_NAME,
    version=read_version(),
    url="",
    description=PACKAGE_NAME,
    license="None",
    packages=find_packages(exclude=("configs", "tests")),
    install_requires=parse_requirements(),
    python_requires=">=3.10,<3.11",
    ext_modules=native_extension_metadata(),
    cmdclass={"build_ext": LazyBuildExtension},
    zip_safe=False,
)
