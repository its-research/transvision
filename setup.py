import os
import subprocess
from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

# transvision version info
version = '0.1.0'
package_name = 'transvision'
cwd = os.path.dirname(os.path.abspath(__file__))

sha = 'Unknown'
try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
except Exception:
    pass


def write_version_file():
    version_path = os.path.join(cwd, 'transvision', 'version.py')
    with open(version_path, 'w') as f:
        f.write(f'__version__ = {version!r}\n')
        f.write(f'git_version = {repr(sha)}\n')


requirements = ['torch', 'torchvision']

torch_ver = [int(x) for x in torch.__version__.split('.')[:2]]


def make_cuda_ext(name, module, sources, sources_cuda=[], extra_args=[], extra_include_path=[]):

    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
            '-gencode=arch=compute_70,code=sm_70',
            '-gencode=arch=compute_75,code=sm_75',
            '-gencode=arch=compute_80,code=sm_80',
            '-gencode=arch=compute_86,code=sm_86',
        ]
        sources += sources_cuda
    else:
        print('Compiling {} without CUDA'.format(name))
        extension = CppExtension

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    import sys
    from os.path import exists

    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            elif '@git+' in line:
                info['package'] = line
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip, rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


if __name__ == '__main__':
    print(f'Building wheel {package_name}-{version}')

    license = 'None'
    write_version_file()

    setup(
        name='transvision',
        version='0.1.0',
        url='',
        description='transvision',
        license=license,
        packages=find_packages(exclude=(
            'configs',
            'tests',
        )),
        ext_modules=[
            make_cuda_ext(
                name='voxel_layer',
                module='transvision.models.voxel',
                sources=[
                    'src/voxelization.cpp',
                    'src/scatter_points_cpu.cpp',
                    'src/scatter_points_cuda.cu',
                    'src/voxelization_cpu.cpp',
                    'src/voxelization_cuda.cu',
                ]),
            make_cuda_ext(
                name='bev_pool_ext',
                module='transvision.models.bev_pool',
                sources=[
                    'src/bev_pool.cpp',
                    'src/bev_pool_cuda.cu',
                ],
            ),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
