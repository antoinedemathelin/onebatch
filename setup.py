from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize
import numpy
import sys
import os

class BuildExtInplace(_build_ext):
    def finalize_options(self):
        super().finalize_options()
        self.inplace = 1

def get_openmp_flags():
    """Get OpenMP compilation flags"""
    if sys.platform == 'darwin':  # macOS
        try:
            homebrew_prefix = os.popen('brew --prefix').read().strip()
            libomp_path = f"{homebrew_prefix}/opt/libomp"
            
            if os.path.exists(libomp_path):
                return {
                    'extra_compile_args': ['-Xpreprocessor', '-fopenmp'],
                    'extra_link_args': ['-lomp'],
                    'include_dirs': [f"{libomp_path}/include"],
                    'library_dirs': [f"{libomp_path}/lib"]
                }
        except:
            pass
    
    return {
        'extra_compile_args': ['-fopenmp'],
        'extra_link_args': ['-fopenmp'],
        'include_dirs': [],
        'library_dirs': []
    }

# Check if building both versions
build_openmp = os.path.exists("onebatch/pam_openmp.pyx")

extensions = [
    Extension(
        "onebatch.pam",
        ["onebatch/pam.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

if build_openmp:
    print("Building with OpenMP support")
    openmp_flags = get_openmp_flags()
    extensions.append(
        Extension(
            "onebatch.pam_openmp",
            ["onebatch/pam_openmp.pyx"],
            extra_compile_args=openmp_flags['extra_compile_args'],
            extra_link_args=openmp_flags['extra_link_args'],
            include_dirs=[numpy.get_include()] + openmp_flags['include_dirs'],
            library_dirs=openmp_flags['library_dirs'],
        )
    )
else:
    print("Building without OpenMP support")

setup(
    ext_modules=cythonize(extensions, annotate=False),
    cmdclass={"build_ext": BuildExtInplace},
)