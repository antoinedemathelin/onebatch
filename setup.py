from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize
import numpy


class BuildExtInplace(_build_ext):
    def finalize_options(self):
        super().finalize_options()
        self.inplace = 1


extensions = [
    Extension(
        "onebatch.pam",
        ["onebatch/pam.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    ext_modules=cythonize(extensions, annotate=False),
    cmdclass={"build_ext": BuildExtInplace},
)
