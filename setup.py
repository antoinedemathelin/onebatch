from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize

# Custom build_ext command to enable "inplace" by default
class BuildExtInplace(_build_ext):
    def finalize_options(self):
        super().finalize_options()
        self.inplace = 1  # Enable in-place build

setup(
    name="onebatchpam",
    version="0.1.0",
    description="OneBatchPAM",
    author="Antoine de Mathelin",
    author_email="antoine.demat@gmail.com",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your_username/your_repository",
    packages=find_packages(),
    ext_modules = cythonize("onebatchpam/pam.pyx", annotate=False),
    install_requires=["numpy>=1.19.0", "Cython>=0.29.0", "scikit-learn>=1.2.0"],
    cmdclass={"build_ext": BuildExtInplace},
)