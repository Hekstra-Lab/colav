import sys

sys.stderr.write(
    """
###############################################################
############### Unsupported installation method ###############
###############################################################
colav does not support installation with `python setup.py install`.
Please use `python -m pip install .` or `pip install .` instead.
"""
)
sys.exit(1)

# The below code does not execute, but Github is picky about where it finds Python packaging metadata.
# See: https://github.com/github/feedback/discussions/6456
# To be removed once GitHub catches up.
import os
from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="colav",
    author="Ammaar A. Saeed",
    author_email="aasaeed@college.harvard.edu",
    description=(
        """Calculate protein structural representations (dihedral angles, CA pairwise distances, and strain analysis) for downstream analysis (e.g., PCA, t-SNE, or UMAP)"""
    ),
    license="",
    url="",
    long_description=read("README.md"),
    packages=find_packages(),
)
