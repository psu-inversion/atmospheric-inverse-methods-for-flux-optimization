import os.path
import sys

from setuptools import setup, find_packages

sys.path.append(os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    "doc", "source"))
from conf import version, man_pages
authors = man_pages[0][3]

with open("README.rst", "r") as in_file:
    ldesc = in_file.read()

setup(
    name="inversion",
    version=version,
    description="A module for geophysical inversions",
    long_description=ldesc,
    author=authors,
    author_email="dfw5129@psu.edu",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Science/Engineering :: Atmospheric Science",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
    ],
    keywords="inversion underdetermined DA assimilation",
    package_dir={'': "src"},
    packages=find_packages("src"),
    install_requires=[
        "six",
        "numpy<1.12",
    ],
    extras_require=dict(
        dask=["dask[array]"],
    ),
    tests_require=[
        "unittest2",
    ],
    test_suite="inversion.tests",
)
