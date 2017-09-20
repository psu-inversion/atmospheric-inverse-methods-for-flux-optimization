"""Describe package metadata for inclusion in a package.

Not yet ready for PyPI.
"""
import os.path
import sys

from setuptools import setup, find_packages

sys.path.append(os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    "doc", "source"))
from conf import man_pages, release  # noqa: E402
authors = man_pages[0][3]

with open("README.rst", "r") as in_file:
    ldesc = in_file.read()

setup(
    name="inversion",
    version=release,
    description="A module for geophysical inversions",
    long_description=ldesc,
    author=authors,
    author_email="dfw5129@psu.edu",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Science/Engineering :: Atmospheric Science",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
    ],
    keywords="inversion underdetermined DA assimilation",
    package_dir={'': "src"},
    packages=find_packages("src"),
    install_requires=[
        "six",
        "numpy",
        "scipy",
        "dask[array]",
    ],
    extras_require=dict(
        homework=[
            "pandas",
            "statsmodels",
            "matplotlib",
            # "git://github.com/Scitools/iris.git#egg=iris",
        ],
        examples=[
            # "git+https://github.com/Scitools/iris.git#egg=iris",
            "xarray",
            "cf_units",
        ],
    ),
    tests_require=[
        "unittest2",
    ],
    test_suite="inversion.tests",
)
