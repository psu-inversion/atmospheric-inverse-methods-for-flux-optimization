"""Describe package metadata for inclusion in a package.

Not yet ready for PyPI.
"""
from __future__ import print_function, division
import os.path
import sys

from setuptools import setup, find_packages

sys.path.append(os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    "doc", "source"))
from conf import man_pages, release  # noqa: E402
authors = man_pages[0][3]

setup(
    version=release,
    author=authors,
    author_email="dfw5129@psu.edu",
    test_suite="inversion.tests",
    package_dir={"": "src"},
    packages=find_packages("src"),
)
