#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

# Get common version number (https://stackoverflow.com/a/7071358)
import re
VERSIONFILE="vtspy/version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(name='vtspy',
      version = verstr,
      author='Donggeun Tak',
      author_email='donggeun.tak@gmail.com',
      url='https://github.com/takdg123/vtspy',
      packages = find_packages(include=["vtspy", "vtspy.*"]),
      install_requires = [''],
      description = "VERITAS analysis tool",
      long_description = long_description,
      long_description_content_type="text/markdown",
      )

