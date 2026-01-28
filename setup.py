from os import path
import sys
from setuptools import setup, find_packages
from setuptools.extension import Extension
import argparse

def parse_requirements(file):
    required_packages = []
    with open(path.join(path.dirname(__file__), file)) as req_file:
        for line in req_file:
            required_packages.append(line.strip())
    return required_packages

packages = [x for x in find_packages() if x != "test"]
setup(name='rrank',
      version='0.34',
      description="Relational ranking",
      url='https://github.com/re3py/re3py',
      python_requires='>3.6.0',
      author='Matej Petković, Blaž Škrlj',
      author_email='matej.petkovic@ijs.si',
      license='bsd-3-clause-clear',
      packages=packages,
      zip_safe=False,
      include_package_data=True,
      install_requires=parse_requirements("requirements.txt"))
