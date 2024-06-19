from setuptools import setup, find_packages

setup(
    name='dynrn',
    version='0.0.1',
    packages=find_packages(include=["dynrn", 'dynrn.*']),
)