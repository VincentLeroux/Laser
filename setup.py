import sys
from setuptools import setup, find_packages

setup(
    name='Laser',
    version=1.0,
    description='Laser simulation tools',
    maintainer='Vincent Leroux',
    packages=find_packages('.'),
    include_package_data=True,
    platforms='any'
    )

