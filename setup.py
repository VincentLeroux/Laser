import sys
from setuptools import setup, find_packages

setup(
    name='Laser',
    version=1.0,
    description='Laser simulation tools',
    maintainer='Vincent Leroux',
    packages=find_packages('.'),
    tests_require=['pytest', 'openpmd_viewer'],
    include_package_data=True,
    platforms='any'
    )

