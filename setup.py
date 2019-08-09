# -*- coding: utf-8 -*-

# Learn more: https://github.com/PabRod/pendulum

from setuptools import setup, find_packages


with open('readme.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pendulum',
    version='0.1.0',
    description='Mechanical simulation of non-inertial simple and double pendula',
    long_description=readme,
    author='Pablo Rodríguez-Sánchez',
    author_email='pablo.rodriguez.sanchez@gmail.com',
    url='https://github.com/PabRod/pendulum',
    license=license,
    install_requires=[
          'sdeint',
      ],
    packages=find_packages(exclude=('tests', 'docs', 'vignettes', 'scripts'))
)
