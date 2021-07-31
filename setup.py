#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='project',
    version='1.0.0',
    description='Few shot classifier project',
    author='Sergiu Ovidiu Oprea',
    author_email='soprea@dtic.ua.es',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/sergiuoprea/clip_with_few_shots',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)

