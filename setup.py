#!/usr/bin/env python3
import os

from setuptools import setup, find_packages


def get_version():
    with open(os.path.join(os.path.dirname(__file__), 'orcinus/VERSION'), 'rb') as f:
        return f.read().decode('ascii').strip()


setup(
    name="orcinus",
    version=get_version(),
    author="Vasiliy Sheredeko <piphon@gmail.com>",
    license="Copyright 2018-2019 (C) Vasiliy Sheredeko",
    description="Compiler for Python-like static typing language",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'orcinus = orcinus.cli:main',
        ],
    },
    install_requires=[
        'attrs==18.1.0',
        'llvmlite==0.26',
        'multidict==4.5.2',
        'multimethod==1.0',
        'colorlog==3.1.4',
        'json-rpc == 1.11.1',
        'pytest==3.6.1',
    ],
    extras_require={
        'mkdocs': [
            'mkdocs',
            'mkdocs-material',
            'markdown-checklist',
            'pygments',
        ]
    },
    include_package_data=True,
)
