# roifile/setup.py

"""Roifile package setuptools script."""

import sys
import re

from setuptools import setup

with open('roifile/roifile.py') as fh:
    code = fh.read()

version = re.search(r"__version__ = '(.*?)'", code).groups()[0]

description = re.search(r'"""(.*)\.(?:\r\n|\r|\n)', code).groups()[0]

readme = re.search(
    r'(?:\r\n|\r|\n){2}"""(.*)"""(?:\r\n|\r|\n){2}[__version__|from]',
    code,
    re.MULTILINE | re.DOTALL,
).groups()[0]

readme = '\n'.join(
    [description, '=' * len(description)] + readme.splitlines()[1:]
)

license = re.search(
    r'(# Copyright.*?(?:\r\n|\r|\n))(?:\r\n|\r|\n)+""',
    code,
    re.MULTILINE | re.DOTALL,
).groups()[0]

license = license.replace('# ', '').replace('#', '')

if 'sdist' in sys.argv:
    with open('LICENSE', 'w') as fh:
        fh.write('BSD 3-Clause License\n\n')
        fh.write(license)
    with open('README.rst', 'w') as fh:
        fh.write(readme)

setup(
    name='roifile',
    version=version,
    description=description,
    long_description=readme,
    author='Christoph Gohlke',
    author_email='cgohlke@uci.edu',
    license='BSD',
    url='https://www.lfd.uci.edu/~gohlke/',
    project_urls={
        'Bug Tracker': 'https://github.com/cgohlke/roifile/issues',
        'Source Code': 'https://github.com/cgohlke/roifile',
        # 'Documentation': 'https://',
    },
    packages=['roifile'],
    entry_points={'console_scripts': ['roifile = roifile.roifile:main']},
    python_requires='>=3.8',
    install_requires=['numpy>=1.19.2'],
    extras_require={'all': ['matplotlib>=3.3', 'tifffile>=2021.11.2']},
    platforms=['any'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
