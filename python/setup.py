from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages, Distribution
import io

# To use a consistent encoding
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with io.open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

compile_mode = open(path.join(here, '.compile_mode'), 'r').read().strip()
if compile_mode == 'gpu':
    name = 'simnets'
else:
    name = 'simnets-cpu'

setup(
    name=name,

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='1.0',

    description='SimNets implementation in tensorflow',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/HUJI-Deep/simnets-tf',

    # Author details
    author='HUJI Deep Learning Group',
    author_email='or+huji.deep@sharir.org',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: C++',
        'Programming Language :: Python :: Implementation :: CPython',
        'Operating System :: POSIX :: Linux',
    ],

    # What does your project relate to?
    keywords='tensorflow simnets machine-learning keras',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(),

    install_requires=['tensorflow-gpu', 'keras'],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
        'simnets.ops': ['libmex_dims_helper.so', 'simnet_ops.so'],
    },
)
