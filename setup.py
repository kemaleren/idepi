#!/usr/bin/env python

from __future__ import division, print_function

import sys

from os.path import abspath, join, split
from setuptools import setup

from idepi import __version__ as _idepi_version

setup(name='idepi',
      version=_idepi_version,
      description='IDentify EPItope',
      author='N Lance Hepler',
      author_email='nlhepler@gmail.com',
      url='http://github.com/veg/idepi',
      license='GNU GPL version 3',
      packages=[
            'idepi',
            'idepi.alphabet',
            'idepi.argument',
            'idepi.databuilder',
            'idepi.datasource',
            'idepi.filter',
            'idepi.future',
            'idepi.globber',
            'idepi.hmmer',
            'idepi.labeledmsa',
            'idepi.labeler',
            'idepi.logging',
            'idepi.normalvalue',
            'idepi.phylogeny',
            'idepi.results',
            'idepi.scorer',
            'idepi.scripts',
            'idepi.simulation',
            'idepi.test',
            'idepi.util'
      ],
      package_dir={
            'idepi': 'idepi',
            'idepi.alphabet': 'idepi/alphabet',
            'idepi.argument': 'idepi/argument',
            'idepi.databuilder': 'idepi/databuilder',
            'idepi.datasource': 'idepi/datasource',
            'idepi.filter': 'idepi/filter',
            'idepi.future': 'idepi/future',
            'idepi.globber': 'idepi/globber',
            'idepi.hmmer': 'idepi/hmmer',
            'idepi.labeledmsa': 'idepi/labeledmsa',
            'idepi.labeler': 'idepi/labeler',
            'idepi.logging': 'idepi/logging',
            'idepi.normalvalue': 'idepi/normalvalue',
            'idepi.phylogeny': 'idepi/phylogeny',
            'idepi.results': 'idepi/results',
            'idepi.scorer': 'idepi/scorer',
            'idepi.scripts': 'idepi/scripts',
            'idepi.simulation': 'idepi/simulation',
            'idepi.test': 'idepi/test',
            'idepi.util': 'idepi/util',
      },
      package_data={'idepi': ['data/*', 'hyphy/*.bf']},
      scripts=['scripts/idepi'],
      requires=[
            'Bio (>=1.58)',
            'BioExt (>=0.14.0)',
            'hppy (>=0.9.5)',
            'numpy (>=1.6)',
            'six',
            'sklearn (>=0.14.0)',
            'sklmrmr (>=0.1.1)',
      ]
     )
