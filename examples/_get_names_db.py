#!/usr/bin/python
# -*- coding: utf-8 -*-
# Morra project
#
# Copyright (C) 2019-present by Sergei Ternovykh
# License: BSD, see LICENSE for details
"""
Example: Evaluate partial Morra models.
"""
from corpuscula.utils import download_file

download_file(
    'https://github.com/fostroll/corpuscula/raw/master/data/names.pickle',
    log_msg='Downloading names.pickle...', overwrite=False
)
download_file(
    'https://github.com/fostroll/corpuscula/raw/master/data/surnames.pickle',
    log_msg='Downloading surnames.pickle...', overwrite=False
)
