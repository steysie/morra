# -*- coding: utf-8 -*-
# Morra project
#
# Copyright (C) 2019-present by Sergei Ternovykh
# License: BSD, see LICENSE for details
"""
Morra is a part of the RuMor project. It provides tools for complete
morphological sentence parsing and Named-entity recognition.
"""
from .base_parser import autotrain
from .morph_parser import MorphParser
from .morph_parser2 import MorphParser2
from .morph_parser3 import MorphParser3
from .morph_parser_ne import MorphParserNE
