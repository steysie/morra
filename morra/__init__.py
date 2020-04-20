# -*- coding: utf-8 -*-
# Morra project
#
# Copyright (C) 2020-present by Sergei Ternovykh
# License: BSD, see LICENSE for details
"""
Morra is a part of the RuMor project. It provides tools for complete
morphological sentence parsing and Named-entity recognition.
"""
from morra._version import __version__
from morra.base_parser import autotrain
from morra.morph_parser import MorphParser
from morra.morph_parser2 import MorphParser2
from morra.morph_parser3 import MorphParser3
from morra.morph_parser_ne import MorphParserNE
