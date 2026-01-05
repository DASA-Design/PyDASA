# -*- coding: utf-8 -*-
"""
Test Module for core/constants.py
===========================================

Tests for configuration constants in PyDASA.

This module tests:
    - DFLT_CFG_FOLDER: Default configuration folder name
    - DFLT_CFG_FILE: Default configuration file name

Note:
    FDU framework configuration has been moved to core/setup.py.
    See core/setup.py for Framework, VarCardinality, CoefCardinality, and AnaliticMode enums.
"""

import unittest
from pydasa.core import constants

# Asserting module imports
assert constants


class TestConstants(unittest.TestCase):
    """Test cases for constants module.

    Tests configuration folder and file constants.
    """

    def test_dflt_cfg_folder_exists(self) -> None:
        """Test that DFLT_CFG_FOLDER constant exists and is a string."""
        assert hasattr(constants, 'DFLT_CFG_FOLDER')
        assert isinstance(constants.DFLT_CFG_FOLDER, str)
        assert constants.DFLT_CFG_FOLDER == "cfg"

    def test_dflt_cfg_file_exists(self) -> None:
        """Test that DFLT_CFG_FILE constant exists and is a string."""
        assert hasattr(constants, 'DFLT_CFG_FILE')
        assert isinstance(constants.DFLT_CFG_FILE, str)
        assert constants.DFLT_CFG_FILE == "default.json"

    def test_cfg_folder_is_immutable_string(self) -> None:
        """Test that DFLT_CFG_FOLDER is an immutable string constant."""
        original_value = constants.DFLT_CFG_FOLDER
        assert isinstance(original_value, str)
        assert len(original_value) > 0

    def test_cfg_file_is_immutable_string(self) -> None:
        """Test that DFLT_CFG_FILE is an immutable string constant."""
        original_value = constants.DFLT_CFG_FILE
        assert isinstance(original_value, str)
        assert original_value.endswith('.json')
