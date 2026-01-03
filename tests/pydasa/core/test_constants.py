# -*- coding: utf-8 -*-
"""
Test Module for core/constants.py
===========================================

Tests for default FDU framework and precedence list constants in PyDASA.

This module tests:
    - DFLT_FDU_FWK_DT: Default Fundamental Dimensional Units framework dictionary
    - DFLT_FDU_PREC_LT: Default FDU precedence list for dimensional matrix
"""

import unittest
import pytest
from pydasa.core import constants
from tests.pydasa.data.test_data import get_config_test_data

# Asserting module imports
assert constants
assert get_config_test_data


class TestConstants(unittest.TestCase):
    """Test cases for constants module.

    Tests DFLT_FDU_FWK_DT and DFLT_FDU_PREC_LT constants.
    """

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """Inject test data fixture."""
        self.test_data = get_config_test_data()

    def test_dflt_fdu_fwk_dt_exists(self) -> None:
        """Test that DFLT_FDU_FWK_DT constant exists and is a dict."""
        assert hasattr(constants, 'DFLT_FDU_FWK_DT')
        assert isinstance(constants.DFLT_FDU_FWK_DT, dict)

    def test_dflt_fdu_fwk_dt_is_mutable(self) -> None:
        """Test that DFLT_FDU_FWK_DT can be modified."""
        fdu_fwk = constants.DFLT_FDU_FWK_DT
        original_state = fdu_fwk.copy()

        # Add test item
        fdu_fwk["TEST"] = "Test Value"
        assert "TEST" in fdu_fwk

        # Cleanup
        fdu_fwk.clear()
        fdu_fwk.update(original_state)

    def test_dflt_fdu_prec_lt_exists(self) -> None:
        """Test that DFLT_FDU_PREC_LT constant exists and is a list."""
        assert hasattr(constants, 'DFLT_FDU_PREC_LT')
        assert isinstance(constants.DFLT_FDU_PREC_LT, list)

    def test_dflt_fdu_prec_lt_is_mutable(self) -> None:
        """Test that DFLT_FDU_PREC_LT can be modified."""
        fdu_prec = constants.DFLT_FDU_PREC_LT
        original_state = fdu_prec.copy()

        # Add test item
        fdu_prec.append("TEST")
        assert "TEST" in fdu_prec

        # Cleanup
        fdu_prec.clear()
        fdu_prec.extend(original_state)

    def test_prec_list_derived_from_framework_keys(self) -> None:
        """Test that DFLT_FDU_PREC_LT is derived from DFLT_FDU_FWK_DT keys."""
        fdu_fwk = constants.DFLT_FDU_FWK_DT
        fdu_prec = constants.DFLT_FDU_PREC_LT

        # Precedence list should match framework dict keys
        assert fdu_prec == list(fdu_fwk.keys())
        assert len(fdu_prec) == len(fdu_fwk)

    def test_constants_support_physical_framework(self) -> None:
        """Test that constants can be populated with physical FDUs."""
        fdu_fwk = constants.DFLT_FDU_FWK_DT
        original_state = fdu_fwk.copy()

        # Populate with physical FDUs
        physical_fdus = {"L": "Length", "M": "Mass", "T": "Time"}
        fdu_fwk.clear()
        fdu_fwk.update(physical_fdus)

        assert len(fdu_fwk) == len(physical_fdus)
        assert all(k in fdu_fwk for k in physical_fdus.keys())

        # Cleanup
        fdu_fwk.clear()
        fdu_fwk.update(original_state)
