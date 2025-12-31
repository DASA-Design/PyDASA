# -*- coding: utf-8 -*-
"""
Test Module for constants.py
===========================================

Tests for Fundamental Dimensional Unit (FDU) constants in PyDASA.
"""

import unittest
import pytest
from pydasa.dimensional import constants
from tests.pydasa.data.test_data import get_config_test_data


class TestConstants(unittest.TestCase):
    """Test cases for dimensional constants module."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self):
        """Inject test data fixture."""
        self.test_data = get_config_test_data()

    # Physical FDUs Tests
    def test_phy_fdu_exists(self) -> None:
        """Test that PHY_FDU_PREC_DT exists and is a dictionary."""
        assert hasattr(constants, 'PHY_FDU_PREC_DT')
        assert isinstance(constants.PHY_FDU_PREC_DT, dict)

    def test_phy_fdu_keys(self) -> None:
        """Test that PHY_FDU_PREC_DT has expected keys."""
        assert set(constants.PHY_FDU_PREC_DT.keys()) == set(self.test_data["PHYSICAL_KEYS"])

    def test_phy_fdu_structure(self) -> None:
        """Test that each physical FDU has required fields."""
        for fdu_key, fdu_data in constants.PHY_FDU_PREC_DT.items():
            assert isinstance(fdu_data, dict)
            for field in self.test_data["REQUIRED_FIELDS"]:
                assert field in fdu_data
                assert isinstance(fdu_data[field], str)

    def test_phy_fdu_units(self) -> None:
        """Test that physical FDUs have correct SI units."""
        for fdu_key, expected_unit in self.test_data["PHYSICAL_UNITS"].items():
            assert constants.PHY_FDU_PREC_DT[fdu_key]["_unit"] == expected_unit

    def test_phy_fdu_immutability(self) -> None:
        """Test that PHY_FDU_PREC_DT is not accidentally modified."""
        original_keys = set(constants.PHY_FDU_PREC_DT.keys())
        assert original_keys == set(self.test_data["PHYSICAL_KEYS"])

    # Computation FDUs Tests
    def test_compu_fdu_exists(self) -> None:
        """Test that COMPU_FDU_PREC_DT exists and is a dictionary."""
        assert hasattr(constants, 'COMPU_FDU_PREC_DT')
        assert isinstance(constants.COMPU_FDU_PREC_DT, dict)

    def test_compu_fdu_keys(self) -> None:
        """Test that COMPU_FDU_PREC_DT has expected keys."""
        assert set(constants.COMPU_FDU_PREC_DT.keys()) == set(self.test_data["COMPUTATION_KEYS"])

    def test_compu_fdu_structure(self) -> None:
        """Test that each computation FDU has required fields."""
        for fdu_key, fdu_data in constants.COMPU_FDU_PREC_DT.items():
            assert isinstance(fdu_data, dict)
            for field in self.test_data["REQUIRED_FIELDS"]:
                assert field in fdu_data
                assert isinstance(fdu_data[field], str)

    def test_compu_fdu_units(self) -> None:
        """Test that computation FDUs have correct units."""
        for fdu_key, expected_unit in self.test_data["COMPUTATION_UNITS"].items():
            assert constants.COMPU_FDU_PREC_DT[fdu_key]["_unit"] == expected_unit

    # Software FDUs Tests
    def test_soft_fdu_exists(self) -> None:
        """Test that SOFT_FDU_PREC_DT exists and is a dictionary."""
        assert hasattr(constants, 'SOFT_FDU_PREC_DT')
        assert isinstance(constants.SOFT_FDU_PREC_DT, dict)

    def test_soft_fdu_keys(self) -> None:
        """Test that SOFT_FDU_PREC_DT has expected keys."""
        assert set(constants.SOFT_FDU_PREC_DT.keys()) == set(self.test_data["SOFTWARE_KEYS"])

    def test_soft_fdu_structure(self) -> None:
        """Test that each software FDU has required fields."""
        for fdu_key, fdu_data in constants.SOFT_FDU_PREC_DT.items():
            assert isinstance(fdu_data, dict)
            for field in self.test_data["REQUIRED_FIELDS"]:
                assert field in fdu_data
                assert isinstance(fdu_data[field], str)

    def test_soft_fdu_units(self) -> None:
        """Test that software FDUs have correct units."""
        for fdu_key, expected_unit in self.test_data["SOFTWARE_UNITS"].items():
            assert constants.SOFT_FDU_PREC_DT[fdu_key]["_unit"] == expected_unit

    # Default Precedence List Tests
    def test_dflt_fdu_prec_lt_exists(self) -> None:
        """Test that DFLT_FDU_PREC_LT exists and is a list."""
        assert hasattr(constants, 'DFLT_FDU_PREC_LT')
        assert isinstance(constants.DFLT_FDU_PREC_LT, list)

    def test_dflt_fdu_prec_lt_order(self) -> None:
        """Test that DFLT_FDU_PREC_LT has expected order."""
        assert constants.DFLT_FDU_PREC_LT == self.test_data["PHYSICAL_KEYS"]

    def test_dflt_fdu_prec_lt_derived_from_phy(self) -> None:
        """Test that DFLT_FDU_PREC_LT is derived from PHY_FDU_PREC_DT."""
        assert constants.DFLT_FDU_PREC_LT == list(constants.PHY_FDU_PREC_DT.keys())

    def test_dflt_fdu_prec_lt_length(self) -> None:
        """Test that DFLT_FDU_PREC_LT has correct length."""
        assert len(constants.DFLT_FDU_PREC_LT) == len(self.test_data["PHYSICAL_KEYS"])

    # Cross-Framework Tests
    def test_all_fdus_have_time(self) -> None:
        """Test that all FDU frameworks include Time dimension."""
        assert "T" in constants.PHY_FDU_PREC_DT
        assert "T" in constants.COMPU_FDU_PREC_DT
        assert "T" in constants.SOFT_FDU_PREC_DT

    def test_fdu_uniqueness_within_framework(self) -> None:
        """Test that FDU keys are unique within each framework."""
        assert len(constants.PHY_FDU_PREC_DT.keys()) == len(set(constants.PHY_FDU_PREC_DT.keys()))
        assert len(constants.COMPU_FDU_PREC_DT.keys()) == len(set(constants.COMPU_FDU_PREC_DT.keys()))
        assert len(constants.SOFT_FDU_PREC_DT.keys()) == len(set(constants.SOFT_FDU_PREC_DT.keys()))

    def test_fdu_dictionaries_are_not_empty(self) -> None:
        """Test that all FDU dictionaries contain at least one entry."""
        assert len(constants.PHY_FDU_PREC_DT) > 0
        assert len(constants.COMPU_FDU_PREC_DT) > 0
        assert len(constants.SOFT_FDU_PREC_DT) > 0

    def test_fdu_descriptions_not_empty(self) -> None:
        """Test that all FDU descriptions are non-empty strings."""
        for fdu_dict in [constants.PHY_FDU_PREC_DT, constants.COMPU_FDU_PREC_DT, constants.SOFT_FDU_PREC_DT]:
            for fdu_key, fdu_data in fdu_dict.items():
                assert len(fdu_data["description"]) > 0
                assert len(fdu_data["name"]) > 0
                assert len(fdu_data["_unit"]) > 0
