# -*- coding: utf-8 -*-
"""
Test Module for config.py
===========================================

Tests for configuration constants, dictionaries, and regex patterns in PyDASA.
"""

import unittest
import pytest
import re
from pydasa.utils import config
from pydasa.dimensional.framework import DimScheme
from tests.pydasa.data.test_data import get_config_test_data


# Test data fixture
# @pytest.fixture(scope="module")
class TestConfig(unittest.TestCase):
    """Test cases for config module."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self):
        """Inject test data fixture."""
        self.test_data = get_config_test_data()
        # resseting config due to singletton patterm fix later
        self.test_scheme = DimScheme(_fwk="PHYSICAL")
        self.test_scheme.update_global_config()

    # Physical FDUs Tests
    def test_phy_fdu_exists(self) -> None:
        """Test that PHY_FDU_PREC_DT exists and is a dictionary."""
        assert hasattr(config, 'PHY_FDU_PREC_DT')
        assert isinstance(config.PHY_FDU_PREC_DT, dict)

    def test_phy_fdu_keys(self) -> None:
        """Test that PHY_FDU_PREC_DT has expected keys."""
        assert set(config.PHY_FDU_PREC_DT.keys()) == set(self.test_data["PHYSICAL_KEYS"])

    def test_phy_fdu_structure(self) -> None:
        """Test that each physical FDU has required fields."""
        for fdu_key, fdu_data in config.PHY_FDU_PREC_DT.items():
            assert isinstance(fdu_data, dict)
            for field in self.test_data["REQUIRED_FIELDS"]:
                assert field in fdu_data
                assert isinstance(fdu_data[field], str)

    def test_phy_fdu_units(self) -> None:
        """Test that physical FDUs have correct SI units."""
        for fdu_key, expected_unit in self.test_data["PHYSICAL_UNITS"].items():
            assert config.PHY_FDU_PREC_DT[fdu_key]["_unit"] == expected_unit

    # Computation FDUs Tests
    def test_compu_fdu_exists(self) -> None:
        """Test that COMPU_FDU_PREC_DT exists and is a dictionary."""
        assert hasattr(config, 'COMPU_FDU_PREC_DT')
        assert isinstance(config.COMPU_FDU_PREC_DT, dict)

    def test_compu_fdu_keys(self) -> None:
        """Test that COMPU_FDU_PREC_DT has expected keys."""
        assert set(config.COMPU_FDU_PREC_DT.keys()) == set(self.test_data["COMPUTATION_KEYS"])

    def test_compu_fdu_units(self) -> None:
        """Test that computation FDUs have correct units."""
        for fdu_key, expected_unit in self.test_data["COMPUTATION_UNITS"].items():
            assert config.COMPU_FDU_PREC_DT[fdu_key]["_unit"] == expected_unit

    # Software FDUs Tests
    def test_soft_fdu_exists(self) -> None:
        """Test that SOFT_FDU_PREC_DT exists and is a dictionary."""
        assert hasattr(config, 'SOFT_FDU_PREC_DT')
        assert isinstance(config.SOFT_FDU_PREC_DT, dict)

    def test_soft_fdu_keys(self) -> None:
        """Test that SOFT_FDU_PREC_DT has expected keys."""
        assert set(config.SOFT_FDU_PREC_DT.keys()) == set(self.test_data["SOFTWARE_KEYS"])

    def test_soft_fdu_units(self) -> None:
        """Test that software FDUs have correct units."""
        for fdu_key, expected_unit in self.test_data["SOFTWARE_UNITS"].items():
            assert config.SOFT_FDU_PREC_DT[fdu_key]["_unit"] == expected_unit

    # Framework Dictionaries Tests
    def test_fdu_fwk_exists(self) -> None:
        """Test that FDU_FWK_DT exists and is a dictionary."""
        assert hasattr(config, 'FDU_FWK_DT')
        assert isinstance(config.FDU_FWK_DT, dict)

    def test_fdu_fwk_keys(self) -> None:
        """Test that FDU_FWK_DT has expected frameworks."""
        assert set(config.FDU_FWK_DT.keys()) == set(self.test_data["FRAMEWORK_KEYS"])

    def test_params_cat_exists(self) -> None:
        """Test that PARAMS_CAT_DT exists and has expected keys."""
        assert hasattr(config, 'PARAMS_CAT_DT')
        assert isinstance(config.PARAMS_CAT_DT, dict)
        assert set(config.PARAMS_CAT_DT.keys()) == set(self.test_data["PARAMS_CAT_KEYS"])

    def test_dc_cat_exists(self) -> None:
        """Test that DC_CAT_DT exists and has expected keys."""
        assert hasattr(config, 'DC_CAT_DT')
        assert isinstance(config.DC_CAT_DT, dict)
        assert set(config.DC_CAT_DT.keys()) == set(self.test_data["DC_CAT_KEYS"])

    def test_sens_ansys_exists(self) -> None:
        """Test that SENS_ANSYS_DT exists and has expected keys."""
        assert hasattr(config, 'SENS_ANSYS_DT')
        assert isinstance(config.SENS_ANSYS_DT, dict)
        assert set(config.SENS_ANSYS_DT.keys()) == set(self.test_data["SENS_ANSYS_KEYS"])

    # Precedence Lists Tests
    def test_dflt_fdu_prec_lt_exists(self) -> None:
        """Test that DFLT_FDU_PREC_LT exists and is a list."""
        assert hasattr(config, 'DFLT_FDU_PREC_LT')
        assert isinstance(config.DFLT_FDU_PREC_LT, list)

    def test_dflt_fdu_prec_lt_order(self) -> None:
        """Test that DFLT_FDU_PREC_LT has expected order."""
        assert config.DFLT_FDU_PREC_LT == self.test_data["PHYSICAL_KEYS"]

    def test_wkng_fdu_prec_lt_exists(self) -> None:
        """Test that WKNG_FDU_PREC_LT exists and matches default."""
        assert hasattr(config, 'WKNG_FDU_PREC_LT')
        assert isinstance(config.WKNG_FDU_PREC_LT, list)
        assert config.WKNG_FDU_PREC_LT == config.DFLT_FDU_PREC_LT

    # Regex Tests
    def test_latex_re_exists(self) -> None:
        """Test that LATEX_RE exists and is a string."""
        assert hasattr(config, 'LATEX_RE')
        assert isinstance(config.LATEX_RE, str)

    def test_latex_re_matches_valid(self) -> None:
        """Test that LATEX_RE matches valid LaTeX symbols."""
        pattern = re.compile(config.LATEX_RE)
        for case in self.test_data["VALID_LATEX"]:
            assert pattern.search(case) is not None

    def test_dflt_fdu_re_exists(self) -> None:
        """Test that DFLT_FDU_RE exists and is a string."""
        assert hasattr(config, 'DFLT_FDU_RE')
        assert isinstance(config.DFLT_FDU_RE, str)

    def test_dflt_fdu_re_matches_valid(self) -> None:
        """Test that DFLT_FDU_RE matches valid dimension strings."""
        pattern = re.compile(config.DFLT_FDU_RE)
        for case in self.test_data["VALID_DIMENSIONS"]:
            assert pattern.match(case)

    def test_dflt_fdu_re_rejects_invalid(self) -> None:
        """Test that DFLT_FDU_RE rejects invalid dimension strings."""
        pattern = re.compile(config.DFLT_FDU_RE)
        for case in self.test_data["INVALID_DIMENSIONS"]:
            assert not pattern.match(case)

    def test_dflt_pow_re_exists(self) -> None:
        """Test that DFLT_POW_RE exists and is a string."""
        assert hasattr(config, 'DFLT_POW_RE')
        assert isinstance(config.DFLT_POW_RE, str)

    def test_dflt_no_pow_re_exists(self) -> None:
        """Test that DFLT_NO_POW_RE exists and is a string."""
        assert hasattr(config, 'DFLT_NO_POW_RE')
        assert isinstance(config.DFLT_NO_POW_RE, str)

    def test_dflt_fdu_sym_re_exists(self) -> None:
        """Test that DFLT_FDU_SYM_RE exists and is a string."""
        assert hasattr(config, 'DFLT_FDU_SYM_RE')
        assert isinstance(config.DFLT_FDU_SYM_RE, str)

    def test_wkng_regex_matches_default(self) -> None:
        """Test that working regex patterns match defaults."""
        assert config.WKNG_FDU_RE == config.DFLT_FDU_RE
        assert config.WKNG_POW_RE == config.DFLT_POW_RE
        assert config.WKNG_NO_POW_RE == config.DFLT_NO_POW_RE
        assert config.WKNG_FDU_SYM_RE == config.DFLT_FDU_SYM_RE

    def test_physical_dimensions_match(self) -> None:
        """Test that common physical dimensions match regex."""
        pattern = re.compile(config.DFLT_FDU_RE)
        for dims in self.test_data["PHYSICAL_DIMS"]:
            assert pattern.match(dims)
