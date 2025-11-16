# -*- coding: utf-8 -*-
"""
Test Module for config.py
===========================================

Tests for configuration constants, dictionaries, and regex patterns in PyDASA.
"""

import pytest
import re
from pydasa.utils import config


class TestPhysicalFDUs:
    """Test Physical FDUs Precedence Dictionary.
    """

    def test_phy_fdu_prec_dt_exists(self) -> None:
        """Test that PHY_FDU_PREC_DT exists and is a dictionary.
        """
        assert hasattr(config, 'PHY_FDU_PREC_DT')
        assert isinstance(config.PHY_FDU_PREC_DT, dict)

    def test_phy_fdu_prec_dt_keys(self) -> None:
        """Test that PHY_FDU_PREC_DT has expected keys.
        """
        expected_keys = ["L", "M", "T", "K", "I", "N", "C"]
        assert set(config.PHY_FDU_PREC_DT.keys()) == set(expected_keys)

    def test_phy_fdu_prec_dt_structure(self) -> None:
        """Test that each FDU has required fields.
        """
        required_fields = ["_unit", "name", "description"]
        for fdu_key, fdu_data in config.PHY_FDU_PREC_DT.items():
            assert isinstance(fdu_data, dict), f"FDU {fdu_key} is not a dict"
            for field in required_fields:
                assert field in fdu_data, f"FDU {fdu_key} missing field {field}"
                assert isinstance(fdu_data[field], str), f"FDU {fdu_key}.{field} is not a string"

    def test_phy_fdu_units(self) -> None:
        """Test that physical FDUs have correct SI units.
        """
        expected_units = {
            "L": "m",
            "M": "kg",
            "T": "s",
            "K": "K",
            "I": "A",
            "N": "mol",
            "C": "cd"
        }
        for fdu_key, expected_unit in expected_units.items():
            assert config.PHY_FDU_PREC_DT[fdu_key]["_unit"] == expected_unit


class TestComputationFDUs:
    """Test Computation FDUs Precedence Dictionary.
    """

    def test_compu_fdu_prec_dt_exists(self) -> None:
        """Test that COMPU_FDU_PREC_DT exists and is a dictionary.
        """
        assert hasattr(config, 'COMPU_FDU_PREC_DT')
        assert isinstance(config.COMPU_FDU_PREC_DT, dict)

    def test_compu_fdu_prec_dt_keys(self) -> None:
        """Test that COMPU_FDU_PREC_DT has expected keys.
        """
        expected_keys = ["T", "S", "N"]
        assert set(config.COMPU_FDU_PREC_DT.keys()) == set(expected_keys)

    def test_compu_fdu_prec_dt_structure(self) -> None:
        """Test that each computation FDU has required fields.
        """
        required_fields = ["_unit", "name", "description"]
        for fdu_key, fdu_data in config.COMPU_FDU_PREC_DT.items():
            assert isinstance(fdu_data, dict)
            for field in required_fields:
                assert field in fdu_data
                assert isinstance(fdu_data[field], str)

    def test_compu_fdu_units(self) -> None:
        """Test that computation FDUs have correct units.
        """
        expected_units = {
            "T": "s",
            "S": "bit",
            "N": "op"
        }
        for fdu_key, expected_unit in expected_units.items():
            assert config.COMPU_FDU_PREC_DT[fdu_key]["_unit"] == expected_unit


class TestSoftwareFDUs:
    """Test Software Architecture FDUs Precedence Dictionary.
    """

    def test_soft_fdu_prec_dt_exists(self) -> None:
        """Test that SOFT_FDU_PREC_DT exists and is a dictionary.
        """
        assert hasattr(config, 'SOFT_FDU_PREC_DT')
        assert isinstance(config.SOFT_FDU_PREC_DT, dict)

    def test_soft_fdu_prec_dt_keys(self) -> None:
        """Test that SOFT_FDU_PREC_DT has expected keys.
        """
        expected_keys = ["T", "D", "E", "C", "A"]
        assert set(config.SOFT_FDU_PREC_DT.keys()) == set(expected_keys)

    def test_soft_fdu_prec_dt_structure(self) -> None:
        """Test that each software FDU has required fields.
        """
        required_fields = ["_unit", "name", "description"]
        for fdu_key, fdu_data in config.SOFT_FDU_PREC_DT.items():
            assert isinstance(fdu_data, dict)
            for field in required_fields:
                assert field in fdu_data
                assert isinstance(fdu_data[field], str)

    def test_soft_fdu_units(self) -> None:
        """Test that software FDUs have correct units.
        """
        expected_units = {
            "T": "s",
            "D": "bit",
            "E": "req",
            "C": "node",
            "A": "process"
        }
        for fdu_key, expected_unit in expected_units.items():
            assert config.SOFT_FDU_PREC_DT[fdu_key]["_unit"] == expected_unit


class TestFrameworkDictionaries:
    """Test framework and category dictionaries.
    """

    def test_fdu_fwk_dt_exists(self) -> None:
        """Test that FDU_FWK_DT exists and is a dictionary.
        """
        assert hasattr(config, 'FDU_FWK_DT')
        assert isinstance(config.FDU_FWK_DT, dict)

    def test_fdu_fwk_dt_keys(self) -> None:
        """Test that FDU_FWK_DT has expected frameworks.
        """
        expected_keys = ["PHYSICAL", "COMPUTATION", "SOFTWARE", "CUSTOM"]
        assert set(config.FDU_FWK_DT.keys()) == set(expected_keys)

    def test_fdu_fwk_dt_values(self) -> None:
        """Test that FDU_FWK_DT values are strings.
        """
        for key, value in config.FDU_FWK_DT.items():
            assert isinstance(value, str)
            assert len(value) > 0

    def test_params_cat_dt_exists(self) -> None:
        """Test that PARAMS_CAT_DT exists and is a dictionary.
        """
        assert hasattr(config, 'PARAMS_CAT_DT')
        assert isinstance(config.PARAMS_CAT_DT, dict)

    def test_params_cat_dt_keys(self) -> None:
        """Test that PARAMS_CAT_DT has expected categories.
        """
        expected_keys = ["IN", "OUT", "CTRL"]
        assert set(config.PARAMS_CAT_DT.keys()) == set(expected_keys)

    def test_dc_cat_dt_exists(self) -> None:
        """Test that DC_CAT_DT exists and is a dictionary.
        """
        assert hasattr(config, 'DC_CAT_DT')
        assert isinstance(config.DC_CAT_DT, dict)

    def test_dc_cat_dt_keys(self) -> None:
        """Test that DC_CAT_DT has expected categories.
        """
        expected_keys = ["COMPUTED", "DERIVED"]
        assert set(config.DC_CAT_DT.keys()) == set(expected_keys)

    def test_sens_ansys_dt_exists(self) -> None:
        """Test that SENS_ANSYS_DT exists and is a dictionary.
        """
        assert hasattr(config, 'SENS_ANSYS_DT')
        assert isinstance(config.SENS_ANSYS_DT, dict)

    def test_sens_ansys_dt_keys(self) -> None:
        """Test that SENS_ANSYS_DT has expected categories.
        """
        expected_keys = ["SYM", "NUM"]
        assert set(config.SENS_ANSYS_DT.keys()) == set(expected_keys)


class TestDefaultPrecedenceLists:
    """Test default FDU precedence lists.
    """

    def test_dflt_fdu_prec_lt_exists(self) -> None:
        """Test that DFLT_FDU_PREC_LT exists and is a list.
        """
        assert hasattr(config, 'DFLT_FDU_PREC_LT')
        assert isinstance(config.DFLT_FDU_PREC_LT, list)

    def test_dflt_fdu_prec_lt_matches_physical(self) -> None:
        """Test that DFLT_FDU_PREC_LT matches PHY_FDU_PREC_DT keys.
        """
        dflt_keys = set(config.DFLT_FDU_PREC_LT)
        phy_keys = set(config.PHY_FDU_PREC_DT.keys())
        assert dflt_keys == phy_keys

    def test_dflt_fdu_prec_lt_order(self) -> None:
        """Test that DFLT_FDU_PREC_LT has expected order.
        """
        expected_order = ["L", "M", "T", "K", "I", "N", "C"]
        assert config.DFLT_FDU_PREC_LT == expected_order

    def test_wkng_fdu_prec_lt_exists(self) -> None:
        """Test that WKNG_FDU_PREC_LT exists and is a list.
        """
        assert hasattr(config, 'WKNG_FDU_PREC_LT')
        assert isinstance(config.WKNG_FDU_PREC_LT, list)

    def test_wkng_fdu_prec_lt_is_copy(self) -> None:
        """Test that WKNG_FDU_PREC_LT is initially a copy of DFLT_FDU_PREC_LT.
        """
        assert config.WKNG_FDU_PREC_LT == config.DFLT_FDU_PREC_LT


class TestRegexPatterns:
    """Test regex patterns for FDU validation.
    """

    def test_latex_re_exists(self) -> None:
        """Test that LATEX_RE exists and is a string.
        """
        assert hasattr(config, 'LATEX_RE')
        assert isinstance(config.LATEX_RE, str)

    def test_latex_re_matches_valid_latex(self) -> None:
        """Test that LATEX_RE matches valid LaTeX symbols.
        """
        pattern = re.compile(config.LATEX_RE)
        valid_cases = [
            "alpha",
            "\\alpha",
            "beta_1",
            "\\beta_{1}",
            "gamma",
            "\\Pi_{0}"
        ]
        for case in valid_cases:
            match = pattern.search(case)
            assert match is not None, f"Failed to match: {case}"

    def test_dflt_fdu_re_exists(self) -> None:
        """Test that DFLT_FDU_RE exists and is a string.
        """
        assert hasattr(config, 'DFLT_FDU_RE')
        assert isinstance(config.DFLT_FDU_RE, str)

    def test_dflt_fdu_re_matches_valid_dimensions(self) -> None:
        """Test that DFLT_FDU_RE matches valid dimension strings.
        """
        pattern = re.compile(config.DFLT_FDU_RE)
        valid_cases = [
            "M",
            "L*T",
            "M*L^-1*T^-2",
            "L^2*T^-1",
            "M*L^2*T^-3",
            "T^-1"
        ]
        for case in valid_cases:
            assert pattern.match(case), f"Failed to match: {case}"

    def test_dflt_fdu_re_rejects_invalid_dimensions(self) -> None:
        """Test that DFLT_FDU_RE rejects invalid dimension strings.
        """
        pattern = re.compile(config.DFLT_FDU_RE)
        invalid_cases = [
            "X",           # Invalid symbol
            "M*X",         # Invalid symbol
            "M**2",        # Wrong exponent syntax
            "M^2.5",       # Decimal exponent
            "M L",         # Missing asterisk
            "",            # Empty string
        ]
        for case in invalid_cases:
            assert not pattern.match(case), f"Incorrectly matched: {case}"

    def test_dflt_pow_re_exists(self) -> None:
        """Test that DFLT_POW_RE exists and is a string.
        """
        assert hasattr(config, 'DFLT_POW_RE')
        assert isinstance(config.DFLT_POW_RE, str)

    def test_dflt_pow_re_matches_exponents(self) -> None:
        """Test that DFLT_POW_RE matches exponent patterns.
        """
        pattern = re.compile(config.DFLT_POW_RE)
        valid_cases = ["1", "2", "-1", "-2", "10", "-10"]
        for case in valid_cases:
            assert pattern.search(case), f"Failed to match: {case}"

    def test_dflt_no_pow_re_exists(self) -> None:
        """Test that DFLT_NO_POW_RE exists and is a string.
        """
        assert hasattr(config, 'DFLT_NO_POW_RE')
        assert isinstance(config.DFLT_NO_POW_RE, str)

    def test_dflt_no_pow_re_matches_no_exponent(self) -> None:
        """Test that DFLT_NO_POW_RE matches FDUs without exponents.
        """
        pattern = re.compile(config.DFLT_NO_POW_RE)
        test_string = "M*L*T"
        matches = pattern.findall(test_string)
        assert len(matches) == 3, f"Expected 3 matches, got {len(matches)}"
        assert set(matches) == {"M", "L", "T"}

    def test_dflt_fdu_sym_re_exists(self) -> None:
        """Test that DFLT_FDU_SYM_RE exists and is a string.
        """
        assert hasattr(config, 'DFLT_FDU_SYM_RE')
        assert isinstance(config.DFLT_FDU_SYM_RE, str)

    def test_dflt_fdu_sym_re_matches_symbols(self) -> None:
        """Test that DFLT_FDU_SYM_RE matches FDU symbols.
        """
        pattern = re.compile(config.DFLT_FDU_SYM_RE)
        test_string = "M*L*T^-2"
        matches = pattern.findall(test_string)
        assert "M" in matches
        assert "L" in matches
        assert "T" in matches


class TestWorkingRegexPatterns:
    """Test working regex patterns.
    """

    def test_wkng_fdu_re_exists(self) -> None:
        """Test that WKNG_FDU_RE exists and is a string.
        """
        assert hasattr(config, 'WKNG_FDU_RE')
        assert isinstance(config.WKNG_FDU_RE, str)

    def test_wkng_fdu_re_initial_value(self) -> None:
        """Test that WKNG_FDU_RE initially equals DFLT_FDU_RE.
        """
        assert config.WKNG_FDU_RE == config.DFLT_FDU_RE

    def test_wkng_pow_re_exists(self) -> None:
        """Test that WKNG_POW_RE exists and is a string.
        """
        assert hasattr(config, 'WKNG_POW_RE')
        assert isinstance(config.WKNG_POW_RE, str)

    def test_wkng_pow_re_initial_value(self) -> None:
        """Test that WKNG_POW_RE initially equals DFLT_POW_RE.
        """
        assert config.WKNG_POW_RE == config.DFLT_POW_RE

    def test_wkng_no_pow_re_exists(self) -> None:
        """Test that WKNG_NO_POW_RE exists and is a string.
        """
        assert hasattr(config, 'WKNG_NO_POW_RE')
        assert isinstance(config.WKNG_NO_POW_RE, str)

    def test_wkng_no_pow_re_initial_value(self) -> None:
        """Test that WKNG_NO_POW_RE initially equals DFLT_NO_POW_RE.
        """
        assert config.WKNG_NO_POW_RE == config.DFLT_NO_POW_RE

    def test_wkng_fdu_sym_re_exists(self) -> None:
        """Test that WKNG_FDU_SYM_RE exists and is a string.
        """
        assert hasattr(config, 'WKNG_FDU_SYM_RE')
        assert isinstance(config.WKNG_FDU_SYM_RE, str)

    def test_wkng_fdu_sym_re_initial_value(self) -> None:
        """Test that WKNG_FDU_SYM_RE initially equals DFLT_FDU_SYM_RE.
        """
        assert config.WKNG_FDU_SYM_RE == config.DFLT_FDU_SYM_RE


class TestComplexDimensionExamples:
    """Test regex patterns with complex real-world dimension examples.
    """

    @pytest.mark.parametrize("dims", [
        "M*L^-1*T^-2",      # Pressure
        "M*L^2*T^-2",       # Energy
        "M*L^2*T^-3",       # Power
        "L*T^-1",           # Velocity
        "L*T^-2",           # Acceleration
        "M*L^-3",           # Density
        "M*T^-1",           # Mass flow rate
        "L^3*T^-1",         # Volumetric flow rate
    ])
    def test_physical_dimensions(self, dims) -> None:
        """Test that common physical dimensions match the regex.
        """
        pattern = re.compile(config.DFLT_FDU_RE)
        assert pattern.match(dims), f"Failed to match: {dims}"

    @pytest.mark.parametrize("invld_dims", [
        "M*X^2",           # Invalid symbol X
        "M L",             # Missing asterisk
        "M^",              # Incomplete exponent
        "M*L^2.5",         # Decimal exponent
        "M**2",            # Wrong exponent syntax
    ])
    def test_invalid_dimensions(self, invld_dims) -> None:
        """Test that invalid dimensions don't match the regex.
        """
        pattern = re.compile(config.DFLT_FDU_RE)
        assert not pattern.match(invld_dims), f"invalid matches: {invld_dims}"
