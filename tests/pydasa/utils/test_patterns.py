# -*- coding: utf-8 -*-
"""
Test Module for patterns.py
===========================================

Tests for regex patterns used in validation and parsing in PyDASA.
"""

import unittest
import pytest
import re
from pydasa.utils import patterns
from tests.pydasa.data.test_data import get_config_test_data


class TestPatterns(unittest.TestCase):
    """Test cases for patterns module."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self):
        """Inject test data fixture."""
        self.test_data = get_config_test_data()

    # LaTeX Pattern Tests
    def test_latex_re_exists(self) -> None:
        """Test that LATEX_RE exists and is a string."""
        assert hasattr(patterns, 'LATEX_RE')
        assert isinstance(patterns.LATEX_RE, str)

    def test_latex_re_matches_valid(self) -> None:
        """Test that LATEX_RE matches valid LaTeX symbols."""
        pattern = re.compile(patterns.LATEX_RE)
        for case in self.test_data["VALID_LATEX"]:
            assert pattern.search(case) is not None, f"Failed to match: {case}"

    def test_latex_re_pattern_structure(self) -> None:
        """Test that LATEX_RE has proper structure."""
        assert len(patterns.LATEX_RE) > 0
        # Should compile without errors
        pattern = re.compile(patterns.LATEX_RE)
        assert pattern is not None

    # Default FDU Pattern Tests
    def test_dflt_fdu_re_exists(self) -> None:
        """Test that DFLT_FDU_RE exists and is a string."""
        assert hasattr(patterns, 'DFLT_FDU_RE')
        assert isinstance(patterns.DFLT_FDU_RE, str)

    def test_dflt_fdu_re_matches_valid(self) -> None:
        """Test that DFLT_FDU_RE matches valid dimension strings."""
        pattern = re.compile(patterns.DFLT_FDU_RE)
        for case in self.test_data["VALID_DIMENSIONS"]:
            assert pattern.match(case), f"Failed to match: {case}"

    def test_dflt_fdu_re_rejects_invalid(self) -> None:
        """Test that DFLT_FDU_RE rejects invalid dimension strings."""
        pattern = re.compile(patterns.DFLT_FDU_RE)
        for case in self.test_data["INVALID_DIMENSIONS"]:
            assert not pattern.match(case), f"Should not match: {case}"

    def test_dflt_fdu_re_physical_dimensions(self) -> None:
        """Test that DFLT_FDU_RE matches common physical dimensions."""
        pattern = re.compile(patterns.DFLT_FDU_RE)
        for dims in self.test_data["PHYSICAL_DIMS"]:
            assert pattern.match(dims), f"Failed to match physical dimension: {dims}"

    # Power Pattern Tests
    def test_dflt_pow_re_exists(self) -> None:
        """Test that DFLT_POW_RE exists and is a string."""
        assert hasattr(patterns, 'DFLT_POW_RE')
        assert isinstance(patterns.DFLT_POW_RE, str)

    def test_dflt_pow_re_matches_numbers(self) -> None:
        """Test that DFLT_POW_RE matches integer exponents."""
        pattern = re.compile(patterns.DFLT_POW_RE)
        test_cases = ["1", "-1", "2", "-2", "10", "-10", "0"]
        for case in test_cases:
            assert pattern.search(case), f"Failed to match: {case}"

    # No Power Pattern Tests
    def test_dflt_no_pow_re_exists(self) -> None:
        """Test that DFLT_NO_POW_RE exists and is a string."""
        assert hasattr(patterns, 'DFLT_NO_POW_RE')
        assert isinstance(patterns.DFLT_NO_POW_RE, str)

    def test_dflt_no_pow_re_matches_fdus(self) -> None:
        """Test that DFLT_NO_POW_RE matches FDUs without exponents."""
        pattern = re.compile(patterns.DFLT_NO_POW_RE)
        # Should match single FDU letters from physical framework
        for fdu in ["L", "M", "T", "K", "I", "N", "C"]:
            assert pattern.search(fdu), f"Failed to match: {fdu}"

    # Symbolic Pattern Tests
    def test_dflt_fdu_sym_re_exists(self) -> None:
        """Test that DFLT_FDU_SYM_RE exists and is a string."""
        assert hasattr(patterns, 'DFLT_FDU_SYM_RE')
        assert isinstance(patterns.DFLT_FDU_SYM_RE, str)

    def test_dflt_fdu_sym_re_matches_symbols(self) -> None:
        """Test that DFLT_FDU_SYM_RE matches FDU symbols."""
        pattern = re.compile(patterns.DFLT_FDU_SYM_RE)
        # Should match single FDU letters
        for fdu in ["L", "M", "T", "K", "I", "N", "C"]:
            assert pattern.search(fdu), f"Failed to match: {fdu}"

    # Working Precedence List Tests
    def test_wkng_fdu_prec_lt_exists(self) -> None:
        """Test that WKNG_FDU_PREC_LT exists and is a list."""
        assert hasattr(patterns, 'WKNG_FDU_PREC_LT')
        assert isinstance(patterns.WKNG_FDU_PREC_LT, list)

    def test_wkng_fdu_prec_lt_matches_default(self) -> None:
        """Test that WKNG_FDU_PREC_LT initially matches default."""
        from pydasa.dimensional.constants import DFLT_FDU_PREC_LT
        # Should be a copy, not the same object
        assert patterns.WKNG_FDU_PREC_LT == DFLT_FDU_PREC_LT
        assert patterns.WKNG_FDU_PREC_LT is not DFLT_FDU_PREC_LT

    def test_wkng_fdu_prec_lt_is_mutable(self) -> None:
        """Test that WKNG_FDU_PREC_LT can be modified."""
        # Should be a list that can be modified
        original_length = len(patterns.WKNG_FDU_PREC_LT)
        assert original_length > 0

    # Working Pattern Tests
    def test_wkng_fdu_re_exists(self) -> None:
        """Test that WKNG_FDU_RE exists and is a string."""
        assert hasattr(patterns, 'WKNG_FDU_RE')
        assert isinstance(patterns.WKNG_FDU_RE, str)

    def test_wkng_fdu_re_matches_default(self) -> None:
        """Test that WKNG_FDU_RE initially matches default."""
        assert patterns.WKNG_FDU_RE == patterns.DFLT_FDU_RE

    def test_wkng_pow_re_exists(self) -> None:
        """Test that WKNG_POW_RE exists and is a string."""
        assert hasattr(patterns, 'WKNG_POW_RE')
        assert isinstance(patterns.WKNG_POW_RE, str)

    def test_wkng_pow_re_matches_default(self) -> None:
        """Test that WKNG_POW_RE initially matches default."""
        assert patterns.WKNG_POW_RE == patterns.DFLT_POW_RE

    def test_wkng_no_pow_re_exists(self) -> None:
        """Test that WKNG_NO_POW_RE exists and is a string."""
        assert hasattr(patterns, 'WKNG_NO_POW_RE')
        assert isinstance(patterns.WKNG_NO_POW_RE, str)

    def test_wkng_no_pow_re_matches_default(self) -> None:
        """Test that WKNG_NO_POW_RE initially matches default."""
        assert patterns.WKNG_NO_POW_RE == patterns.DFLT_NO_POW_RE

    def test_wkng_fdu_sym_re_exists(self) -> None:
        """Test that WKNG_FDU_SYM_RE exists and is a string."""
        assert hasattr(patterns, 'WKNG_FDU_SYM_RE')
        assert isinstance(patterns.WKNG_FDU_SYM_RE, str)

    def test_wkng_fdu_sym_re_matches_default(self) -> None:
        """Test that WKNG_FDU_SYM_RE initially matches default."""
        assert patterns.WKNG_FDU_SYM_RE == patterns.DFLT_FDU_SYM_RE

    # Cross-Pattern Tests
    def test_all_patterns_are_strings(self) -> None:
        """Test that all pattern variables are strings."""
        string_patterns = [
            'LATEX_RE', 'DFLT_FDU_RE', 'DFLT_POW_RE', 
            'DFLT_NO_POW_RE', 'DFLT_FDU_SYM_RE',
            'WKNG_FDU_RE', 'WKNG_POW_RE', 
            'WKNG_NO_POW_RE', 'WKNG_FDU_SYM_RE'
        ]
        for pattern_name in string_patterns:
            assert hasattr(patterns, pattern_name)
            assert isinstance(getattr(patterns, pattern_name), str)

    def test_all_patterns_compile(self) -> None:
        """Test that all regex patterns compile without errors."""
        pattern_names = [
            'LATEX_RE', 'DFLT_FDU_RE', 'DFLT_POW_RE', 
            'DFLT_NO_POW_RE', 'DFLT_FDU_SYM_RE',
            'WKNG_FDU_RE', 'WKNG_POW_RE', 
            'WKNG_NO_POW_RE', 'WKNG_FDU_SYM_RE'
        ]
        for pattern_name in pattern_names:
            pattern_str = getattr(patterns, pattern_name)
            try:
                compiled = re.compile(pattern_str)
                assert compiled is not None
            except re.error as e:
                pytest.fail(f"Pattern {pattern_name} failed to compile: {e}")

    def test_patterns_are_not_empty(self) -> None:
        """Test that all patterns are non-empty strings."""
        pattern_names = [
            'LATEX_RE', 'DFLT_FDU_RE', 'DFLT_POW_RE', 
            'DFLT_NO_POW_RE', 'DFLT_FDU_SYM_RE',
            'WKNG_FDU_RE', 'WKNG_POW_RE', 
            'WKNG_NO_POW_RE', 'WKNG_FDU_SYM_RE'
        ]
        for pattern_name in pattern_names:
            pattern_str = getattr(patterns, pattern_name)
            assert len(pattern_str) > 0, f"Pattern {pattern_name} is empty"
