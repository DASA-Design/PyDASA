# -*- coding: utf-8 -*-
"""
Test Module for patterns.py
===========================================

Tests for regex patterns used in validation and parsing in PyDASA.

This module tests:
    - LATEX_RE: LaTeX symbol pattern
    - DFLT_POW_RE: Power/exponent pattern

Note:
    FDU framework patterns have been moved to core/setup.py.
    See core/setup.py for framework-specific dimensional patterns.
"""

import unittest
import pytest
import re
from pydasa.validations import patterns as pat
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
        assert hasattr(pat, 'LATEX_RE')
        assert isinstance(pat.LATEX_RE, str)

    def test_latex_re_matches_valid(self) -> None:
        """Test that LATEX_RE matches valid LaTeX symbols."""
        pattern = re.compile(pat.LATEX_RE)
        for case in self.test_data["VALID_LATEX"]:
            assert pattern.search(case) is not None, f"Failed to match: {case}"

    def test_latex_re_pattern_structure(self) -> None:
        """Test that LATEX_RE has proper structure."""
        assert len(pat.LATEX_RE) > 0
        # Should compile without errors
        pattern = re.compile(pat.LATEX_RE)
        assert pattern is not None

    def test_latex_re_matches_alphanumeric(self) -> None:
        """Test that LATEX_RE matches simple alphanumeric symbols."""
        pattern = re.compile(pat.LATEX_RE)
        test_cases = ["x", "y", "abc", "V", "rho"]
        for case in test_cases:
            assert pattern.search(case), f"Failed to match: {case}"

    def test_latex_re_matches_backslash_commands(self) -> None:
        """Test that LATEX_RE matches LaTeX backslash commands."""
        pattern = re.compile(pat.LATEX_RE)
        test_cases = ["\\alpha", "\\beta", "\\gamma", "\\Pi", "\\mu"]
        for case in test_cases:
            assert pattern.search(case), f"Failed to match: {case}"

    def test_latex_re_matches_subscripts(self) -> None:
        """Test that LATEX_RE matches LaTeX symbols with subscripts."""
        pattern = re.compile(pat.LATEX_RE)
        test_cases = ["\\alpha_{1}", "\\beta_{12}", "x_{i}", "\\Pi_{0}"]
        for case in test_cases:
            assert pattern.search(case), f"Failed to match: {case}"

    # Power Pattern Tests
    def test_dflt_pow_re_exists(self) -> None:
        """Test that DFLT_POW_RE exists and is a string."""
        assert hasattr(pat, 'DFLT_POW_RE')
        assert isinstance(pat.DFLT_POW_RE, str)

    def test_dflt_pow_re_matches_positive(self) -> None:
        """Test that DFLT_POW_RE matches positive integers."""
        pattern = re.compile(pat.DFLT_POW_RE)
        test_cases = ["1", "2", "10", "100", "0"]
        for case in test_cases:
            assert pattern.search(case), f"Failed to match: {case}"

    def test_dflt_pow_re_matches_negative(self) -> None:
        """Test that DFLT_POW_RE matches negative integers."""
        pattern = re.compile(pat.DFLT_POW_RE)
        test_cases = ["-1", "-2", "-10", "-100"]
        for case in test_cases:
            assert pattern.search(case), f"Failed to match: {case}"

    def test_dflt_pow_re_compiles(self) -> None:
        """Test that DFLT_POW_RE compiles without errors."""
        try:
            pattern = re.compile(pat.DFLT_POW_RE)
            assert pattern is not None
        except re.error as e:
            pytest.fail(f"DFLT_POW_RE failed to compile: {e}")

    # Cross-Pattern Tests
    def test_all_patterns_exist(self) -> None:
        """Test that all expected pattern variables exist."""
        expected_patterns = ['LATEX_RE', 'DFLT_POW_RE']
        for pattern_name in expected_patterns:
            assert hasattr(pat, pattern_name), f"Missing pattern: {pattern_name}"

    def test_all_patterns_are_strings(self) -> None:
        """Test that all pattern variables are strings."""
        pattern_names = ['LATEX_RE', 'DFLT_POW_RE']
        for pattern_name in pattern_names:
            pattern_value = getattr(pat, pattern_name)
            assert isinstance(pattern_value, str), f"{pattern_name} is not a string"

    def test_all_patterns_compile(self) -> None:
        """Test that all regex patterns compile without errors."""
        pattern_names = ['LATEX_RE', 'DFLT_POW_RE']
        for pattern_name in pattern_names:
            pattern_str = getattr(pat, pattern_name)
            try:
                compiled = re.compile(pattern_str)
                assert compiled is not None
            except re.error as e:
                pytest.fail(f"Pattern {pattern_name} failed to compile: {e}")

    def test_patterns_are_not_empty(self) -> None:
        """Test that all patterns are non-empty strings."""
        pattern_names = ['LATEX_RE', 'DFLT_POW_RE']
        for pattern_name in pattern_names:
            pattern_str = getattr(pat, pattern_name)
            assert len(pattern_str) > 0, f"Pattern {pattern_name} is empty"

    def test_patterns_have_docstrings(self) -> None:
        # NOTE unnecessary but didnt know you could do this
        """Test that pattern module has proper documentation."""
        assert pat.__doc__ is not None
        assert len(pat.__doc__) > 0
