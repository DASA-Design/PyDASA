# -*- coding: utf-8 -*-
"""
Test Module for latex.py
===========================================

Tests for LaTeX parsing and conversion functions in PyDASA.
"""

import unittest
import pytest
from sympy import Symbol
from pydasa.serialization.parser import latex_to_python
from pydasa.serialization.parser import extract_latex_vars
from pydasa.serialization.parser import create_latex_mapping
from pydasa.serialization.parser import IGNORE_EXPR

from tests.pydasa.data.test_data import get_latex_test_data


# # Test data fixture
# @pytest.fixture(scope="module")
class TestIgnoreExpressions(unittest.TestCase):
    """Test cases for IGNORE_EXPR constant."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self):
        """Inject test data fixture."""
        self.test_data = get_latex_test_data()

    def test_ignore_expr_exists(self) -> None:
        """Test that IGNORE_EXPR exists and is a set."""
        assert IGNORE_EXPR is not None
        assert isinstance(IGNORE_EXPR, set)

    def test_ignore_expr_contains_functions(self) -> None:
        """Test that IGNORE_EXPR contains common LaTeX functions."""
        assert self.test_data["EXPECTED_FUNCTIONS"].issubset(IGNORE_EXPR)

    def test_ignore_expr_all_strings(self) -> None:
        """Test that all elements in IGNORE_EXPR are strings."""
        for expr in IGNORE_EXPR:
            assert isinstance(expr, str)
            assert expr.startswith("\\")


class TestLatexToPython(unittest.TestCase):
    """Test cases for latex_to_python() function."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self):
        """Inject test data fixture."""
        self.test_data = get_latex_test_data()

    def test_simple_alphanumeric(self) -> None:
        """Test conversion of simple alphanumeric strings."""
        for string in self.test_data["SIMPLE_STRINGS"]:
            assert latex_to_python(string) == string

    def test_latex_symbol_conversion(self) -> None:
        """Test conversion of LaTeX symbols."""
        for latex, python in self.test_data["LATEX_SYMBOLS"].items():
            assert latex_to_python(latex) == python

    def test_subscript_conversion(self) -> None:
        """Test conversion of LaTeX subscripts."""
        for latex, python in self.test_data["SUBSCRIPTS"].items():
            assert latex_to_python(latex) == python

    def test_nested_braces(self) -> None:
        """Test conversion with nested braces."""
        result = latex_to_python("\\alpha_{12}")
        assert result == "alpha_12"
        assert "_{" not in result
        assert "}" not in result

    def test_empty_string(self) -> None:
        """Test conversion of empty string."""
        assert latex_to_python("") == ""

    def test_greek_letters(self) -> None:
        """Test with various Greek letters."""
        for letter in self.test_data["GREEK_LETTERS"]:
            result = latex_to_python(letter)
            assert result == letter.lstrip("\\")

    def test_numbers_in_subscripts(self) -> None:
        """Test with numbers in subscripts."""
        numbers = ["0", "1", "2", "10", "123"]
        for num in numbers:
            expr = f"\\mu_{{{num}}}"
            result = latex_to_python(expr)
            assert result == f"mu_{num}"


class TestExtractLatexVars(unittest.TestCase):
    """Test cases for extract_latex_vars() function."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self):
        """Inject test data fixture."""
        self.test_data = get_latex_test_data()

    def test_simple_variable(self) -> None:
        """Test extraction of simple variable."""
        latex_to_py, py_to_latex = extract_latex_vars("\\alpha")

        assert "\\alpha" in latex_to_py
        assert latex_to_py["\\alpha"] == "alpha"
        assert "alpha" in py_to_latex
        assert py_to_latex["alpha"] == "\\alpha"

    def test_variable_with_subscript(self) -> None:
        """Test extraction of variable with subscript."""
        latex_to_py, py_to_latex = extract_latex_vars("\\mu_{1}")

        assert "\\mu_{1}" in latex_to_py
        assert latex_to_py["\\mu_{1}"] == "mu_1"
        assert "mu_1" in py_to_latex
        assert py_to_latex["mu_1"] == "\\mu_{1}"

    def test_multiple_variables(self) -> None:
        """Test extraction of multiple variables."""
        expr = self.test_data["COMPLEX_EXPR"]
        latex_to_py, py_to_latex = extract_latex_vars(expr)

        assert len(latex_to_py) == 3
        assert "\\alpha" in latex_to_py
        assert "\\beta_{1}" in latex_to_py
        assert "\\gamma_{2}" in latex_to_py

    def test_ignore_latex_functions(self) -> None:
        """Test that LaTeX functions are ignored."""
        expr = "\\frac{\\alpha}{\\beta} + \\sin(\\gamma)"
        latex_to_py, py_to_latex = extract_latex_vars(expr)

        assert "\\frac" not in latex_to_py
        assert "\\sin" not in latex_to_py
        assert "\\alpha" in latex_to_py
        assert "\\beta" in latex_to_py
        assert "\\gamma" in latex_to_py

    def test_complex_expression(self) -> None:
        """Test extraction from complex expression."""
        expr = "\\Pi_{0} * \\Pi_{1} / \\mu_{2}"
        latex_to_py, py_to_latex = extract_latex_vars(expr)

        assert len(latex_to_py) == 3
        assert latex_to_py["\\Pi_{0}"] == "Pi_0"
        assert latex_to_py["\\Pi_{1}"] == "Pi_1"
        assert latex_to_py["\\mu_{2}"] == "mu_2"

    def test_empty_expression(self) -> None:
        """Test extraction from empty expression."""
        latex_to_py, py_to_latex = extract_latex_vars("")

        assert len(latex_to_py) == 0
        assert len(py_to_latex) == 0

    def test_bidirectional_mapping(self) -> None:
        """Test that mappings are bidirectional."""
        expr = "\\Pi_{1} + \\mu_{2}"
        latex_to_py, py_to_latex = extract_latex_vars(expr)

        for latex_var, py_var in latex_to_py.items():
            assert py_to_latex[py_var] == latex_var

        for py_var, latex_var in py_to_latex.items():
            assert latex_to_py[latex_var] == py_var

    def test_whitespace_handling(self) -> None:
        """Test handling of whitespace."""
        expr_with_spaces = "\\alpha + \\beta"
        expr_no_spaces = "\\alpha+\\beta"

        result1 = extract_latex_vars(expr_with_spaces)
        result2 = extract_latex_vars(expr_no_spaces)

        assert len(result1[0]) == len(result2[0])


class TestCreateLatexMapping(unittest.TestCase):
    """Test cases for create_latex_mapping() function."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self):
        """Inject test data fixture."""
        self.test_data = get_latex_test_data()

    def test_simple_mapping(self) -> None:
        """Test creation of simple variable mapping."""
        expr = "\\alpha"
        symbol_map, py_symbol_map, latex_to_py, py_to_latex = create_latex_mapping(expr)

        assert len(symbol_map) >= 1
        assert len(py_symbol_map) >= 1
        assert "alpha" in py_symbol_map
        assert isinstance(py_symbol_map["alpha"], Symbol)

    def test_subscript_mapping(self) -> None:
        """Test creation of mapping with subscripts."""
        expr = "\\mu_{1} + \\mu_{2}"
        symbol_map, py_symbol_map, latex_to_py, py_to_latex = create_latex_mapping(expr)

        assert "mu_1" in py_symbol_map
        assert "mu_2" in py_symbol_map
        assert isinstance(py_symbol_map["mu_1"], Symbol)
        assert isinstance(py_symbol_map["mu_2"], Symbol)

    def test_complex_expression_mapping(self) -> None:
        """Test creation of mapping from complex expression."""
        expr = "\\frac{\\Pi_{0}}{\\Pi_{1}}"
        symbol_map, py_symbol_map, latex_to_py, py_to_latex = create_latex_mapping(expr)

        assert "Pi_0" in py_symbol_map or "\\Pi_{0}" in latex_to_py
        assert "Pi_1" in py_symbol_map or "\\Pi_{1}" in latex_to_py

    def test_symbol_map_contains_symbols(self) -> None:
        """Test that symbol_map contains sympy Symbol objects."""
        expr = "\\alpha + \\beta"
        symbol_map, py_symbol_map, latex_to_py, py_to_latex = create_latex_mapping(expr)

        for key, value in symbol_map.items():
            assert isinstance(key, Symbol)
            assert isinstance(value, Symbol)

    def test_py_symbol_map_string_keys(self) -> None:
        """Test that py_symbol_map has string keys."""
        expr = "\\gamma_{1} * \\delta_{2}"
        symbol_map, py_symbol_map, latex_to_py, py_to_latex = create_latex_mapping(expr)

        for key in py_symbol_map.keys():
            assert isinstance(key, str)

    def test_return_tuple_length(self) -> None:
        """Test that function returns tuple of 4 elements."""
        expr = "\\alpha"
        result = create_latex_mapping(expr)

        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_all_returns_are_dicts(self) -> None:
        """Test that all return values are dictionaries."""
        expr = "\\beta_{1}"
        symbol_map, py_symbol_map, latex_to_py, py_to_latex = create_latex_mapping(expr)

        assert isinstance(symbol_map, dict)
        assert isinstance(py_symbol_map, dict)
        assert isinstance(latex_to_py, dict)
        assert isinstance(py_to_latex, dict)


class TestRealWorldExpressions(unittest.TestCase):
    """Test cases with real-world dimensional analysis expressions."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self):
        """Inject test data fixture."""
        self.test_data = get_latex_test_data()

    def test_dimensional_expressions(self) -> None:
        """Test extraction from dimensional analysis expressions."""
        for expr, expected_vars in self.test_data["DIMENSIONAL_CASES"]:
            latex_to_py, py_to_latex = extract_latex_vars(expr)

            assert len(latex_to_py) > 0

            py_vars = list(latex_to_py.values())
            for expected_var in expected_vars:
                var_found = any(
                    expected_var in py_var or py_var in expected_var
                    for py_var in py_vars
                )
                assert var_found, f"Expected variable {expected_var} not found in {py_vars}"

    def test_fluid_dynamics_expression(self) -> None:
        """Test with fluid dynamics expression."""
        expr = self.test_data["PHYSICS_EXPR"]
        latex_to_py, py_to_latex = extract_latex_vars(expr)

        expected_count = 5
        assert len(latex_to_py) >= expected_count - 1

        py_vars = list(latex_to_py.values())
        assert any("U" in v for v in py_vars)
        assert any("d" in v for v in py_vars)

    def test_expression_with_powers(self) -> None:
        """Test expression with powers."""
        expr = "d^{2} * P^{3}"
        latex_to_py, py_to_latex = extract_latex_vars(expr)

        py_vars = list(latex_to_py.values())
        assert any("d" in v for v in py_vars)
        assert any("P" in v for v in py_vars)
