# -*- coding: utf-8 -*-
"""
Test Module for vashchy.py
===========================================

Tests for the **Coefficient** class in *PyDASA*.
"""

# import testing package
import unittest
import pytest
import numpy as np
from typing import Any, cast

# import the module to test
from pydasa.dimensional.buckingham import Coefficient
from pydasa.elements.parameter import Variable
from pydasa.dimensional.vaschy import Schema

# import the data to test
from tests.pydasa.data.test_data import get_coefficient_test_data

# asserting module imports
assert Coefficient
assert Variable
assert Schema
assert get_coefficient_test_data


class TestCoefficient(unittest.TestCase):
    """Test cases for Coefficient class."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """Inject test data fixture."""
        self.test_data = get_coefficient_test_data()

        # Setup dimensional scheme
        self.test_scheme = Schema(_fwk="PHYSICAL")

        # Create test variables from test data
        self.test_variables = {}
        for var_sym, var_data in self.test_data["TEST_VARIABLES"].items():
            self.test_variables[var_sym] = Variable(**var_data)

    # ========================================================================
    # Initialization tests
    # ========================================================================

    def test_default_initialization(self) -> None:
        """Test creating Coefficient with default values."""
        coef = Coefficient()

        assert coef is not None
        assert coef._idx == -1
        assert coef._sym.startswith("\\Pi_")
        assert coef._alias != ""
        assert coef._fwk == "PHYSICAL"
        assert coef._cat == "COMPUTED"
        assert coef.relevance is True
        assert isinstance(coef._variables, dict)
        assert len(coef._variables) == 0
        assert isinstance(coef._dim_col, list)
        assert isinstance(coef, Coefficient)

    def test_variables_from_fixture(self) -> None:
        """Test that fixture variables are properly created."""
        assert len(self.test_variables) == 6
        assert all(isinstance(var, Variable) for var in self.test_variables.values())

        # Check specific variables
        v_var = self.test_variables["v"]
        assert v_var._sym == "v"
        assert v_var._dims == "L*T^-1"
        assert v_var._fwk == "PHYSICAL"

        L_var = self.test_variables["L"]
        assert L_var._sym == "L"
        assert L_var._dims == "L"

        rho_var = self.test_variables["\\rho"]
        assert rho_var._sym == "\\rho"
        assert rho_var._dims == "M*L^-3"

    def test_coefficient_with_test_variables(self) -> None:
        """Test creating Coefficient with fixture test variables."""
        # Use subset of variables
        simple_vars = {
            "v": self.test_variables["v"],
            "L": self.test_variables["L"],
            "\\rho": self.test_variables["\\rho"],
            "\\mu": self.test_variables["\\mu"]
        }

        coef = Coefficient(
            _idx=0,
            _sym="\\Pi_{Re}",
            _alias="Pi_Re",
            _fwk="PHYSICAL",
            _cat="COMPUTED",
            _variables=simple_vars,
            _dim_col=[1, 1, 1, -1],  # v*L*rho/mu
            _name="Reynolds Number",
            description="Test coefficient with fixture variables"
        )

        assert coef._idx == 0
        assert coef._sym == "\\Pi_{Re}"
        assert len(coef._variables) == 4
        assert "v" in coef._variables
        assert "L" in coef._variables
        assert "\\rho" in coef._variables
        assert "\\mu" in coef._variables

    def test_coefficient_expression_building(self) -> None:
        """Test automatic expression building."""
        # Use subset of test_variables for simple expression
        simple_vars = {
            "v": self.test_variables["v"],
            "L": self.test_variables["L"]
        }

        coef = Coefficient(
            _variables=simple_vars,
            _dim_col=[1, -1]  # v/L
        )

        assert coef._pi_expr is not None
        assert isinstance(coef._pi_expr, str)
        assert coef.var_dims is not None
        assert isinstance(coef.var_dims, dict)
        assert coef.var_dims["v"] == 1
        assert coef.var_dims["L"] == -1

    # ========================================================================
    # Category property tests
    # ========================================================================

    def test_cat_getter(self) -> None:
        """Test cat property getter."""
        coef = Coefficient(_cat="DERIVED")
        assert coef.cat == "DERIVED"

    def test_cat_setter_valid(self) -> None:
        """Test cat property setter with valid values."""
        coef = Coefficient()
        for cat in self.test_data["VALID_CATEGORIES"]:
            coef.cat = cat
            assert coef.cat == cat.upper()

    def test_cat_setter_invalid(self) -> None:
        """Test cat property setter with invalid values."""
        coef = Coefficient()
        for invalid_cat in self.test_data["INVALID_CATEGORIES"]:
            with pytest.raises(ValueError) as excinfo:
                coef.cat = invalid_cat
            assert "Invalid cat" in str(excinfo.value)

    # ========================================================================
    # Variables property tests
    # ========================================================================

    def test_variables_getter(self) -> None:
        """Test variables property getter."""
        coef = Coefficient()
        assert isinstance(coef.variables, dict)
        assert len(coef.variables) == 0

    def test_variables_setter_valid(self) -> None:
        """Test variables property setter with valid dictionary."""
        coef = Coefficient()

        # Use test_variables fixture
        simple_vars = {
            "v": self.test_variables["v"],
            "L": self.test_variables["L"]
        }

        coef.variables = simple_vars
        assert len(coef.variables) == 2
        assert "v" in coef.variables
        assert "L" in coef.variables

    def test_variables_setter_invalid_type(self) -> None:
        """Test variables property setter with invalid type."""
        coef = Coefficient()
        with pytest.raises(ValueError) as excinfo:
            coef.variables = cast(Any, ["v", "L"])
        assert "must be dict" in str(excinfo.value)

    def test_variables_setter_invalid_keys(self) -> None:
        """Test variables property setter with invalid keys."""
        coef = Coefficient()
        with pytest.raises(ValueError) as excinfo:
            coef.variables = {123: self.test_variables["v"]}     # type: ignore
        assert "must contain" in str(excinfo.value)

    def test_variables_setter_invalid_values(self) -> None:
        """Test variables property setter with invalid values."""
        coef = Coefficient()
        with pytest.raises(ValueError) as excinfo:
            coef.variables = {"v": "not a variable"}     # type: ignore
        assert "must contain" in str(excinfo.value)

    # ========================================================================
    # Dimensional column property tests
    # ========================================================================

    def test_dim_col_getter(self) -> None:
        """Test dim_col property getter."""
        coef = Coefficient(_dim_col=[1, -1, 0])
        assert coef.dim_col == [1, -1, 0]

    def test_dim_col_setter_valid_ints(self) -> None:
        """Test dim_col property setter with valid integers."""
        coef = Coefficient()
        coef.dim_col = [1, -1, 2, 0]
        assert coef.dim_col == [1, -1, 2, 0]

    def test_dim_col_setter_valid_floats(self) -> None:
        """Test dim_col property setter with float values."""
        coef = Coefficient()
        coef.dim_col = [1.0, -1.0, 2.0, 0.0]
        assert coef.dim_col == [1, -1, 2, 0]  # Converted to ints

    def test_dim_col_setter_invalid_type(self) -> None:
        """Test dim_col property setter with invalid type."""
        coef = Coefficient()
        with pytest.raises(ValueError) as excinfo:
            coef.dim_col = cast(Any, "not a list")
        assert "must be list" in str(excinfo.value)

    def test_dim_col_setter_invalid_elements(self) -> None:
        """Test dim_col property setter with invalid elements."""
        coef = Coefficient()
        with pytest.raises(ValueError) as excinfo:
            coef.dim_col = [1, "two", 3]    # type: ignore
        assert "must contain" in str(excinfo.value)

    # ========================================================================
    # Pivot list property tests
    # ========================================================================

    def test_pivot_lt_getter(self) -> None:
        """Test pivot_lt property getter."""
        coef = Coefficient(_pivot_lt=[0, 1, 2])
        assert coef.pivot_lt == [0, 1, 2]

    def test_pivot_lt_setter_valid(self) -> None:
        """Test pivot_lt property setter with valid list."""
        coef = Coefficient()
        coef.pivot_lt = [0, 1, 2]
        assert coef.pivot_lt == [0, 1, 2]

    def test_pivot_lt_setter_none(self) -> None:
        """Test pivot_lt property setter with None."""
        coef = Coefficient()
        coef.pivot_lt = None    # type: ignore
        assert coef.pivot_lt is None

    # ========================================================================
    # Pi expression property tests
    # ========================================================================

    def test_pi_expr_getter(self) -> None:
        """Test pi_expr property getter."""
        coef = Coefficient()
        assert isinstance(coef.pi_expr, (str, type(None)))

    def test_pi_expr_setter_valid(self) -> None:
        """Test pi_expr property setter with valid string."""
        coef = Coefficient()
        coef.pi_expr = "\\frac{v*L}{\\nu}"
        assert coef.pi_expr == "\\frac{v*L}{\\nu}"

    def test_pi_expr_setter_invalid_type(self) -> None:
        """Test pi_expr property setter with invalid type."""
        coef = Coefficient()
        with pytest.raises(ValueError) as excinfo:
            coef.pi_expr = cast(Any, 123)
        assert "pi_expr must be str" in str(excinfo.value)

    # ========================================================================
    # Value range property tests
    # ========================================================================

    def test_min_setter_valid(self) -> None:
        """Test min property setter with valid value."""
        coef = Coefficient()
        coef.max = 10.0
        coef.min = 0.0
        assert coef.min == 0.0

    def test_min_setter_invalid_type(self) -> None:
        """Test min property setter with invalid type."""
        coef = Coefficient()
        with pytest.raises(ValueError) as excinfo:
            coef.min = cast(Any, "not a number")
        assert "min must be int or float" in str(excinfo.value)

    def test_min_max_relationship(self) -> None:
        """Test min and max relationship validation."""
        coef = Coefficient()
        coef.min = 0.0
        coef.max = 10.0

        with pytest.raises(ValueError) as excinfo:
            coef.min = 15.0
        assert "cannot be greater than maximum" in str(excinfo.value)

    def test_max_setter_valid(self) -> None:
        """Test max property setter with valid value."""
        coef = Coefficient()
        coef.min = 0.0
        coef.max = 10.0
        assert coef.max == 10.0

    def test_max_setter_invalid_type(self) -> None:
        """Test max property setter with invalid type."""
        coef = Coefficient()
        with pytest.raises(ValueError) as excinfo:
            coef.max = cast(Any, "not a number")
        assert "max must be int or float" in str(excinfo.value)

    def test_mean_setter_valid(self) -> None:
        """Test mean property setter with valid value."""
        coef = Coefficient()
        coef.min = 0.0
        coef.max = 10.0
        coef.mean = 5.0
        assert coef.mean == 5.0

    def test_mean_setter_out_of_range(self) -> None:
        """Test mean property setter with value outside range."""
        coef = Coefficient()
        coef.min = 0.0
        coef.max = 10.0

        with pytest.raises(ValueError) as excinfo:
            coef.mean = 15.0
        assert "cannot be greater than maximum" in str(excinfo.value)

    def test_step_setter_valid(self) -> None:
        """Test step property setter with valid value."""
        coef = Coefficient()
        coef.min = 0.0
        coef.max = 10.0
        coef.step = 0.1
        assert coef.step == 0.1

    def test_step_setter_zero(self) -> None:
        """Test step property setter with zero value."""
        coef = Coefficient()
        with pytest.raises(ValueError) as excinfo:
            coef.step = 0.0
        assert "cannot be zero" in str(excinfo.value)

    # ========================================================================
    # Data array property tests
    # ========================================================================

    def test_data_automatic_generation(self) -> None:
        """Test automatic data array generation."""
        coef = Coefficient()
        coef.min = 0.0
        coef.max = 10.0
        coef.step = 2.0

        expected_data = np.arange(0.0, 10.0, 2.0)
        assert np.array_equal(coef.data, expected_data)

    def test_data_manual_override(self) -> None:
        """Test that explicitly set data overrides automatic generation."""
        coef = Coefficient()
        coef.min = 0.0
        coef.max = 10.0
        coef.step = 2.0

        # Explicitly set different data
        custom_data = np.array([1.0, 3.0, 5.0])
        coef.data = custom_data

        # Should return custom data, not auto-generated
        assert np.array_equal(coef.data, custom_data)
        assert not np.array_equal(coef.data, np.arange(0.0, 10.0, 2.0))

    # ========================================================================
    # Expression building tests
    # ========================================================================

    def test_build_expression_simple(self) -> None:
        """Test building simple expression."""
        coef = Coefficient()
        var_lt = ["v", "L"]
        dim_col = [1, -1]

        expr, var_dims = coef._build_expression(var_lt, dim_col)

        assert isinstance(expr, str)
        assert isinstance(var_dims, dict)
        assert var_dims["v"] == 1
        assert var_dims["L"] == -1
        assert "\\frac" in expr

    def test_build_expression_complex(self) -> None:
        """Test building complex expression with multiple exponents."""
        coef = Coefficient()
        var_lt = ["v", "L", "\\rho", "\\mu"]
        dim_col = [1, 1, 1, -1]

        expr, var_dims = coef._build_expression(var_lt, dim_col)

        assert isinstance(expr, str)
        assert "\\frac" in expr
        assert len(var_dims) == 4
        assert var_dims["v"] == 1
        assert var_dims["L"] == 1
        assert var_dims["\\rho"] == 1
        assert var_dims["\\mu"] == -1

    def test_build_expression_numerator_only(self) -> None:
        """Test building expression with only numerator."""
        coef = Coefficient()
        var_lt = ["v", "L"]
        dim_col = [1, 2]

        expr, var_dims = coef._build_expression(var_lt, dim_col)

        assert isinstance(expr, str)
        assert "\\frac" not in expr
        assert "v*L^{2}" in expr

    def test_build_expression_with_zeros(self) -> None:
        """Test building expression with zero exponents."""
        coef = Coefficient()
        var_lt = ["v", "L", "t"]
        dim_col = [1, -1, 0]

        expr, var_dims = coef._build_expression(var_lt, dim_col)

        assert "t" not in var_dims
        assert len(var_dims) == 2

    def test_build_expression_invalid_length_mismatch(self) -> None:
        """Test building expression with mismatched lengths."""
        coef = Coefficient()
        var_lt = ["v", "L"]
        dim_col = [1, -1, 0]

        with pytest.raises(ValueError) as excinfo:
            coef._build_expression(var_lt, dim_col)
        assert "must be equal" in str(excinfo.value)

    # ========================================================================
    # Foundation method tests
    # ========================================================================

    def test_validate_sequence_valid_single_type(self) -> None:
        """Test _validate_sequence with valid single type."""
        coef = Coefficient()
        assert coef._validate_sequence([1, 2, 3], int) is True
        assert coef._validate_sequence([1.0, 2.0], float) is True
        assert coef._validate_sequence(["a", "b"], str) is True

    def test_validate_sequence_valid_multiple_types(self) -> None:
        """Test _validate_sequence with multiple valid types."""
        coef = Coefficient()
        assert coef._validate_sequence([1, 2.0, 3], (int, float)) is True

    def test_validate_sequence_invalid_not_sequence(self) -> None:
        """Test _validate_sequence with non-sequence."""
        coef = Coefficient()
        with pytest.raises(ValueError) as excinfo:
            coef._validate_sequence(cast(Any, "not a sequence"), int)
        # String is rejected with specific message
        assert "must be a list or tuple" in str(excinfo.value)

    def test_validate_sequence_invalid_empty(self) -> None:
        """Test _validate_sequence with empty sequence."""
        coef = Coefficient()
        with pytest.raises(ValueError) as excinfo:
            coef._validate_sequence([], int)
        assert "cannot be empty" in str(excinfo.value)

    def test_validate_sequence_invalid_wrong_types(self) -> None:
        """Test _validate_sequence with wrong element types."""
        coef = Coefficient()
        with pytest.raises(ValueError) as excinfo:
            coef._validate_sequence([1, "two", 3], int)
        assert "must contain" in str(excinfo.value)

    # ========================================================================
    # Utility methods tests
    # ========================================================================

    def test_clear_resets_all_attributes(self) -> None:
        """Test clear method resets all attributes."""
        simple_vars = {
            "v": self.test_variables["v"],
            "L": self.test_variables["L"]
        }

        coef = Coefficient(
            _idx=1,
            _sym="\\Pi_{1}",
            _variables=simple_vars,
            _dim_col=[1, -1],
            _name="Test")

        coef.clear()

        assert coef._idx == -1
        assert coef._sym == ""
        assert coef._alias == ""
        assert coef._fwk == "PHYSICAL"
        assert coef.name == ""
        assert coef._cat == "COMPUTED"
        assert len(coef._variables) == 0
        assert len(coef._dim_col) == 0
        assert coef.relevance is True

    def test_to_dict_structure(self) -> None:
        """Test to_dict method returns correct structure."""
        simple_vars = {
            "v": self.test_variables["v"],
            "L": self.test_variables["L"]
        }

        coef = Coefficient(
            _idx=1,
            _sym="\\Pi_{1}",
            _variables=simple_vars,
            _dim_col=[1, -1])

        result = coef.to_dict()

        assert isinstance(result, dict)
        assert "idx" in result
        assert "sym" in result
        assert "variables" in result
        assert "dim_col" in result

    def test_to_dict_removes_underscores(self) -> None:
        """Test to_dict removes leading underscores."""
        coef = Coefficient(_idx=5, _sym="\\Pi_{5}")
        result = coef.to_dict()

        assert "idx" in result
        assert "_idx" not in result
        assert result["idx"] == 5

    def test_from_dict_creates_instance(self) -> None:
        """Test from_dict method creates Coefficient instance."""
        simple_vars = {
            "v": self.test_variables["v"],
            "L": self.test_variables["L"]
        }

        dict_data = {
            "idx": 1,
            "sym": "\\Pi_{1}",
            "dim_col": [1, -1],
            "variables": simple_vars
        }

        coef = Coefficient.from_dict(dict_data)

        assert isinstance(coef, Coefficient)
        assert coef._idx == 1
        assert coef._sym == "\\Pi_{1}"

    def test_to_dict_from_dict_roundtrip(self) -> None:
        """Test round-trip conversion."""
        simple_vars = {
            "v": self.test_variables["v"],
            "L": self.test_variables["L"]
        }

        coef1 = Coefficient(
            _idx=1,
            _sym="\\Pi_{1}",
            _variables=simple_vars,
            _dim_col=[1, -1]
        )

        dict_data = coef1.to_dict()
        coef2 = Coefficient.from_dict(dict_data)

        assert coef1._idx == coef2._idx
        assert coef1._sym == coef2._sym
        assert coef1._dim_col == coef2._dim_col
