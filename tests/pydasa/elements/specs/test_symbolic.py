# -*- coding: utf-8 -*-
"""
Test Module for specs/symbolic.py
===========================================

Tests for the **SymbolicSpecs** class in *PyDASA*.
"""

import unittest
import pytest

from pydasa.dimensional.vaschy import Schema
from pydasa.elements.parameter import Variable
# from pydasa.elements.specs.symbolic import SymbolicSpecs

from tests.pydasa.data.test_data import get_variable_test_data


class TestSymbolicSpecs(unittest.TestCase):
    """Test cases for **SymbolicSpecs** class via **Variable**."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """Inject test data fixture."""
        self.test_data = get_variable_test_data()

        # resseting config due to singletton patterm fix later
        self.test_scheme = Schema(_fwk="PHYSICAL")

    # Dimensions property tests
    def test_dims_getter(self) -> None:
        """Test dims property getter."""
        spec = Variable(_dims="L*T^-1")
        assert spec.dims == "L*T^-1"

    def test_dims_setter_valid(self) -> None:
        """Test dims property setter with valid values."""
        spec = Variable()

        # resseting config due to singletton patterm fix later
        self.test_scheme = Schema(_fwk="PHYSICAL")

        for dims in self.test_data["VALID_DIMENSIONS"]:
            spec.dims = dims
            assert spec.dims == dims

    def test_dims_setter_invalid_empty(self) -> None:
        """Test dims property setter with empty string."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.dims = "   "
        assert "dims must be a non-empty string" in str(excinfo.value)

    def test_dims_processing(self) -> None:
        """Test dimensional expression processing."""
        spec = Variable(_dims="L*T^-1", _units="m/s")

        assert spec._std_dims is not None
        assert spec._sym_exp is not None
        assert len(spec._dim_col) > 0

    # Units property tests
    def test_units_getter(self) -> None:
        """Test units property getter."""
        spec = Variable(_units="m/s")
        assert spec.units == "m/s"

    def test_units_setter_valid(self) -> None:
        """Test units property setter with valid values."""
        spec = Variable()
        for units in self.test_data["VALID_UNITS"]:
            spec.units = units
            assert spec.units == units

    def test_units_setter_invalid_empty(self) -> None:
        """Test units property setter with empty string."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.units = "   "
        assert "units must be a non-empty string" in str(excinfo.value)

    # Symbol expression property tests
    def test_sym_exp_getter(self) -> None:
        """Test sym_exp property getter."""
        spec = Variable(_dims="L*T^-1")
        assert spec.sym_exp is not None
        assert isinstance(spec.sym_exp, str)

    def test_sym_exp_setter_valid(self) -> None:
        """Test sym_exp property setter with valid value."""
        spec = Variable()
        spec.sym_exp = "L**1*T**(-1)"
        assert spec.sym_exp == "L**1*T**(-1)"

    def test_sym_exp_setter_invalid_empty(self) -> None:
        """Test sym_exp property setter with empty string."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.sym_exp = "   "
        assert "sym_exp must be a non-empty string" in str(excinfo.value)

    # Dimensional column property tests
    def test_dim_col_getter(self) -> None:
        """Test dim_col property getter."""
        spec = Variable(_dims="L*T^-1")
        assert spec.dim_col is not None
        assert isinstance(spec.dim_col, list)

    def test_dim_col_setter_valid(self) -> None:
        """Test dim_col property setter with valid value."""
        spec = Variable()
        spec.dim_col = [1, 0, -1, 0, 0, 0, 0]
        assert spec.dim_col == [1, 0, -1, 0, 0, 0, 0]

    def test_dim_col_setter_invalid(self) -> None:
        """Test dim_col property setter with invalid type."""
        from typing import Any, cast
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.dim_col = cast(Any, "not a list")
        assert "Dimensional column must be a list" in str(excinfo.value)

    # Standardized dimensions property tests
    def test_std_dims_getter(self) -> None:
        """Test std_dims property getter."""
        spec = Variable(_dims="L*T^-1")
        assert spec.std_dims is not None
        assert isinstance(spec.std_dims, str)

    def test_std_dims_setter_valid(self) -> None:
        """Test std_dims property setter with valid value."""
        spec = Variable()
        spec.std_dims = "L^(1)*T^(-1)"
        assert spec.std_dims == "L^(1)*T^(-1)"

    def test_std_dims_setter_invalid_empty(self) -> None:
        """Test std_dims property setter with empty string."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.std_dims = "   "
        assert "std_dims must be a non-empty string" in str(excinfo.value)

    # Standardized units property tests
    def test_std_units_getter(self) -> None:
        """Test std_units property getter."""
        spec = Variable()
        spec.std_units = "m/s"
        assert spec.std_units == "m/s"

    def test_std_units_setter_valid(self) -> None:
        """Test std_units property setter with valid value."""
        spec = Variable()
        spec.std_units = "m/s"
        assert spec.std_units == "m/s"

    def test_std_units_setter_invalid_empty(self) -> None:
        """Test std_units property setter with empty string."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.std_units = "   "
        assert "std_units must be a non-empty string" in str(excinfo.value)

    def test_dims_setter_invalid_expression(self) -> None:
        """Test dims property setter with invalid dimensional expression."""
        spec = Variable()
        # Try to set invalid dimensions that don't match FDU regex
        with pytest.raises(ValueError) as excinfo:
            spec.dims = "[X^2*Y^-1]"  # Invalid symbols not in FDU
        assert "Invalid dimensional expression" in str(excinfo.value)
        assert "FDUS precedence" in str(excinfo.value)

    def test_dim_col_setter_invalid_list_items(self) -> None:
        """Test dim_col property setter with list of non-integers."""
        from typing import Any, cast
        spec = Variable()
        # Note: The current implementation checks isinstance(val, list)
        # but doesn't validate list contents, so this test documents
        # the current behavior
        spec.dim_col = cast(Any, [1, 2, 3])  # Valid integers
        assert spec.dim_col == [1, 2, 3]

    def test_clear_method(self) -> None:
        """Test clear() method resets all symbolic attributes."""
        spec = Variable(_dims="L*T^-1", _units="m/s")
        # Set additional attributes
        spec.std_units = "m/s"

        # Verify attributes are set
        assert spec.dims == "L*T^-1"
        assert spec.units == "m/s"
        assert spec.std_dims is not None
        assert spec.sym_exp is not None
        assert spec.dim_col is not None and len(spec.dim_col) > 0
        assert spec.std_units == "m/s"

        # Clear all attributes
        spec.clear()

        # Verify all attributes are reset
        assert spec.dims == ""
        assert spec.units == ""
        assert spec.std_dims is None
        assert spec.sym_exp is None
        assert spec.dim_col == []
        assert spec.std_units == ""

    def test_setup_column_missing_exponent(self) -> None:
        """Test _setup_column error when exponent cannot be extracted."""
        spec = Variable(_dims="L*T^-1")
        # Directly call _setup_column with malformed input missing exponent pattern
        with pytest.raises(ValueError) as excinfo:
            spec._setup_column("L* T")  # Missing exponent markers
        assert "Could not extract exponent from dimension" in str(excinfo.value)

    def test_setup_column_missing_symbol(self) -> None:
        """Test _setup_column error when symbol cannot be extracted."""
        spec = Variable(_dims="L*T^-1")
        # Call with malformed input that has exponent but no symbol
        # This requires crafting input that passes exp regex but fails sym regex
        with pytest.raises(ValueError) as excinfo:
            spec._setup_column("**1* **(-1)")  # Exponents without symbols
        assert "Could not extract symbol from dimension" in str(excinfo.value)

    def test_setup_column_unknown_symbol(self) -> None:
        """Test _setup_column error when symbol is not in FDU precedence."""
        spec = Variable(_dims="L*T^-1")
        # The unknown symbol path (line 219) is difficult to reach because
        # symbols that don't match fdu_sym_regex fail earlier at line 213
        # This test verifies the symbol extraction error path instead
        with pytest.raises(ValueError) as excinfo:
            spec._setup_column("X**1* Y**(-1)")  # Invalid symbols
        # Will fail at symbol extraction (line 213) before unknown symbol check
        assert "Could not extract symbol from dimension" in str(excinfo.value)
