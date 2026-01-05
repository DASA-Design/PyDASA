# -*- coding: utf-8 -*-
"""
Test Module for specs/symbolic.py
===========================================

Tests for the **SymbolicSpecs** class in *PyDASA*.
"""

import unittest
import pytest

from pydasa.dimensional.framework import Schema
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
        self.test_scheme.update_global_config()

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
        self.test_scheme.update_global_config()

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
