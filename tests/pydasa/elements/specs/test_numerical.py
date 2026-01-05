# -*- coding: utf-8 -*-
"""
Test Module for specs/numerical.py
===========================================

Tests for the **NumericalSpecs** class in *PyDASA*.
"""

import unittest
import pytest
import numpy as np
from typing import Any, cast

from pydasa.elements.parameter import Variable
# from pydasa.elements.specs.numerical import NumericalSpecs


class TestNumericalSpecs(unittest.TestCase):
    """Test cases for **NumericalSpecs** class via **Variable**."""

    # Original units range tests (min, max, mean, dev)
    def test_min_getter(self) -> None:
        """Test min property getter."""
        spec = Variable()
        spec.min = 0.0
        assert spec.min == 0.0

    def test_min_setter_valid(self) -> None:
        """Test min property setter with valid value."""
        spec = Variable()
        spec.max = 10.0
        spec.min = 0.0
        assert spec.min == 0.0

    def test_min_setter_invalid_type(self) -> None:
        """Test min property setter with invalid type."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.min = cast(Any, "not a number")
        assert "min must be int or float" in str(excinfo.value)

    def test_min_max_relationship(self) -> None:
        """Test min and max relationship validation."""
        spec = Variable()
        spec.min = 0.0
        spec.max = 10.0

        assert spec.min == 0.0
        assert spec.max == 10.0

        # Test min > max raises error
        with pytest.raises(ValueError) as excinfo:
            spec.min = 15.0
        assert "cannot be greater than maximum" in str(excinfo.value)

    def test_max_getter(self) -> None:
        """Test max property getter."""
        spec = Variable()
        spec.max = 10.0
        assert spec.max == 10.0

    def test_max_setter_valid(self) -> None:
        """Test max property setter with valid value."""
        spec = Variable()
        spec.min = 0.0
        spec.max = 10.0
        assert spec.max == 10.0

    def test_max_setter_invalid_type(self) -> None:
        """Test max property setter with invalid type."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.max = cast(Any, "not a number")
        assert "max must be int or float" in str(excinfo.value)

    def test_max_min_relationship(self) -> None:
        """Test max and min relationship validation."""
        spec = Variable()
        spec.max = 10.0
        spec.min = 0.0

        # Test max < min raises error
        with pytest.raises(ValueError) as excinfo:
            spec.max = -5.0
        assert "cannot be less than minimum" in str(excinfo.value)

    def test_mean_getter(self) -> None:
        """Test mean property getter."""
        spec = Variable()
        spec.min = 0.0
        spec.max = 10.0
        spec.mean = 5.0
        assert spec.mean == 5.0

    def test_mean_setter_valid(self) -> None:
        """Test mean property setter with valid value."""
        spec = Variable()
        spec.min = 0.0
        spec.max = 10.0
        spec.mean = 5.0
        assert spec.mean == 5.0

    def test_mean_setter_invalid_type(self) -> None:
        """Test mean property setter with invalid type."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.mean = cast(Any, "not a number")
        assert "mean must be int or float" in str(excinfo.value)

    def test_mean_setter_out_of_range(self) -> None:
        """Test mean property setter with value outside range."""
        spec = Variable()
        spec.min = 0.0
        spec.max = 10.0

        # Test mean > max raises error
        with pytest.raises(ValueError) as excinfo:
            spec.mean = 15.0
        assert "must be between" in str(excinfo.value)

        # Test mean < min raises error
        with pytest.raises(ValueError) as excinfo:
            spec.mean = -5.0
        assert "must be between" in str(excinfo.value)

    # Standardized unit range tests (std_* units, min, max, mean, dev)
    def test_std_min_getter(self) -> None:
        """Test std_min property getter."""
        spec = Variable()
        spec.std_min = 0.0
        assert spec.std_min == 0.0

    def test_std_min_setter_valid(self) -> None:
        """Test std_min property setter with valid value."""
        spec = Variable()
        spec.std_max = 100.0
        spec.std_min = 0.0
        assert spec.std_min == 0.0

    def test_std_min_setter_invalid_type(self) -> None:
        """Test std_min property setter with invalid type."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.std_min = cast(Any, "not a number")
        assert "Standardized minimum must be a number" in str(excinfo.value)

    def test_std_max_getter(self) -> None:
        """Test std_max property getter."""
        spec = Variable()
        spec.std_max = 100.0
        assert spec.std_max == 100.0

    def test_std_max_setter_valid(self) -> None:
        """Test std_max property setter with valid value."""
        spec = Variable()
        spec.std_min = 0.0
        spec.std_max = 100.0
        assert spec.std_max == 100.0

    def test_std_max_setter_invalid_type(self) -> None:
        """Test std_max property setter with invalid type."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.std_max = cast(Any, "not a number")
        assert "Standardized maximum must be a number" in str(excinfo.value)

    def test_std_min_max_relationship(self) -> None:
        """Test std_min and std_max relationship validation."""
        spec = Variable()
        spec.std_min = 0.0
        spec.std_max = 100.0

        assert spec.std_min == 0.0
        assert spec.std_max == 100.0

        # Test std_min > std_max raises error
        with pytest.raises(ValueError) as excinfo:
            spec.std_min = 150.0
        assert "cannot be greater" in str(excinfo.value)

        # Test std_max < std_min raises error
        with pytest.raises(ValueError) as excinfo:
            spec.std_max = -50.0
        assert "cannot be less" in str(excinfo.value)

    def test_std_mean_getter(self) -> None:
        """Test std_mean property getter."""
        spec = Variable()
        spec.std_min = 0.0
        spec.std_max = 100.0
        spec.std_mean = 50.0
        assert spec.std_mean == 50.0

    def test_std_mean_setter_valid(self) -> None:
        """Test std_mean property setter with valid value."""
        spec = Variable()
        spec.std_min = 0.0
        spec.std_max = 100.0
        spec.std_mean = 50.0
        assert spec.std_mean == 50.0

    def test_std_mean_setter_invalid_type(self) -> None:
        """Test std_mean property setter with invalid type."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.std_mean = cast(Any, "not a number")
        assert "Standardized mean must be a number" in str(excinfo.value)

    def test_std_mean_setter_out_of_range(self) -> None:
        """Test std_mean property setter with value outside range."""
        spec = Variable()
        spec.std_min = 0.0
        spec.std_max = 100.0

        # Test std_mean > std_max raises error
        with pytest.raises(ValueError) as excinfo:
            spec.std_mean = 150.0
        assert "must be between" in str(excinfo.value)

        # Test std_mean < std_min raises error
        with pytest.raises(ValueError) as excinfo:
            spec.std_mean = -50.0
        assert "must be between" in str(excinfo.value)

    def test_std_dev_getter(self) -> None:
        """Test std_dev property getter."""
        spec = Variable()
        spec.std_dev = 10.0
        assert spec.std_dev == 10.0

    def test_std_dev_setter_valid(self) -> None:
        """Test std_dev property setter with valid value."""
        spec = Variable()
        spec.std_dev = 10.0
        assert spec.std_dev == 10.0

    def test_std_dev_setter_invalid_type(self) -> None:
        """Test std_dev property setter with invalid type."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.std_dev = cast(Any, "not a number")
        assert "Standardized standard deviation must be a number" in str(excinfo.value)

    def test_std_dev_negative(self) -> None:
        """Test std_dev property setter with negative value."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.std_dev = -5.0
        assert "cannot be negative" in str(excinfo.value)

    # Step and range tests
    def test_step_getter(self) -> None:
        """Test step property getter."""
        spec = Variable()
        spec.std_min = 0.0
        spec.std_max = 10.0
        spec.step = 1.0
        assert spec.step == 1.0

    def test_step_setter_valid(self) -> None:
        """Test step property setter with valid value."""
        spec = Variable()
        spec.std_min = 0.0
        spec.std_max = 10.0
        spec.step = 1.0
        assert spec.step == 1.0

    def test_step_setter_invalid_type(self) -> None:
        """Test step property setter with invalid type."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.step = cast(Any, "not a number")
        assert "Step must be a number" in str(excinfo.value)

    def test_step_setter_zero(self) -> None:
        """Test step property setter with zero value."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.step = 0.0
        assert "Step cannot be zero" in str(excinfo.value)

    def test_step_setter_too_large(self) -> None:
        """Test step property setter with value larger than range."""
        spec = Variable()
        spec.std_min = 0.0
        spec.std_max = 10.0

        with pytest.raises(ValueError) as excinfo:
            spec.step = 15.0
        assert "must be less than range" in str(excinfo.value)

    def test_std_range_getter(self) -> None:
        """Test std_range property getter."""
        spec = Variable()
        spec.std_min = 0.0
        spec.std_max = 10.0
        spec.step = 2.0

        assert isinstance(spec.std_range, np.ndarray)
        assert len(spec.std_range) > 0

    def test_std_range_automatic_generation(self) -> None:
        """Test automatic std_range generation."""
        spec = Variable()
        spec.std_min = 0.0
        spec.std_max = 10.0
        spec.step = 2.0

        expected_range = np.arange(0.0, 10.0, 2.0)
        assert np.array_equal(spec.std_range, expected_range)

    def test_std_range_setter_valid(self) -> None:
        """Test std_range property setter with valid array."""
        spec = Variable()
        custom_range = np.array([0.0, 1.0, 2.0, 3.0])
        spec.std_range = custom_range
        assert np.array_equal(spec.std_range, custom_range)

    def test_std_range_setter_invalid_type(self) -> None:
        """Test std_range property setter with invalid type."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.std_range = cast(Any, [0, 1, 2, 3])
        assert "Range must be a numpy array" in str(excinfo.value)
