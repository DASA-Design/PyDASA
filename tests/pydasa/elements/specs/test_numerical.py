# -*- coding: utf-8 -*-
"""
Test Module for specs/numerical.py
===========================================

Tests for the **BoundsSpecs**, **StandardizedSpecs**, and **NumericalSpecs** classes in *PyDASA*.
"""

import unittest
import pytest
import numpy as np
from typing import Any, cast

from pydasa.elements.parameter import Variable
from tests.pydasa.data.test_data import get_numerical_test_data
# from pydasa.elements.specs.numerical import BoundsSpecs, StandardizedSpecs, NumericalSpecs


class TestBoundsSpecs(unittest.TestCase):
    """Test cases for **BoundsSpecs** class via **Variable**."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """Inject test data fixture."""
        self.test_data = get_numerical_test_data()
        self.bounds = self.test_data["BOUNDS_VALUES"]
        self.invalid = self.test_data["INVALID_VALUES"]

    # Setpoint tests
    def test_setpoint_getter(self) -> None:
        """Test setpoint property getter."""
        spec = Variable()
        spec.setpoint = self.bounds["setpoint"]
        assert spec.setpoint == self.bounds["setpoint"]

    def test_setpoint_setter_valid(self) -> None:
        """Test setpoint property setter with valid value."""
        spec = Variable()
        spec.setpoint = self.bounds["setpoint"]
        assert spec.setpoint == self.bounds["setpoint"]

    def test_setpoint_setter_invalid_type(self) -> None:
        """Test setpoint property setter with invalid type."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.setpoint = cast(Any, self.invalid["type_error"])
        assert "setpoint must be int or float" in str(excinfo.value)

    # Original units range tests (min, max, mean, dev)
    def test_min_getter(self) -> None:
        """Test min property getter."""
        spec = Variable()
        spec.min = self.bounds["min"]
        assert spec.min == self.bounds["min"]

    def test_min_setter_valid(self) -> None:
        """Test min property setter with valid value."""
        spec = Variable()
        spec.max = self.bounds["max"]
        spec.min = self.bounds["min"]
        assert spec.min == self.bounds["min"]

    def test_min_setter_invalid_type(self) -> None:
        """Test min property setter with invalid type."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.min = cast(Any, self.invalid["type_error"])
        assert "min must be int or float" in str(excinfo.value)

    def test_min_max_relationship(self) -> None:
        """Test min and max relationship validation."""
        spec = Variable()
        spec.min = self.bounds["min"]
        spec.max = self.bounds["max"]

        assert spec.min == self.bounds["min"]
        assert spec.max == self.bounds["max"]

        # Test min > max raises error
        with pytest.raises(ValueError) as excinfo:
            spec.min = self.invalid["out_of_range_high"]
        assert "cannot be greater than maximum" in str(excinfo.value)

    def test_max_getter(self) -> None:
        """Test max property getter."""
        spec = Variable()
        spec.max = self.bounds["max"]
        assert spec.max == self.bounds["max"]

    def test_max_setter_valid(self) -> None:
        """Test max property setter with valid value."""
        spec = Variable()
        spec.min = self.bounds["min"]
        spec.max = self.bounds["max"]
        assert spec.max == self.bounds["max"]

    def test_max_setter_invalid_type(self) -> None:
        """Test max property setter with invalid type."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.max = cast(Any, self.invalid["type_error"])
        assert "max must be int or float" in str(excinfo.value)

    def test_max_min_relationship(self) -> None:
        """Test max and min relationship validation."""
        spec = Variable()
        spec.max = self.bounds["max"]
        spec.min = self.bounds["min"]

        # Test max < min raises error
        with pytest.raises(ValueError) as excinfo:
            spec.max = self.invalid["out_of_range_low"]
        assert "cannot be less than minimum" in str(excinfo.value)

    def test_mean_getter(self) -> None:
        """Test mean property getter."""
        spec = Variable()
        spec.min = self.bounds["min"]
        spec.max = self.bounds["max"]
        spec.mean = self.bounds["mean"]
        assert spec.mean == self.bounds["mean"]

    def test_mean_setter_valid(self) -> None:
        """Test mean property setter with valid value."""
        spec = Variable()
        spec.min = self.bounds["min"]
        spec.max = self.bounds["max"]
        spec.mean = self.bounds["mean"]
        assert spec.mean == self.bounds["mean"]

    def test_mean_setter_invalid_type(self) -> None:
        """Test mean property setter with invalid type."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.mean = cast(Any, self.invalid["type_error"])
        assert "mean must be int or float" in str(excinfo.value)

    def test_mean_setter_out_of_range(self) -> None:
        """Test mean property setter with value outside range."""
        spec = Variable()
        spec.min = self.bounds["min"]
        spec.max = self.bounds["max"]

        # Test mean > max raises error
        with pytest.raises(ValueError) as excinfo:
            spec.mean = self.invalid["out_of_range_high"]
        assert "cannot be greater than maximum" in str(excinfo.value)

        # Test mean < min raises error
        with pytest.raises(ValueError) as excinfo:
            spec.mean = self.invalid["out_of_range_low"]
        assert "cannot be less than minimum" in str(excinfo.value)

    def test_median_getter(self) -> None:
        """Test median property getter."""
        spec = Variable()
        spec.min = self.bounds["min"]
        spec.max = self.bounds["max"]
        spec.median = self.bounds["median"]
        assert spec.median == self.bounds["median"]

    def test_median_setter_valid(self) -> None:
        """Test median property setter with valid value."""
        spec = Variable()
        spec.min = self.bounds["min"]
        spec.max = self.bounds["max"]
        spec.median = self.bounds["median"]
        assert spec.median == self.bounds["median"]

    def test_median_setter_invalid_type(self) -> None:
        """Test median property setter with invalid type."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.median = cast(Any, self.invalid["type_error"])
        assert "median must be int or float" in str(excinfo.value)

    def test_median_setter_out_of_range(self) -> None:
        """Test median property setter with value outside range."""
        spec = Variable()
        spec.min = self.bounds["min"]
        spec.max = self.bounds["max"]

        # Test median > max raises error
        with pytest.raises(ValueError) as excinfo:
            spec.median = self.invalid["out_of_range_high"]
        assert "cannot be greater than maximum" in str(excinfo.value)

        # Test median < min raises error
        with pytest.raises(ValueError) as excinfo:
            spec.median = self.invalid["out_of_range_low"]
        assert "cannot be less than minimum" in str(excinfo.value)


class TestStandardizedSpecs(unittest.TestCase):
    """Test cases for **StandardizedSpecs** class via **Variable**."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """Inject test data fixture."""
        self.test_data = get_numerical_test_data()
        self.std = self.test_data["STANDARDIZED_VALUES"]
        self.invalid = self.test_data["INVALID_VALUES"]

    # Standardized setpoint tests
    def test_std_setpoint_getter(self) -> None:
        """Test std_setpoint property getter."""
        spec = Variable()
        spec.std_setpoint = self.std["std_setpoint"]
        assert spec.std_setpoint == self.std["std_setpoint"]

    def test_std_setpoint_setter_valid(self) -> None:
        """Test std_setpoint property setter with valid value."""
        spec = Variable()
        spec.std_setpoint = self.std["std_setpoint"]
        assert spec.std_setpoint == self.std["std_setpoint"]

    def test_std_setpoint_setter_invalid_type(self) -> None:
        """Test std_setpoint property setter with invalid type."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.std_setpoint = cast(Any, self.invalid["type_error"])
        assert "std_setpoint must be int or float" in str(excinfo.value)

    # Standardized unit range tests (std_* units, min, max, mean, dev)
    def test_std_min_getter(self) -> None:
        """Test std_min property getter."""
        spec = Variable()
        spec.std_min = self.std["std_min"]
        assert spec.std_min == self.std["std_min"]

    def test_std_min_setter_valid(self) -> None:
        """Test std_min property setter with valid value."""
        spec = Variable()
        spec.std_max = self.std["std_max"]
        spec.std_min = self.std["std_min"]
        assert spec.std_min == self.std["std_min"]

    def test_std_min_setter_invalid_type(self) -> None:
        """Test std_min property setter with invalid type."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.std_min = cast(Any, self.invalid["type_error"])
        assert "std_min must be int or float" in str(excinfo.value)

    def test_std_max_getter(self) -> None:
        """Test std_max property getter."""
        spec = Variable()
        spec.std_max = self.std["std_max"]
        assert spec.std_max == self.std["std_max"]

    def test_std_max_setter_valid(self) -> None:
        """Test std_max property setter with valid value."""
        spec = Variable()
        spec.std_min = self.std["std_min"]
        spec.std_max = self.std["std_max"]
        assert spec.std_max == self.std["std_max"]

    def test_std_max_setter_invalid_type(self) -> None:
        """Test std_max property setter with invalid type."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.std_max = cast(Any, self.invalid["type_error"])
        assert "std_max must be int or float" in str(excinfo.value)

    def test_std_min_max_relationship(self) -> None:
        """Test std_min and std_max relationship validation."""
        spec = Variable()
        spec.std_min = self.std["std_min"]
        spec.std_max = self.std["std_max"]

        assert spec.std_min == self.std["std_min"]
        assert spec.std_max == self.std["std_max"]

        # Test std_min > std_max raises error
        with pytest.raises(ValueError) as excinfo:
            spec.std_min = self.invalid["out_of_range_high"]
        assert "cannot be greater" in str(excinfo.value)

        # Test std_max < std_min raises error
        with pytest.raises(ValueError) as excinfo:
            spec.std_max = self.invalid["out_of_range_low"]
        assert "cannot be less" in str(excinfo.value)

    def test_std_mean_getter(self) -> None:
        """Test std_mean property getter."""
        spec = Variable()
        spec.std_min = self.std["std_min"]
        spec.std_max = self.std["std_max"]
        spec.std_mean = self.std["std_mean"]
        assert spec.std_mean == self.std["std_mean"]

    def test_std_mean_setter_valid(self) -> None:
        """Test std_mean property setter with valid value."""
        spec = Variable()
        spec.std_min = self.std["std_min"]
        spec.std_max = self.std["std_max"]
        spec.std_mean = self.std["std_mean"]
        assert spec.std_mean == self.std["std_mean"]

    def test_std_mean_setter_invalid_type(self) -> None:
        """Test std_mean property setter with invalid type."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.std_mean = cast(Any, self.invalid["type_error"])
        assert "std_mean must be int or float" in str(excinfo.value)

    def test_std_mean_setter_out_of_range(self) -> None:
        """Test std_mean property setter with value outside range."""
        spec = Variable()
        spec.std_min = self.std["std_min"]
        spec.std_max = self.std["std_max"]

        # Test std_mean > std_max raises error
        with pytest.raises(ValueError) as excinfo:
            spec.std_mean = self.invalid["out_of_range_high"]
        assert "cannot be greater than maximum" in str(excinfo.value)

        # Test std_mean < std_min raises error
        with pytest.raises(ValueError) as excinfo:
            spec.std_mean = self.invalid["out_of_range_low"]
        assert "cannot be less than minimum" in str(excinfo.value)

    def test_std_median_getter(self) -> None:
        """Test std_median property getter."""
        spec = Variable()
        spec.std_min = self.std["std_min"]
        spec.std_max = self.std["std_max"]
        spec.std_median = self.std["std_median"]
        assert spec.std_median == self.std["std_median"]

    def test_std_median_setter_valid(self) -> None:
        """Test std_median property setter with valid value."""
        spec = Variable()
        spec.std_min = self.std["std_min"]
        spec.std_max = self.std["std_max"]
        spec.std_median = self.std["std_median"]
        assert spec.std_median == self.std["std_median"]

    def test_std_median_setter_invalid_type(self) -> None:
        """Test std_median property setter with invalid type."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.std_median = cast(Any, self.invalid["type_error"])
        assert "std_median must be int or float" in str(excinfo.value)

    def test_std_median_setter_out_of_range(self) -> None:
        """Test std_median property setter with value outside range."""
        spec = Variable()
        spec.std_min = self.std["std_min"]
        spec.std_max = self.std["std_max"]

        # Test std_median > std_max raises error
        with pytest.raises(ValueError) as excinfo:
            spec.std_median = self.invalid["out_of_range_high"]
        assert "cannot be greater than maximum" in str(excinfo.value)

        # Test std_median < std_min raises error
        with pytest.raises(ValueError) as excinfo:
            spec.std_median = self.invalid["out_of_range_low"]
        assert "cannot be less than minimum" in str(excinfo.value)

    def test_std_dev_getter(self) -> None:
        """Test std_dev property getter."""
        spec = Variable()
        spec.std_dev = self.std["std_dev"]
        assert spec.std_dev == self.std["std_dev"]

    def test_std_dev_setter_valid(self) -> None:
        """Test std_dev property setter with valid value."""
        spec = Variable()
        spec.std_dev = self.std["std_dev"]
        assert spec.std_dev == self.std["std_dev"]

    def test_std_dev_setter_invalid_type(self) -> None:
        """Test std_dev property setter with invalid type."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.std_dev = cast(Any, self.invalid["type_error"])
        assert "std_dev must be int or float" in str(excinfo.value)

    def test_std_dev_negative(self) -> None:
        """Test std_dev property setter with negative value."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.std_dev = self.invalid["negative_dev"]
        assert "std_dev must be >= 0" in str(excinfo.value)


class TestNumericalSpecs(unittest.TestCase):
    """Test cases for **NumericalSpecs** class via **Variable**.

    Tests the combined functionality of BoundsSpecs, StandardizedSpecs, and discretization.
    """

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """Inject test data fixture."""
        self.test_data = get_numerical_test_data()
        self.std = self.test_data["STANDARDIZED_VALUES"]
        self.discret = self.test_data["DISCRETIZATION_VALUES"]
        self.invalid = self.test_data["INVALID_VALUES"]

    # Step and range tests (now part of NumericalSpecs)
    def test_step_getter(self) -> None:
        """Test step property getter."""
        spec = Variable()
        spec.std_min = self.std["std_min"]
        spec.std_max = 1.0
        spec.step = self.discret["step"]
        assert spec.step == self.discret["step"]

    def test_step_setter_valid(self) -> None:
        """Test step property setter with valid value."""
        spec = Variable()
        spec.std_min = self.std["std_min"]
        spec.std_max = 1.0
        spec.step = self.discret["step"]
        assert spec.step == self.discret["step"]

    def test_step_setter_invalid_type(self) -> None:
        """Test step property setter with invalid type."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.step = cast(Any, self.invalid["type_error"])
        assert "step must be int or float" in str(excinfo.value)

    def test_step_setter_zero(self) -> None:
        """Test step property setter with zero value."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.step = self.invalid["zero_step"]
        assert "step must be > 0" in str(excinfo.value)

    def test_step_setter_too_large(self) -> None:
        """Test step property setter with value larger than range."""
        spec = Variable()
        spec.std_min = self.std["std_min"]
        spec.std_max = 1.0

        with pytest.raises(ValueError) as excinfo:
            spec.step = self.invalid["large_step"]
        assert "cannot be greater than maximum" in str(excinfo.value)

    def test_data_getter(self) -> None:
        """Test data property getter."""
        spec = Variable()
        spec.std_min = self.std["std_min"]
        spec.std_max = 1.0
        spec.step = 0.5
        spec.generate_data()

        assert isinstance(spec.data, np.ndarray)
        assert len(spec.data) > 0

    def test_data_automatic_generation(self) -> None:
        """Test automatic data generation."""
        spec = Variable()
        spec.std_min = self.std["std_min"]
        spec.std_max = 1.0
        spec.step = 0.25
        spec.generate_data()

        # numpy.arange doesn't include endpoint
        assert len(spec.data) == 4  # 0.0, 0.25, 0.5, 0.75

    def test_data_setter_valid(self) -> None:
        """Test data property setter with valid array."""
        spec = Variable()
        custom_range = np.array([1.0, 2.0, 3.0])
        spec.data = custom_range
        assert np.array_equal(spec.data, custom_range)

    def test_data_setter_invalid_type(self) -> None:
        """Test data property setter with invalid type."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.data = cast(Any, self.invalid["type_error"])
        assert "data must be" in str(excinfo.value)

    def test_clear(self) -> None:
        """Test clear() method resets all numerical attributes."""
        spec = Variable()
        bounds = self.test_data["BOUNDS_VALUES"]
        std = self.test_data["STANDARDIZED_VALUES"]
        discret = self.test_data["DISCRETIZATION_VALUES"]

        # Set all properties
        spec.min = bounds["min"]
        spec.max = bounds["max"]
        spec.mean = bounds["mean"]
        spec.median = bounds["median"]
        spec.std_min = std["std_min"]
        spec.std_max = std["std_max"]
        spec.std_mean = std["std_mean"]
        spec.std_median = std["std_median"]
        spec.step = discret["step"]

        spec.clear()

        assert spec.min is None
        assert spec.max is None
        assert spec.mean is None
        assert spec.median is None
        assert spec.std_min is None
        assert spec.std_max is None
        assert spec.std_mean is None
        assert spec.std_median is None
        assert spec.step is None
        assert len(spec.data) == 0
