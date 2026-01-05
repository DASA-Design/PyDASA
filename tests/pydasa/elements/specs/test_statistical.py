# -*- coding: utf-8 -*-
"""
Test Module for specs/statistical.py
===========================================

Tests for the **StatisticalSpecs** class in *PyDASA*.
"""

import unittest
import pytest
import numpy as np
from typing import Any, cast

# from pydasa.elements.specs.statistical import StatisticalSpecs
from pydasa.elements.parameter import Variable
from tests.pydasa.data.test_data import get_variable_test_data


class TestStatisticalSpecs(unittest.TestCase):
    """Test cases for StatisticalSpecs class."""

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """Inject test data fixture."""
        self.test_data = get_variable_test_data()

    # Distribution type tests
    def test_dist_type_getter(self) -> None:
        """Test dist_type property getter."""
        spec = Variable()
        assert spec.dist_type == "uniform"

    def test_dist_type_setter_valid(self) -> None:
        """Test dist_type property setter with valid values."""
        spec = Variable()
        for dist_type in self.test_data["VALID_DIST_TYPES"]:
            spec.dist_type = dist_type
            assert spec.dist_type == dist_type

    def test_dist_type_setter_invalid(self) -> None:
        """Test dist_type property setter with invalid value."""
        spec = Variable()
        with pytest.raises(ValueError) as excinfo:
            spec.dist_type = "invalid_distribution"
        assert "Unsupported distribution type" in str(excinfo.value)

    # ========================================================================
    # Distribution Function Tests (sample() and has_function())
    # ========================================================================

    def test_has_function_no_distribution(self) -> None:
        """Test has_function returns False when no distribution is set."""
        spec = Variable()
        assert spec.has_function() is False

    def test_has_function_with_distribution(self) -> None:
        """Test has_function returns True when distribution is set."""
        spec = Variable()
        spec.dist_func = self.test_data["SAMPLE_TEST_FUNCTIONS"]["constant"]
        assert spec.has_function() is True

    def test_sample_no_distribution_raises_error(self) -> None:
        """Test sample raises error when no distribution is set."""
        spec = Variable(_sym="x")

        with pytest.raises(ValueError) as excinfo:
            spec.sample()
        assert "No distribution set for variable 'x'" in str(excinfo.value)

    def test_sample_with_lambda_function(self) -> None:
        """Test sample with simple lambda function."""
        spec = Variable()
        spec.dist_func = self.test_data["SAMPLE_TEST_FUNCTIONS"]["constant"]

        result = spec.sample()
        assert result == 42.0
        assert isinstance(result, float)

    def test_sample_with_numpy_uniform(self) -> None:
        """Test sample with numpy uniform distribution."""
        spec = Variable()
        spec.dist_func = self.test_data["SAMPLE_TEST_FUNCTIONS"]["uniform"]

        # Generate samples and check they're in range
        samples = [spec.sample() for _ in range(50)]

        assert all(0 <= s <= 10 for s in samples)
        assert len(set(samples)) > 1  # Should have variety

    def test_sample_with_kwargs(self) -> None:
        """Test sample with keyword arguments for dependent variables."""
        spec = Variable(_sym="y", _depends=["x"])
        spec.dist_func = self.test_data["SAMPLE_TEST_FUNCTIONS"]["dependent"]

        result = spec.sample(5.0)
        assert result == 11.0

    def test_sample_returns_float_type(self) -> None:
        """Test sample always returns float, not array."""
        spec = Variable()
        spec.dist_func = self.test_data["SAMPLE_TEST_FUNCTIONS"]["uniform"]

        result = spec.sample()
        assert isinstance(result, float)
        assert not isinstance(result, np.ndarray)

    # Distribution parameters tests
    def test_dist_params_getter(self) -> None:
        """Test dist_params property getter."""
        spec = Variable()
        assert isinstance(spec.dist_params, dict)

    def test_dist_params_uniform_valid(self) -> None:
        """Test dist_params for uniform distribution with valid values."""
        spec = Variable()
        spec.dist_type = "uniform"
        spec.dist_params = {"min": 0, "max": 10}

        assert spec.dist_params["min"] == 0
        assert spec.dist_params["max"] == 10

    def test_dist_params_uniform_missing_keys(self) -> None:
        """Test dist_params for uniform distribution with missing keys."""
        spec = Variable()
        spec.dist_type = "uniform"

        with pytest.raises(ValueError) as excinfo:
            spec.dist_params = {"min": 0}
        assert "needs 'min' and 'max' parameters" in str(excinfo.value)

    def test_dist_params_uniform_invalid_range(self) -> None:
        """Test dist_params for uniform distribution with invalid range."""
        spec = Variable()
        spec.dist_type = "uniform"

        with pytest.raises(ValueError) as excinfo:
            spec.dist_params = {"min": 10, "max": 0}
        assert "needs 'min' to be less than 'max'" in str(excinfo.value)

    def test_dist_params_normal_valid(self) -> None:
        """Test dist_params for normal distribution with valid values."""
        spec = Variable()
        spec.dist_type = "normal"
        spec.dist_params = {"mean": 5.0, "std": 1.0}

        assert spec.dist_params["mean"] == 5.0
        assert spec.dist_params["std"] == 1.0

    def test_dist_params_normal_missing_keys(self) -> None:
        """Test dist_params for normal distribution with missing keys."""
        spec = Variable()
        spec.dist_type = "normal"

        with pytest.raises(ValueError) as excinfo:
            spec.dist_params = {"mean": 5.0}
        assert "needs 'mean' and 'std' parameters" in str(excinfo.value)

    def test_dist_params_normal_negative_std(self) -> None:
        """Test dist_params for normal distribution with negative std."""
        spec = Variable()
        spec.dist_type = "normal"

        with pytest.raises(ValueError) as excinfo:
            spec.dist_params = {"mean": 5.0, "std": -1.0}
        assert "requires 'std' to be positive" in str(excinfo.value)

    # Distribution function tests
    def test_dist_func_getter(self) -> None:
        """Test dist_func property getter."""
        spec = Variable()
        assert spec.dist_func is None

    def test_dist_func_setter_valid_callable(self) -> None:
        """Test dist_func property setter with valid callable."""
        spec = Variable()

        def custom_dist():
            return np.random.uniform(0, 1)

        spec.dist_func = custom_dist
        assert spec.dist_func == custom_dist
        assert callable(spec.dist_func)

    def test_dist_func_setter_none(self) -> None:
        """Test dist_func property setter with None."""
        spec = Variable()
        spec.dist_func = None
        assert spec.dist_func is None

    def test_dist_func_setter_invalid_not_callable(self) -> None:
        """Test dist_func property setter with non-callable."""
        spec = Variable()
        with pytest.raises(TypeError) as excinfo:
            spec.dist_func = cast(Any, "not a function")
        assert "must be callable" in str(excinfo.value)

    # Dependencies tests
    def test_depends_getter(self) -> None:
        """Test depends property getter."""
        spec = Variable()
        assert isinstance(spec.depends, list)
        assert len(spec.depends) == 0

    def test_depends_setter_valid(self) -> None:
        """Test depends property setter with valid list."""
        spec = Variable()
        depends_list = ["var1", "var2", "var3"]
        spec.depends = depends_list

        assert spec.depends == depends_list

    def test_depends_setter_empty_list(self) -> None:
        """Test depends property setter with empty list."""
        spec = Variable()
        spec.depends = []
        assert spec.depends == []

    def test_depends_setter_invalid_not_list(self) -> None:
        """Test depends property setter with non-list."""
        spec = Variable()

        with pytest.raises(ValueError) as excinfo:
            spec.depends = cast(Any, "not a list")
        assert "must be a list of strings" in str(excinfo.value)

    def test_depends_setter_invalid_non_string_elements(self) -> None:
        """Test depends property setter with non-string elements."""
        spec = Variable()

        with pytest.raises(ValueError) as excinfo:
            spec.depends = cast(Any, ["var1", 123, "var3"])
        assert "must be a list of strings" in str(excinfo.value)
