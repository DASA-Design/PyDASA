# # -*- coding: utf-8 -*-
# """
# Test Module for parameter.py
# ===========================================

# Tests for the **Variable** class in *PyDASA*.
# """

# import unittest
# import pytest
# import numpy as np
# from typing import Any, cast

# from pydasa.dimensional.framework import Schema
# from pydasa.core.parameter import Variable
# from tests.pydasa.data.test_data import get_variable_test_data


# class TestVariable(unittest.TestCase):
#     """Test cases for Variable class."""

#     @pytest.fixture(autouse=True)
#     def inject_fixtures(self) -> None:
#         """Inject test data fixture."""
#         self.test_data = get_variable_test_data()

#         # resseting config due to singletton patterm fix later
#         self.test_scheme = Schema(_fwk="PHYSICAL")
#         self.test_scheme.update_global_config()

#     # Initialization tests
#     def test_default_initialization(self) -> None:
#         """Test creating Variable with default values."""
#         var = Variable()

#         assert var is not None
#         assert var._idx == -1
#         assert var._sym.startswith("V_")
#         assert var._alias != ""
#         assert var._fwk == "PHYSICAL"
#         assert var._cat == "IN"
#         assert var._dims == ""
#         assert var._units == ""
#         assert var.relevant is False
#         assert isinstance(var, Variable)

#     def test_specific_variable_types(self) -> None:
#         """Test creating specific variable types (physical, computation, software)."""
#         variable_types = [
#             self.test_data["PHYSICAL_VARIABLE"],
#             self.test_data["COMPUTATION_VARIABLE"],
#             self.test_data["SOFTWARE_VARIABLE"]
#         ]

#         for data in variable_types:
#             # Update dimensional scheme for each framework type
#             scheme = Schema(_fwk=data["_fwk"])
#             scheme.update_global_config()

#             var = Variable(
#                 _idx=data["_idx"],
#                 _sym=data["_sym"],
#                 _alias=data["_alias"],
#                 _fwk=data["_fwk"],
#                 _cat=data["_cat"],
#                 _dims=data["_dims"],
#                 _units=data["_units"],
#                 name=data["name"],
#                 description=data["description"])

#             assert var._idx == data["_idx"]
#             assert var._sym == data["_sym"]
#             assert var._alias == data["_alias"]
#             assert var._fwk == data["_fwk"]
#             assert var._cat == data["_cat"]
#             assert var._dims == data["_dims"]
#             assert var._units == data["_units"]
#             assert var.name == data["name"]

#     # Category property tests
#     def test_cat_getter(self) -> None:
#         """Test cat property getter."""
#         var = Variable(_cat="OUT")
#         assert var.cat == "OUT"

#     def test_cat_setter_valid(self) -> None:
#         """Test cat property setter with valid values."""
#         var = Variable()
#         for cat in self.test_data["VALID_CATEGORIES"]:
#             var.cat = cat
#             assert var.cat == cat.upper()

#     def test_cat_setter_invalid(self) -> None:
#         """Test cat property setter with invalid values."""
#         var = Variable()
#         for invalid_cat in self.test_data["INVALID_CATEGORIES"]:
#             with pytest.raises(ValueError) as excinfo:
#                 var.cat = invalid_cat
#             assert "Invalid category" in str(excinfo.value)

#     # Dimensions property tests
#     def test_dims_getter(self) -> None:
#         """Test dims property getter."""
#         var = Variable(_dims="L*T^-1")
#         assert var.dims == "L*T^-1"

#     def test_dims_setter_valid(self) -> None:
#         """Test dims property setter with valid values."""
#         var = Variable()

#         # resseting config due to singletton patterm fix later
#         self.test_scheme = Schema(_fwk="PHYSICAL")
#         self.test_scheme.update_global_config()

#         for dims in self.test_data["VALID_DIMENSIONS"]:
#             var.dims = dims
#             assert var.dims == dims

#     def test_dims_setter_invalid_empty(self) -> None:
#         """Test dims property setter with empty string."""
#         var = Variable()
#         with pytest.raises(ValueError) as excinfo:
#             var.dims = "   "
#         assert "Dimensions cannot be empty" in str(excinfo.value)

#     def test_dims_processing(self) -> None:
#         """Test dimensional expression processing."""
#         var = Variable(_dims="L*T^-1", _units="m/s")

#         assert var._std_dims is not None
#         assert var._sym_exp is not None
#         assert len(var._dim_col) > 0

#     # Units property tests
#     def test_units_getter(self) -> None:
#         """Test units property getter."""
#         var = Variable(_units="m/s")
#         assert var.units == "m/s"

#     def test_units_setter_valid(self) -> None:
#         """Test units property setter with valid values."""
#         var = Variable()
#         for units in self.test_data["VALID_UNITS"]:
#             var.units = units
#             assert var.units == units

#     def test_units_setter_invalid_empty(self) -> None:
#         """Test units property setter with empty string."""
#         var = Variable()
#         with pytest.raises(ValueError) as excinfo:
#             var.units = "   "
#         assert "Units of Measure cannot be empty" in str(excinfo.value)

#     # Symbol expression property tests
#     def test_sym_exp_getter(self) -> None:
#         """Test sym_exp property getter."""
#         var = Variable(_dims="L*T^-1")
#         assert var.sym_exp is not None
#         assert isinstance(var.sym_exp, str)

#     def test_sym_exp_setter_valid(self) -> None:
#         """Test sym_exp property setter with valid value."""
#         var = Variable()
#         var.sym_exp = "L**1*T**(-1)"
#         assert var.sym_exp == "L**1*T**(-1)"

#     def test_sym_exp_setter_invalid_empty(self) -> None:
#         """Test sym_exp property setter with empty string."""
#         var = Variable()
#         with pytest.raises(ValueError) as excinfo:
#             var.sym_exp = "   "
#         assert "Dimensional expression cannot be empty" in str(excinfo.value)

#     # Dimensional column property tests
#     def test_dim_col_getter(self) -> None:
#         """Test dim_col property getter."""
#         var = Variable(_dims="L*T^-1")
#         assert var.dim_col is not None
#         assert isinstance(var.dim_col, list)

#     def test_dim_col_setter_valid(self) -> None:
#         """Test dim_col property setter with valid value."""
#         var = Variable()
#         var.dim_col = [1, 0, -1, 0, 0, 0, 0]
#         assert var.dim_col == [1, 0, -1, 0, 0, 0, 0]

#     def test_dim_col_setter_invalid(self) -> None:
#         """Test dim_col property setter with invalid type."""
#         var = Variable()
#         with pytest.raises(ValueError) as excinfo:
#             var.dim_col = cast(Any, "not a list")
#         assert "Dimensional column must be a list" in str(excinfo.value)

#     # Standardized dimensions property tests
#     def test_std_dims_getter(self) -> None:
#         """Test std_dims property getter."""
#         var = Variable(_dims="L*T^-1")
#         assert var.std_dims is not None
#         assert isinstance(var.std_dims, str)

#     def test_std_dims_setter_valid(self) -> None:
#         """Test std_dims property setter with valid value."""
#         var = Variable()
#         var.std_dims = "L^(1)*T^(-1)"
#         assert var.std_dims == "L^(1)*T^(-1)"

#     def test_std_dims_setter_invalid_empty(self) -> None:
#         """Test std_dims property setter with empty string."""
#         var = Variable()
#         with pytest.raises(ValueError) as excinfo:
#             var.std_dims = "   "
#         assert "Standardized dimensions cannot be empty" in str(excinfo.value)

#     # Original units range tests (min, max, mean, dev)
#     def test_min_getter(self) -> None:
#         """Test min property getter."""
#         var = Variable()
#         var.min = 0.0
#         assert var.min == 0.0

#     def test_min_setter_valid(self) -> None:
#         """Test min property setter with valid value."""
#         var = Variable()
#         var.max = 10.0
#         var.min = 0.0
#         assert var.min == 0.0

#     def test_min_setter_invalid_type(self) -> None:
#         """Test min property setter with invalid type."""
#         var = Variable()
#         with pytest.raises(ValueError) as excinfo:
#             var.min = cast(Any, "not a number")
#         assert "Minimum range must be a number" in str(excinfo.value)

#     def test_min_max_relationship(self) -> None:
#         """Test min and max relationship validation."""
#         var = Variable()
#         var.min = 0.0
#         var.max = 10.0

#         assert var.min == 0.0
#         assert var.max == 10.0

#         # Test min > max raises error
#         with pytest.raises(ValueError) as excinfo:
#             var.min = 15.0
#         assert "cannot be greater than maximum" in str(excinfo.value)

#     def test_max_getter(self) -> None:
#         """Test max property getter."""
#         var = Variable()
#         var.max = 10.0
#         assert var.max == 10.0

#     def test_max_setter_valid(self) -> None:
#         """Test max property setter with valid value."""
#         var = Variable()
#         var.min = 0.0
#         var.max = 10.0
#         assert var.max == 10.0

#     def test_max_setter_invalid_type(self) -> None:
#         """Test max property setter with invalid type."""
#         var = Variable()
#         with pytest.raises(ValueError) as excinfo:
#             var.max = cast(Any, "not a number")
#         assert "Maximum val must be a number" in str(excinfo.value)

#     def test_max_min_relationship(self) -> None:
#         """Test max and min relationship validation."""
#         var = Variable()
#         var.max = 10.0
#         var.min = 0.0

#         # Test max < min raises error
#         with pytest.raises(ValueError) as excinfo:
#             var.max = -5.0
#         assert "cannot be less than minimum" in str(excinfo.value)

#     def test_mean_getter(self) -> None:
#         """Test mean property getter."""
#         var = Variable()
#         var.min = 0.0
#         var.max = 10.0
#         var.mean = 5.0
#         assert var.mean == 5.0

#     def test_mean_setter_valid(self) -> None:
#         """Test mean property setter with valid value."""
#         var = Variable()
#         var.min = 0.0
#         var.max = 10.0
#         var.mean = 5.0
#         assert var.mean == 5.0

#     def test_mean_setter_invalid_type(self) -> None:
#         """Test mean property setter with invalid type."""
#         var = Variable()
#         with pytest.raises(ValueError) as excinfo:
#             var.mean = cast(Any, "not a number")
#         assert "Mean value must be a number" in str(excinfo.value)

#     def test_mean_setter_out_of_range(self) -> None:
#         """Test mean property setter with value outside range."""
#         var = Variable()
#         var.min = 0.0
#         var.max = 10.0

#         # Test mean > max raises error
#         with pytest.raises(ValueError) as excinfo:
#             var.mean = 15.0
#         assert "must be between" in str(excinfo.value)

#         # Test mean < min raises error
#         with pytest.raises(ValueError) as excinfo:
#             var.mean = -5.0
#         assert "must be between" in str(excinfo.value)

#     # Standardized unit range tests (std_* units, min, max, mean, dev)
#     def test_std_units_getter(self) -> None:
#         """Test std_units property getter."""
#         var = Variable()
#         var.std_units = "m/s"
#         assert var.std_units == "m/s"

#     def test_std_units_setter_valid(self) -> None:
#         """Test std_units property setter with valid value."""
#         var = Variable()
#         var.std_units = "m/s"
#         assert var.std_units == "m/s"

#     def test_std_units_setter_invalid_empty(self) -> None:
#         """Test std_units property setter with empty string."""
#         var = Variable()
#         with pytest.raises(ValueError) as excinfo:
#             var.std_units = "   "
#         assert "Standardized Units of Measure cannot be empty" in str(excinfo.value)

#     def test_std_min_getter(self) -> None:
#         """Test std_min property getter."""
#         var = Variable()
#         var.std_min = 0.0
#         assert var.std_min == 0.0

#     def test_std_min_setter_valid(self) -> None:
#         """Test std_min property setter with valid value."""
#         var = Variable()
#         var.std_max = 100.0
#         var.std_min = 0.0
#         assert var.std_min == 0.0

#     def test_std_min_setter_invalid_type(self) -> None:
#         """Test std_min property setter with invalid type."""
#         var = Variable()
#         with pytest.raises(ValueError) as excinfo:
#             var.std_min = cast(Any, "not a number")
#         assert "Standardized minimum must be a number" in str(excinfo.value)

#     def test_std_max_getter(self) -> None:
#         """Test std_max property getter."""
#         var = Variable()
#         var.std_max = 100.0
#         assert var.std_max == 100.0

#     def test_std_max_setter_valid(self) -> None:
#         """Test std_max property setter with valid value."""
#         var = Variable()
#         var.std_min = 0.0
#         var.std_max = 100.0
#         assert var.std_max == 100.0

#     def test_std_max_setter_invalid_type(self) -> None:
#         """Test std_max property setter with invalid type."""
#         var = Variable()
#         with pytest.raises(ValueError) as excinfo:
#             var.std_max = cast(Any, "not a number")
#         assert "Standardized maximum must be a number" in str(excinfo.value)

#     def test_std_min_max_relationship(self) -> None:
#         """Test std_min and std_max relationship validation."""
#         var = Variable()
#         var.std_min = 0.0
#         var.std_max = 100.0

#         assert var.std_min == 0.0
#         assert var.std_max == 100.0

#         # Test std_min > std_max raises error
#         with pytest.raises(ValueError) as excinfo:
#             var.std_min = 150.0
#         assert "cannot be greater" in str(excinfo.value)

#         # Test std_max < std_min raises error
#         with pytest.raises(ValueError) as excinfo:
#             var.std_max = -50.0
#         assert "cannot be less" in str(excinfo.value)

#     def test_std_mean_getter(self) -> None:
#         """Test std_mean property getter."""
#         var = Variable()
#         var.std_min = 0.0
#         var.std_max = 100.0
#         var.std_mean = 50.0
#         assert var.std_mean == 50.0

#     def test_std_mean_setter_valid(self) -> None:
#         """Test std_mean property setter with valid value."""
#         var = Variable()
#         var.std_min = 0.0
#         var.std_max = 100.0
#         var.std_mean = 50.0
#         assert var.std_mean == 50.0

#     def test_std_mean_setter_invalid_type(self) -> None:
#         """Test std_mean property setter with invalid type."""
#         var = Variable()
#         with pytest.raises(ValueError) as excinfo:
#             var.std_mean = cast(Any, "not a number")
#         assert "Standardized mean must be a number" in str(excinfo.value)

#     def test_std_mean_setter_out_of_range(self) -> None:
#         """Test std_mean property setter with value outside range."""
#         var = Variable()
#         var.std_min = 0.0
#         var.std_max = 100.0

#         # Test std_mean > std_max raises error
#         with pytest.raises(ValueError) as excinfo:
#             var.std_mean = 150.0
#         assert "must be between" in str(excinfo.value)

#         # Test std_mean < std_min raises error
#         with pytest.raises(ValueError) as excinfo:
#             var.std_mean = -50.0
#         assert "must be between" in str(excinfo.value)

#     def test_std_dev_getter(self) -> None:
#         """Test std_dev property getter."""
#         var = Variable()
#         var.std_dev = 10.0
#         assert var.std_dev == 10.0

#     def test_std_dev_setter_valid(self) -> None:
#         """Test std_dev property setter with valid value."""
#         var = Variable()
#         var.std_dev = 10.0
#         assert var.std_dev == 10.0

#     def test_std_dev_setter_invalid_type(self) -> None:
#         """Test std_dev property setter with invalid type."""
#         var = Variable()
#         with pytest.raises(ValueError) as excinfo:
#             var.std_dev = cast(Any, "not a number")
#         assert "Standardized standard deviation must be a number" in str(excinfo.value)

#     def test_std_dev_negative(self) -> None:
#         """Test std_dev property setter with negative value."""
#         var = Variable()
#         with pytest.raises(ValueError) as excinfo:
#             var.std_dev = -5.0
#         assert "cannot be negative" in str(excinfo.value)

#     # Step and range tests
#     def test_step_getter(self) -> None:
#         """Test step property getter."""
#         var = Variable()
#         var.std_min = 0.0
#         var.std_max = 10.0
#         var.step = 1.0
#         assert var.step == 1.0

#     def test_step_setter_valid(self) -> None:
#         """Test step property setter with valid value."""
#         var = Variable()
#         var.std_min = 0.0
#         var.std_max = 10.0
#         var.step = 1.0
#         assert var.step == 1.0

#     def test_step_setter_invalid_type(self) -> None:
#         """Test step property setter with invalid type."""
#         var = Variable()
#         with pytest.raises(ValueError) as excinfo:
#             var.step = cast(Any, "not a number")
#         assert "Step must be a number" in str(excinfo.value)

#     def test_step_setter_zero(self) -> None:
#         """Test step property setter with zero value."""
#         var = Variable()
#         with pytest.raises(ValueError) as excinfo:
#             var.step = 0.0
#         assert "Step cannot be zero" in str(excinfo.value)

#     def test_step_setter_too_large(self) -> None:
#         """Test step property setter with value larger than range."""
#         var = Variable()
#         var.std_min = 0.0
#         var.std_max = 10.0

#         with pytest.raises(ValueError) as excinfo:
#             var.step = 15.0
#         assert "must be less than range" in str(excinfo.value)

#     def test_std_range_getter(self) -> None:
#         """Test std_range property getter."""
#         var = Variable()
#         var.std_min = 0.0
#         var.std_max = 10.0
#         var.step = 2.0

#         assert isinstance(var.std_range, np.ndarray)
#         assert len(var.std_range) > 0

#     def test_std_range_automatic_generation(self) -> None:
#         """Test automatic std_range generation."""
#         var = Variable()
#         var.std_min = 0.0
#         var.std_max = 10.0
#         var.step = 2.0

#         expected_range = np.arange(0.0, 10.0, 2.0)
#         assert np.array_equal(var.std_range, expected_range)

#     def test_std_range_setter_valid(self) -> None:
#         """Test std_range property setter with valid array."""
#         var = Variable()
#         custom_range = np.array([0.0, 1.0, 2.0, 3.0])
#         var.std_range = custom_range
#         assert np.array_equal(var.std_range, custom_range)

#     def test_std_range_setter_invalid_type(self) -> None:
#         """Test std_range property setter with invalid type."""
#         var = Variable()
#         with pytest.raises(ValueError) as excinfo:
#             var.std_range = cast(Any, [0, 1, 2, 3])
#         assert "Range must be a numpy array" in str(excinfo.value)

#     # Distribution type tests
#     def test_dist_type_getter(self) -> None:
#         """Test dist_type property getter."""
#         var = Variable()
#         assert var.dist_type == "uniform"

#     def test_dist_type_setter_valid(self) -> None:
#         """Test dist_type property setter with valid values."""
#         var = Variable()
#         for dist_type in self.test_data["VALID_DIST_TYPES"]:
#             var.dist_type = dist_type
#             assert var.dist_type == dist_type

#     def test_dist_type_setter_invalid(self) -> None:
#         """Test dist_type property setter with invalid value."""
#         var = Variable()
#         with pytest.raises(ValueError) as excinfo:
#             var.dist_type = "invalid_distribution"
#         assert "Unsupported distribution type" in str(excinfo.value)

#     # ========================================================================
#     # Distribution Function Tests (sample() and has_function())
#     # ========================================================================

#     def test_has_function_no_distribution(self) -> None:
#         """Test has_function returns False when no distribution is set."""
#         var = Variable()
#         assert var.has_function() is False

#     def test_has_function_with_distribution(self) -> None:
#         """Test has_function returns True when distribution is set."""
#         var = Variable()
#         var.dist_func = self.test_data["SAMPLE_TEST_FUNCTIONS"]["constant"]
#         assert var.has_function() is True

#     def test_sample_no_distribution_raises_error(self) -> None:
#         """Test sample raises error when no distribution is set."""
#         var = Variable(_sym="x")

#         with pytest.raises(ValueError) as excinfo:
#             var.sample()
#         assert "No distribution set for variable 'x'" in str(excinfo.value)

#     def test_sample_with_lambda_function(self) -> None:
#         """Test sample with simple lambda function."""
#         var = Variable()
#         var.dist_func = self.test_data["SAMPLE_TEST_FUNCTIONS"]["constant"]

#         result = var.sample()
#         assert result == 42.0
#         assert isinstance(result, float)

#     def test_sample_with_numpy_uniform(self) -> None:
#         """Test sample with numpy uniform distribution."""
#         var = Variable()
#         var.dist_func = self.test_data["SAMPLE_TEST_FUNCTIONS"]["uniform"]

#         # Generate samples and check they're in range
#         samples = [var.sample() for _ in range(50)]

#         assert all(0 <= s <= 10 for s in samples)
#         assert len(set(samples)) > 1  # Should have variety

#     def test_sample_with_kwargs(self) -> None:
#         """Test sample with keyword arguments for dependent variables."""
#         var = Variable(_sym="y", _depends=["x"])
#         var.dist_func = self.test_data["SAMPLE_TEST_FUNCTIONS"]["dependent"]

#         result = var.sample(5.0)
#         assert result == 11.0

#     def test_sample_returns_float_type(self) -> None:
#         """Test sample always returns float, not array."""
#         var = Variable()
#         var.dist_func = self.test_data["SAMPLE_TEST_FUNCTIONS"]["uniform"]

#         result = var.sample()
#         assert isinstance(result, float)
#         assert not isinstance(result, np.ndarray)

#     # Distribution parameters tests
#     def test_dist_params_getter(self) -> None:
#         """Test dist_params property getter."""
#         var = Variable()
#         assert isinstance(var.dist_params, dict)

#     def test_dist_params_uniform_valid(self) -> None:
#         """Test dist_params for uniform distribution with valid values."""
#         var = Variable()
#         var.dist_type = "uniform"
#         var.dist_params = {"min": 0, "max": 10}

#         assert var.dist_params["min"] == 0
#         assert var.dist_params["max"] == 10

#     def test_dist_params_uniform_missing_keys(self) -> None:
#         """Test dist_params for uniform distribution with missing keys."""
#         var = Variable()
#         var.dist_type = "uniform"

#         with pytest.raises(ValueError) as excinfo:
#             var.dist_params = {"min": 0}
#         assert "needs 'min' and 'max' parameters" in str(excinfo.value)

#     def test_dist_params_uniform_invalid_range(self) -> None:
#         """Test dist_params for uniform distribution with invalid range."""
#         var = Variable()
#         var.dist_type = "uniform"

#         with pytest.raises(ValueError) as excinfo:
#             var.dist_params = {"min": 10, "max": 0}
#         assert "needs 'min' to be less than 'max'" in str(excinfo.value)

#     def test_dist_params_normal_valid(self) -> None:
#         """Test dist_params for normal distribution with valid values."""
#         var = Variable()
#         var.dist_type = "normal"
#         var.dist_params = {"mean": 5.0, "std": 1.0}

#         assert var.dist_params["mean"] == 5.0
#         assert var.dist_params["std"] == 1.0

#     def test_dist_params_normal_missing_keys(self) -> None:
#         """Test dist_params for normal distribution with missing keys."""
#         var = Variable()
#         var.dist_type = "normal"

#         with pytest.raises(ValueError) as excinfo:
#             var.dist_params = {"mean": 5.0}
#         assert "needs 'mean' and 'std' parameters" in str(excinfo.value)

#     def test_dist_params_normal_negative_std(self) -> None:
#         """Test dist_params for normal distribution with negative std."""
#         var = Variable()
#         var.dist_type = "normal"

#         with pytest.raises(ValueError) as excinfo:
#             var.dist_params = {"mean": 5.0, "std": -1.0}
#         assert "requires 'std' to be positive" in str(excinfo.value)

#     # Distribution function tests
#     def test_dist_func_getter(self) -> None:
#         """Test dist_func property getter."""
#         var = Variable()
#         assert var.dist_func is None

#     def test_dist_func_setter_valid_callable(self) -> None:
#         """Test dist_func property setter with valid callable."""
#         var = Variable()

#         def custom_dist():
#             return np.random.uniform(0, 1)

#         var.dist_func = custom_dist
#         assert var.dist_func == custom_dist
#         assert callable(var.dist_func)

#     def test_dist_func_setter_none(self) -> None:
#         """Test dist_func property setter with None."""
#         var = Variable()
#         var.dist_func = None
#         assert var.dist_func is None

#     def test_dist_func_setter_invalid_not_callable(self) -> None:
#         """Test dist_func property setter with non-callable."""
#         var = Variable()
#         with pytest.raises(TypeError) as excinfo:
#             var.dist_func = cast(Any, "not a function")
#         assert "must be callable" in str(excinfo.value)

#     # Dependencies tests
#     def test_depends_getter(self) -> None:
#         """Test depends property getter."""
#         var = Variable()
#         assert isinstance(var.depends, list)
#         assert len(var.depends) == 0

#     def test_depends_setter_valid(self) -> None:
#         """Test depends property setter with valid list."""
#         var = Variable()
#         depends_list = ["var1", "var2", "var3"]
#         var.depends = depends_list

#         assert var.depends == depends_list

#     def test_depends_setter_empty_list(self) -> None:
#         """Test depends property setter with empty list."""
#         var = Variable()
#         var.depends = []
#         assert var.depends == []

#     def test_depends_setter_invalid_not_list(self) -> None:
#         """Test depends property setter with non-list."""
#         var = Variable()

#         with pytest.raises(ValueError) as excinfo:
#             var.depends = cast(Any, "not a list")
#         assert "must be a list of strings" in str(excinfo.value)

#     def test_depends_setter_invalid_non_string_elements(self) -> None:
#         """Test depends property setter with non-string elements."""
#         var = Variable()

#         with pytest.raises(ValueError) as excinfo:
#             var.depends = cast(Any, ["var1", 123, "var3"])
#         assert "must be a list of strings" in str(excinfo.value)

#     # Utility methods tests
#     def test_clear_resets_all_attributes(self) -> None:
#         """Test clear method resets all attributes."""
#         data = self.test_data["PHYSICAL_VARIABLE"]
#         var = Variable(
#             _idx=data["_idx"],
#             _sym=data["_sym"],
#             _alias=data["_alias"],
#             _fwk=data["_fwk"],
#             _cat=data["_cat"],
#             _dims=data["_dims"],
#             _units=data["_units"],
#             name=data["name"],
#             description=data["description"])

#         var.clear()

#         assert var._idx == -1
#         assert var._sym == ""
#         assert var._alias == ""
#         assert var._fwk == "PHYSICAL"
#         assert var._cat == "IN"
#         assert var._dims == ""
#         assert var._units == ""
#         assert var.name == ""
#         assert var.description == ""
#         assert var.relevant is False

#     def test_to_dict_structure(self) -> None:
#         """Test to_dict method returns correct structure."""
#         data = self.test_data["PHYSICAL_VARIABLE"]
#         var = Variable(
#             _idx=data["_idx"],
#             _sym=data["_sym"],
#             _alias=data["_alias"],
#             _fwk=data["_fwk"],
#             _cat=data["_cat"],
#             _dims=data["_dims"],
#             _units=data["_units"],
#             name=data["name"],
#             description=data["description"])

#         result = var.to_dict()

#         assert isinstance(result, dict)
#         assert "idx" in result
#         assert "sym" in result
#         assert "alias" in result
#         assert "fwk" in result
#         assert "cat" in result
#         assert "dims" in result
#         assert "units" in result
#         assert "name" in result
#         assert "description" in result

#     def test_to_dict_values(self) -> None:
#         """Test to_dict method returns correct values."""
#         data = self.test_data["PHYSICAL_VARIABLE"]
#         var = Variable(
#             _idx=data["_idx"],
#             _sym=data["_sym"],
#             _alias=data["_alias"],
#             _fwk=data["_fwk"],
#             _cat=data["_cat"],
#             _dims=data["_dims"],
#             _units=data["_units"],
#             name=data["name"],
#             description=data["description"])

#         result = var.to_dict()

#         assert result["idx"] == data["_idx"]
#         assert result["sym"] == data["_sym"]
#         assert result["alias"] == data["_alias"]
#         assert result["fwk"] == data["_fwk"]
#         assert result["cat"] == data["_cat"]
#         assert result["dims"] == data["_dims"]
#         assert result["units"] == data["_units"]
#         assert result["name"] == data["name"]

#     def test_from_dict_creates_instance(self) -> None:
#         """Test from_dict method creates Variable instance."""
#         data = self.test_data["PHYSICAL_VARIABLE"]

#         var = Variable.from_dict(data)

#         assert isinstance(var, Variable)
#         assert var._idx == data["_idx"]
#         assert var._sym == data["_sym"]
#         assert var._alias == data["_alias"]
#         assert var._fwk == data["_fwk"]
#         assert var._cat == data["_cat"]

#     def test_to_dict_from_dict_roundtrip(self) -> None:
#         """Test round-trip conversion between Variable and dict."""
#         data = self.test_data["PHYSICAL_VARIABLE"]
#         var1 = Variable(
#             _idx=data["_idx"],
#             _sym=data["_sym"],
#             _alias=data["_alias"],
#             _fwk=data["_fwk"],
#             _cat=data["_cat"],
#             _dims=data["_dims"],
#             _units=data["_units"],
#             name=data["name"],
#             description=data["description"])

#         # Convert to dict and back
#         dict_data = var1.to_dict()
#         var2 = Variable.from_dict(dict_data)

#         # Test key attributes match
#         assert var1._idx == var2._idx
#         assert var1._sym == var2._sym
#         assert var1._alias == var2._alias
#         assert var1._fwk == var2._fwk
#         assert var1._cat == var2._cat
#         assert var1._dims == var2._dims
#         assert var1._units == var2._units
#         assert var1.name == var2.name

#     # Inheritance tests
#     def test_inheritance_from_validation(self) -> None:
#         """Test that Variable inherits from Foundation hierarchy."""
#         from pydasa.core.basic import Foundation, IdxBasis, SymBasis

#         var = Variable()

#         assert isinstance(var, Variable)
#         assert isinstance(var, Foundation)
#         assert isinstance(var, IdxBasis)
#         assert isinstance(var, SymBasis)
