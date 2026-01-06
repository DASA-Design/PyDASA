# -*- coding: utf-8 -*-
"""
Module test_influence.py
===========================================

Tests for **SensitivityAnalysis** in *PyDASA*.

This module provides unit tests for managing sensitivity analyses.
"""
# import testing package
import unittest
import pytest

# import the module to test
from pydasa.workflows.influence import SensitivityAnalysis

# import required classes
from pydasa.elements.parameter import Variable
from pydasa.dimensional.buckingham import Coefficient
from pydasa.dimensional.vaschy import Schema

# import test data
from tests.pydasa.data.test_data import get_simulation_test_data

# asserting module imports
assert SensitivityAnalysis
assert get_simulation_test_data


class TestSensitivityAnalysis(unittest.TestCase):
    """**TestSensitivityAnalysis** implements unit tests for Sensitivity handler.

    Args:
        unittest (TestCase): unittest.TestCase class for Python unit testing.
    """

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """*inject_fixtures()* injects global test parameters as fixture."""
        # Get test data
        self.test_data = get_simulation_test_data()

        # Setup dimensional schema
        self.dim_schema = Schema(_fwk="CUSTOM",
                                 _fdu_lt=self.test_data["FDU_LIST"],
                                 _idx=0)
        self.dim_schema._setup_fdus()

        # Setup variables
        self.test_variables = {}
        for sym, var_data in self.test_data["CHANNEL_FLOW_VARIABLES"].items():
            var = Variable(**var_data)
            self.test_variables[sym] = var

        # Setup mock coefficients (minimal for testing)
        self.test_coefficients = {
            "\\Pi_{0}": Coefficient(
                _idx=0,
                _sym="\\Pi_{0}",
                _alias="Pi_0",
                _fwk="CUSTOM",
                _cat="COMPUTED",
                _name="Test Coefficient 0",
                description="Test dimensionless coefficient"
            ),
            "\\Pi_{1}": Coefficient(
                _idx=1,
                _sym="\\Pi_{1}",
                _alias="Pi_1",
                _fwk="CUSTOM",
                _cat="COMPUTED",
                _name="Test Coefficient 1",
                description="Test dimensionless coefficient"
            )
        }

        # Setup coefficient variable dimensions and expressions
        self.test_coefficients["\\Pi_{0}"].var_dims = {
            "U": 1,
            "d": -1
        }
        self.test_coefficients["\\Pi_{0}"]._pi_expr = "U/d"

        self.test_coefficients["\\Pi_{1}"].var_dims = {
            "\\mu_{1}": 1,
            "y_{2}": -1,
        }
        self.test_coefficients["\\Pi_{1}"]._pi_expr = "\\mu_{1}/y_{2}"

    def test_default_handler(self) -> None:
        """*test_default_handler()* tests creating handler with defaults."""
        # Create handler with defaults
        handler = SensitivityAnalysis()

        # Test if handler is not None
        assert handler is not None
        # Test default category
        assert handler._cat == "SYM"
        # Test empty collections
        assert len(handler._variables) == 0
        assert len(handler._coefficients) == 0
        assert len(handler._analyses) == 0
        assert len(handler._results) == 0
        # Test if handler is instance
        assert isinstance(handler, SensitivityAnalysis)

    def test_custom_handler(self) -> None:
        """*test_custom_handler()* tests creating handler with custom values."""
        # Create handler with custom data
        handler = SensitivityAnalysis(
            _idx=0,
            _fwk="CUSTOM",
            _cat="NUM",
            _variables=self.test_variables,
            _coefficients=self.test_coefficients,
            _name="Test Sensitivity Handler",
            description="Test Sensitivity Analysis Handler"
        )

        # Test if handler is not None
        assert handler is not None
        # Test custom category
        assert handler._cat == "NUM"
        # Test variables set
        assert len(handler._variables) == len(self.test_variables)
        # Test coefficients set
        assert len(handler._coefficients) == len(self.test_coefficients)
        # Test name and description
        assert handler.name == "Test Sensitivity Handler"
        assert handler.description == "Test Sensitivity Analysis Handler"

    def test_create_analyses(self) -> None:
        """*test_create_analyses()* tests creating sensitivity analyses for coefficients."""
        # Create handler
        handler = SensitivityAnalysis(
            _fwk="CUSTOM",
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )

        # Create analyses
        handler._create_analyses()

        # Test analyses created
        assert len(handler._analyses) > 0
        # Test one analysis per coefficient
        assert len(handler._analyses) == len(self.test_coefficients)
        # Test each coefficient has an analysis
        for pi in self.test_coefficients.keys():
            assert pi in handler._analyses

    def test_get_variable_value_mean(self) -> None:
        """*test_get_variable_value_mean()* tests getting variable mean values."""
        # Create handler
        handler = SensitivityAnalysis(
            _variables=self.test_variables
        )

        # Get mean value for U
        value = handler._get_variable_value("U", "mean")
        assert value is not None
        assert isinstance(value, float)
        assert value == 7.5  # From test data

    def test_get_variable_value_min(self) -> None:
        """*test_get_variable_value_min()* tests getting variable minimum values."""
        # Create handler
        handler = SensitivityAnalysis(
            _variables=self.test_variables
        )

        # Get min value for U
        value = handler._get_variable_value("U", "min")
        assert value is not None
        assert isinstance(value, float)
        assert value == 0.0  # From test data

    def test_get_variable_value_max(self) -> None:
        """*test_get_variable_value_max()* tests getting variable maximum values."""
        # Create handler
        handler = SensitivityAnalysis(
            _variables=self.test_variables
        )

        # Get max value for U
        value = handler._get_variable_value("U", "max")
        assert value is not None
        assert isinstance(value, float)
        assert value == 15.0  # From test data

    def test_get_variable_value_invalid(self) -> None:
        """*test_get_variable_value_invalid()* tests error handling for invalid inputs."""
        # Create handler
        handler = SensitivityAnalysis(
            _variables=self.test_variables
        )

        # Test non-existent variable
        with pytest.raises(ValueError) as excinfo:
            handler._get_variable_value("INVALID", "mean")
        assert "not found" in str(excinfo.value)

        # Test invalid value type
        with pytest.raises(ValueError) as excinfo:
            handler._get_variable_value("U", "invalid")
        assert "Invalid value type" in str(excinfo.value)

    def test_validate_dict(self) -> None:
        """*test_validate_dict()* tests dictionary validation."""
        # Create handler
        handler = SensitivityAnalysis()

        # Test valid dictionary
        valid_dict = {"key1": Variable(), "key2": Variable()}
        assert handler._validate_dict(valid_dict, Variable)

        # Test invalid type
        with pytest.raises(ValueError) as excinfo:
            handler._validate_dict("not a dict", Variable)  # type: ignore
        assert "must be a dictionary" in str(excinfo.value)

        # Test empty dictionary
        with pytest.raises(ValueError) as excinfo:
            handler._validate_dict({}, Variable)
        assert "cannot be empty" in str(excinfo.value)

    def test_analyze_symbolic(self) -> None:
        """*test_analyze_symbolic()* tests symbolic sensitivity analysis."""
        # Create handler
        handler = SensitivityAnalysis(
            _fwk="CUSTOM",
            _cat="SYM",
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )

        # Run symbolic analysis
        results = handler.analyze_symbolic(val_type="mean")

        # Test results returned
        assert results is not None
        assert isinstance(results, dict)
        # Test results stored
        assert len(handler._results) > 0

    def test_analyze_numeric(self) -> None:
        """*test_analyze_numeric()* tests numerical sensitivity analysis."""
        # Create handler
        handler = SensitivityAnalysis(
            _fwk="CUSTOM",
            _cat="NUM",
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )

        # Run numerical analysis (with sample size > 64 for SALib FAST)
        results = handler.analyze_numeric(n_samples=200)

        # Test results returned
        assert results is not None
        assert isinstance(results, dict)
        # Test results stored
        assert len(handler._results) > 0

    def test_clear(self) -> None:
        """*test_clear()* tests clearing handler state."""
        # Create handler with data
        handler = SensitivityAnalysis(
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )

        # Create analyses
        handler._create_analyses()

        # Clear handler
        handler.clear()

        # Test collections cleared
        assert len(handler._analyses) == 0
        assert len(handler._results) == 0
        assert len(handler._variables) == 0
        assert len(handler._coefficients) == 0

    def test_properties_cat(self) -> None:
        """*test_properties_cat()* tests category property getter and setter."""
        # Create handler
        handler = SensitivityAnalysis()

        # Test default cat property
        assert handler.cat == "SYM"

        # Test cat setter
        handler.cat = "NUM"
        assert handler.cat == "NUM"

    def test_properties_variables(self) -> None:
        """*test_properties_variables()* tests variables property getter and setter."""
        # Create handler
        handler = SensitivityAnalysis()

        # Test initial empty
        assert len(handler.variables) == 0

        # Test setter
        handler.variables = self.test_variables
        assert len(handler.variables) == len(self.test_variables)

        # Test invalid type
        with pytest.raises(ValueError) as excinfo:
            handler.variables = "not a dict"  # type: ignore
        assert "must be dict" in str(excinfo.value)

    def test_properties_coefficients(self) -> None:
        """*test_properties_coefficients()* tests coefficients property getter and setter."""
        # Create handler
        handler = SensitivityAnalysis()

        # Test initial empty
        assert len(handler.coefficients) == 0

        # Test setter
        handler.coefficients = self.test_coefficients
        assert len(handler.coefficients) == len(self.test_coefficients)

        # Test invalid type
        with pytest.raises(ValueError) as excinfo:
            handler.coefficients = "not a dict"  # type: ignore
        assert "must be dict" in str(excinfo.value)

    def test_properties_analyses(self) -> None:
        """*test_properties_analyses()* tests analyses property getter (read-only)."""
        # Create handler
        handler = SensitivityAnalysis(
            _fwk="CUSTOM",
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )

        # Create analyses
        handler._create_analyses()

        # Test getter
        analyses = handler.analyses
        assert isinstance(analyses, dict)
        assert len(analyses) == len(self.test_coefficients)

    def test_properties_results(self) -> None:
        """*test_properties_results()* tests results property getter (read-only)."""
        # Create handler
        handler = SensitivityAnalysis(
            _fwk="CUSTOM",
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )

        # Run analysis to generate results
        handler.analyze_symbolic()

        # Test getter
        results = handler.results
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_to_dict(self) -> None:
        """*test_to_dict()* tests converting handler to dictionary."""
        # Create handler
        handler = SensitivityAnalysis(
            _idx=0,
            _fwk="CUSTOM",
            _name="Test",
            description="Test handler"
        )

        # Convert to dict
        data = handler.to_dict()

        # Test dictionary structure
        assert isinstance(data, dict)
        assert "name" in data
        assert "description" in data
        assert "idx" in data
        assert "fwk" in data
        assert data["name"] == "Test"
        assert data["description"] == "Test handler"

    def test_analyses_create_automatically(self) -> None:
        """*test_analyses_create_automatically()* tests that analyses are created when needed."""
        # Create handler
        handler = SensitivityAnalysis(
            _fwk="CUSTOM",
            _cat="SYM",
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )

        # Initially no analyses
        assert len(handler._analyses) == 0

        # Run symbolic analysis - should create analyses automatically
        handler.analyze_symbolic()

        # Now analyses should exist
        assert len(handler._analyses) > 0
        assert len(handler._analyses) == len(self.test_coefficients)

    def test_multiple_analyses(self) -> None:
        """*test_multiple_analyses()* tests running multiple analysis types."""
        # Create handler
        handler = SensitivityAnalysis(
            _fwk="CUSTOM",
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )

        # Run symbolic analysis
        handler._cat = "SYM"
        symbolic_results = handler.analyze_symbolic()
        assert len(symbolic_results) > 0

        # Run numeric analysis
        handler._cat = "NUM"
        numeric_results = handler.analyze_numeric(n_samples=200)
        assert len(numeric_results) > 0

        # Both analyses should produce results
        assert isinstance(symbolic_results, dict)
        assert isinstance(numeric_results, dict)
        # Results should have same coefficient keys
        assert set(symbolic_results.keys()) == set(numeric_results.keys())

    def test_handler_with_no_variables(self) -> None:
        """*test_handler_with_no_variables()* tests error handling when no variables."""
        # Create handler without variables
        handler = SensitivityAnalysis(
            _coefficients=self.test_coefficients
        )

        # Attempting to get variable value should fail
        with pytest.raises(ValueError) as excinfo:
            handler._get_variable_value("U", "mean")
        assert "not found" in str(excinfo.value)

    def test_handler_with_no_coefficients(self) -> None:
        """*test_handler_with_no_coefficients()* tests handler with no coefficients."""
        # Create handler without coefficients
        handler = SensitivityAnalysis(
            _variables=self.test_variables
        )

        # Create analyses - should work but create empty dict
        handler._create_analyses()
        assert len(handler._analyses) == 0

        # Run analysis - should return empty results
        results = handler.analyze_symbolic()
        assert len(results) == 0
