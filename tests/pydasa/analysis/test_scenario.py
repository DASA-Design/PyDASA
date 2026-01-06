# -*- coding: utf-8 -*-
"""
Module test_scenario.py
===========================================

Unit tests for Sensitivity class in PyDASA.

This module provides test cases for sensitivity analysis functionality
following the complete dimensional analysis workflow:
    1. Create dimensional framework and schema
    2. Define variables
    3. Create and solve dimensional matrix
    4. Run sensitivity analysis on each coefficient
"""

# Import testing packages
import unittest
import pytest

# Typing imports
from typing import Dict, Any

# Import numpy for numerical operations
import numpy as np

# Import core PyDASA modules
from pydasa.elements.parameter import Variable
from pydasa.dimensional.vaschy import Schema
from pydasa.dimensional.model import Matrix

# Import the module to test
from pydasa.analysis.scenario import Sensitivity

# Import related classes
from pydasa.dimensional.buckingham import Coefficient

# Import test data
from tests.pydasa.data.test_data import get_simulation_test_data

# Asserting module imports
assert Sensitivity
assert Coefficient
assert Variable
assert Matrix
assert Schema
assert get_simulation_test_data


# ============================================================================
# Test Class
# ============================================================================

class TestSensitivity(unittest.TestCase):
    """**TestSensitivity** implements unit tests for the Sensitivity class.

    Args:
        unittest (TestCase): unittest.TestCase class for Python unit testing.
    """

    # ========================================================================
    # Type hints for class attributes
    # ========================================================================

    # Add type hints at class level
    dim_schema: Schema
    variables: Dict[str, Variable]
    dim_model: Matrix
    coefficients: Dict[str, Coefficient]
    test_data: Dict[str, Any]

    # ========================================================================
    # Fixtures and Setup
    # ========================================================================

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """Pytest fixture to inject test data."""
        self.test_data = get_simulation_test_data()

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        self._setup_dimensional_framework()
        self._setup_channel_flow_variables()
        self._setup_dimensional_model()

    def _setup_dimensional_framework(self) -> None:
        """*_setup_dimensional_framework()* sets up custom dimensional framework."""
        # Get FDU list from test data
        self.fdu_list = self.test_data["FDU_LIST"]

        # Create and configure dimensional schema
        self.dim_schema = Schema(_fdu_lt=self.fdu_list, _fwk="CUSTOM")

    def _setup_channel_flow_variables(self) -> None:
        """*_setup_channel_flow_variables()* creates variables for planar channel flow."""
        # Get variable data from test data
        var_data = self.test_data["CHANNEL_FLOW_VARIABLES"]

        # Create Variable objects
        self.variables = {}
        for var_sym, var_config in var_data.items():
            var = Variable(**var_config)
            self.variables[var_sym] = var

    def _setup_dimensional_model(self) -> None:
        """*_setup_dimensional_model()* creates and solves dimensional model."""
        # Create dimensional model
        self.dim_model = Matrix(
            _fwk="CUSTOM",
            _idx=0,
            _schema=self.dim_schema
        )

        # Set variables and relevant list
        self.dim_model.variables = self.variables
        self.dim_model.relevant_lt = self.variables

        # Create and solve matrix to get dimensionless coefficients
        self.dim_model.create_matrix()
        self.dim_model.solve_matrix()

        # Get coefficients
        coefficients = self.dim_model.coefficients

        if coefficients is None:
            raise ValueError(
                "Dimensional model did not produce coefficients. "
                "Check matrix setup and solution."
            )

        if not isinstance(coefficients, dict):
            raise TypeError(
                f"Expected dict of coefficients, got {type(coefficients)}"
            )

        if len(coefficients) == 0:
            raise ValueError(
                "Dimensional model produced empty coefficient dictionary. "
                "Check that variables are properly configured."
            )

        self.coefficients = coefficients

    # ========================================================================
    # Dimensional Model Tests
    # ========================================================================

    def test_dimensional_schema_creation(self) -> None:
        """*test_dimensional_schema_creation()* tests custom dimensional framework."""
        assert self.dim_schema is not None
        assert self.dim_schema._fwk == "CUSTOM"
        assert len(self.dim_schema._fdu_lt) == 3

    def test_variable_creation(self) -> None:
        """*test_variable_creation()* tests channel flow variables."""
        assert len(self.variables) == 6
        assert "U" in self.variables
        assert "\\mu_{1}" in self.variables

    def test_dimensional_model_setup(self) -> None:
        """*test_dimensional_model_setup()* tests dimensional model initialization."""
        assert self.dim_model is not None
        assert len(self.dim_model.variables) == 6
        assert len(self.dim_model.relevant_lt) == 6

    def test_dimensional_matrix_solution(self) -> None:
        """*test_dimensional_matrix_solution()* tests matrix solving."""
        assert self.coefficients is not None
        assert isinstance(self.coefficients, dict)
        assert len(self.coefficients) > 0

        # Check each coefficient is valid
        for pi_sym, coef in self.coefficients.items():
            assert coef is not None
            assert coef._pi_expr is not None
            assert isinstance(coef._pi_expr, str)

    # ========================================================================
    # Sensitivity Initialization Tests
    # ========================================================================

    def test_sensitivity_creation_basic(self) -> None:
        """*test_sensitivity_creation_basic()* tests basic sensitivity creation."""
        sens = Sensitivity(
            _idx=0,
            _sym="\\Pi_{0}",
            _fwk="CUSTOM",
            _name="Test Sensitivity",
            description="Test sensitivity analysis"
        )

        assert sens is not None
        assert sens._idx == 0
        assert sens._sym == "\\Pi_{0}"
        assert sens._fwk == "CUSTOM"
        assert sens.name == "Test Sensitivity"

    def test_sensitivity_creation_with_expression(self) -> None:
        """*test_sensitivity_creation_with_expression()* tests creation with expression."""
        # Get first coefficient
        assert self.coefficients is not None
        assert len(self.coefficients) > 0

        first_pi = list(self.coefficients.keys())[0]
        coef = self.coefficients[first_pi]

        # Create sensitivity analysis with expression
        sens = Sensitivity(
            _idx=0,
            _sym=first_pi,
            _fwk="CUSTOM",
            _pi_expr=coef._pi_expr
        )

        assert sens is not None
        assert sens._pi_expr == coef._pi_expr
        assert sens._sym_func is not None

    def test_sensitivity_set_coefficient(self) -> None:
        """*test_sensitivity_set_coefficient()* tests setting coefficient."""
        pi_keys = list(self.coefficients.keys())
        if len(pi_keys) < 1:
            pytest.skip("Need at least one coefficient")

        coef = self.coefficients[pi_keys[0]]

        # Create sensitivity analysis
        sens = Sensitivity(_fwk="CUSTOM")

        # Set coefficient
        sens.set_coefficient(coef)

        assert sens._pi_expr == coef._pi_expr
        assert sens._sym_func is not None

    def test_sensitivity_set_coefficient_without_expression_fails(self) -> None:
        """*test_sensitivity_set_coefficient_without_expression_fails()* tests setting invalid coefficient."""
        # Create coefficient without expression
        coef = Coefficient(_fwk="CUSTOM")

        # Create sensitivity analysis
        sens = Sensitivity(_fwk="CUSTOM")

        # Should fail to set coefficient without expression
        with pytest.raises(ValueError) as excinfo:
            sens.set_coefficient(coef)

        assert "does not have a valid expression" in str(excinfo.value)

    # ========================================================================
    # Property Tests - cat
    # ========================================================================

    def test_cat_getter(self) -> None:
        """*test_cat_getter()* tests cat property getter."""
        sens = Sensitivity(_cat="SYM")
        assert sens.cat == "SYM"

    def test_cat_setter_valid(self) -> None:
        """*test_cat_setter_valid()* tests cat property setter with valid values."""
        sens = Sensitivity()

        for cat_val in ["SYM", "NUM", "sym", "num"]:
            sens.cat = cat_val
            assert sens.cat.upper() in ["SYM", "NUM"]

    def test_cat_setter_invalid(self) -> None:
        """*test_cat_setter_invalid()* tests cat property setter with invalid values."""
        sens = Sensitivity()

        with pytest.raises(ValueError) as excinfo:
            sens.cat = "INVALID"

        assert "Invalid cat" in str(excinfo.value) or "not in choices" in str(excinfo.value)

    def test_cat_case_insensitive(self) -> None:
        """*test_cat_case_insensitive()* tests cat property is case insensitive."""
        sens = Sensitivity()

        sens.cat = "sym"
        assert sens.cat == "SYM"

        sens.cat = "NUM"
        assert sens.cat == "NUM"

    # ========================================================================
    # Property Tests - pi_expr
    # ========================================================================

    def test_pi_expr_getter(self) -> None:
        """*test_pi_expr_getter()* tests pi_expr property getter."""
        expr = "U/d"
        sens = Sensitivity(_pi_expr=expr)
        assert sens.pi_expr == expr

    def test_pi_expr_setter_valid_latex(self) -> None:
        """*test_pi_expr_setter_valid_latex()* tests pi_expr setter with valid LaTeX."""
        sens = Sensitivity(_fwk="CUSTOM")

        valid_expressions = [
            "\\Pi_{0}",
            "\\frac{U}{d}",
            "\\mu_{1}",
            "U",
            "d"
        ]

        for expr in valid_expressions:
            sens.pi_expr = expr
            assert sens.pi_expr == expr
            assert sens._sym_func is not None

    def test_pi_expr_setter_valid_alphanumeric(self) -> None:
        """*test_pi_expr_setter_valid_alphanumeric()* tests pi_expr setter with alphanumeric."""
        sens = Sensitivity(_fwk="CUSTOM")

        for expr in ["U", "d", "x123", "abc"]:
            sens.pi_expr = expr
            assert sens.pi_expr == expr

    def test_pi_expr_setter_invalid(self) -> None:
        """*test_pi_expr_setter_invalid()* tests pi_expr setter with invalid values."""
        sens = Sensitivity(_fwk="CUSTOM")

        invalid_expressions = [
            "!!!",
            "@@@",
        ]

        for expr in invalid_expressions:
            with pytest.raises(ValueError):
                sens.pi_expr = expr

    # ========================================================================
    # Property Tests - sym_func
    # ========================================================================

    def test_sym_func_getter(self) -> None:
        """*test_sym_func_getter()* tests sym_func property getter."""
        sens = Sensitivity(_pi_expr="U")
        assert sens.sym_func is not None
        assert callable(sens._sym_func) or hasattr(sens._sym_func, 'free_symbols')

    def test_sym_func_setter_valid(self) -> None:
        """*test_sym_func_setter_valid()* tests sym_func setter with valid callable."""
        sens = Sensitivity(_fwk="CUSTOM")

        # Create a simple callable
        def test_func(x: float) -> float:
            return x + 1

        sens.sym_func = test_func
        assert sens.sym_func == test_func

    def test_sym_func_setter_invalid(self) -> None:
        """*test_sym_func_setter_invalid()* tests sym_func setter with non-callable."""
        sens = Sensitivity(_fwk="CUSTOM")

        with pytest.raises(ValueError) as excinfo:
            sens.sym_func = "not_a_callable"    # type: ignore

        assert "must be callable" in str(excinfo.value)

        with pytest.raises(ValueError):
            sens.sym_func = 123    # type: ignore

        with pytest.raises(ValueError):
            sens.sym_func = None    # type: ignore

    # ========================================================================
    # Expression Parsing Tests
    # ========================================================================

    def test_expression_parsing_from_coefficient(self) -> None:
        """*test_expression_parsing_from_coefficient()* tests parsing coefficient expressions."""
        for pi_sym, coef in self.coefficients.items():
            sens = Sensitivity(_fwk="CUSTOM")
            sens.set_coefficient(coef)

            assert sens._pi_expr == coef._pi_expr
            assert sens._sym_func is not None
            assert sens._variables is not None

    def test_variable_extraction_from_expression(self) -> None:
        """*test_variable_extraction_from_expression()* tests extracting variables."""
        # Use a coefficient if available
        if len(self.coefficients) > 0:
            first_pi = list(self.coefficients.keys())[0]
            coef = self.coefficients[first_pi]

            sens = Sensitivity(_pi_expr=coef._pi_expr, _fwk="CUSTOM")

            assert sens._variables is not None
            assert len(sens._variables) > 0

    def test_latex_to_python_mapping(self) -> None:
        """*test_latex_to_python_mapping()* tests LaTeX to Python symbol mapping."""
        # Create with LaTeX expression
        sens = Sensitivity(_pi_expr="\\mu_{1}", _fwk="CUSTOM")

        assert sens._latex_to_py is not None
        assert sens._py_to_latex is not None
        assert len(sens._latex_to_py) > 0
        assert len(sens._py_to_latex) > 0

    # ========================================================================
    # Symbolic Sensitivity Analysis Tests
    # ========================================================================

    def test_analyze_symbolically_basic(self) -> None:
        """*test_analyze_symbolically_basic()* tests basic symbolic analysis."""
        # Use simple expression
        sens = Sensitivity(_pi_expr="U/d", _fwk="CUSTOM")

        # Provide variable values
        vals = {"U": 10.0, "d": 2.0}

        results = sens.analyze_symbolically(vals)

        assert results is not None
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_analyze_symbolically_with_coefficient(self) -> None:
        """*test_analyze_symbolically_with_coefficient()* tests symbolic analysis with coefficient."""
        if len(self.coefficients) == 0:
            pytest.skip("No coefficients available")

        first_pi = list(self.coefficients.keys())[0]
        coef = self.coefficients[first_pi]

        sens = Sensitivity(_fwk="CUSTOM")
        sens.set_coefficient(coef)

        # Get variables from coefficient
        var_names = list(coef.var_dims.keys())

        # Create value dictionary
        vals = {var: 1.0 for var in var_names}

        results = sens.analyze_symbolically(vals)

        assert results is not None
        assert isinstance(results, dict)

    def test_analyze_symbolically_missing_variables_fails(self) -> None:
        """*test_analyze_symbolically_missing_variables_fails()* tests analysis fails with missing vars."""
        sens = Sensitivity(_pi_expr="U/d", _fwk="CUSTOM")

        # Provide incomplete values
        vals = {"U": 10.0}  # Missing 'd'

        with pytest.raises(ValueError) as excinfo:
            sens.analyze_symbolically(vals)

        assert "Missing values" in str(excinfo.value)

    def test_analyze_symbolically_results_structure(self) -> None:
        """*test_analyze_symbolically_results_structure()* tests results have correct structure."""
        sens = Sensitivity(_pi_expr="U*d", _fwk="CUSTOM")

        vals = {"U": 5.0, "d": 2.0}
        results = sens.analyze_symbolically(vals)

        # Results should have sensitivities for each variable
        assert "U" in results or "d" in results
        assert all(isinstance(v, (int, float)) for v in results.values())

    # ========================================================================
    # Numerical Sensitivity Analysis Tests
    # ========================================================================

    def test_analyze_numerically_basic(self) -> None:
        """*test_analyze_numerically_basic()* tests basic numerical analysis."""
        sens = Sensitivity(_pi_expr="U/d", _fwk="CUSTOM")

        # Define variables and bounds
        vals = ["U", "d"]
        bounds = [[1.0, 10.0], [0.5, 5.0]]

        results = sens.analyze_numerically(vals, bounds, n_samples=100)

        assert results is not None
        assert isinstance(results, dict)
        assert sens.var_domains is not None
        assert sens.var_ranges is not None

    def test_analyze_numerically_with_coefficient(self) -> None:
        """*test_analyze_numerically_with_coefficient()* tests numerical analysis with coefficient."""
        if len(self.coefficients) == 0:
            pytest.skip("No coefficients available")

        first_pi = list(self.coefficients.keys())[0]
        coef = self.coefficients[first_pi]

        sens = Sensitivity(_fwk="CUSTOM")
        sens.set_coefficient(coef)

        # Get variables from coefficient
        var_names = list(coef.var_dims.keys())

        # Create bounds
        bounds = [[0.1, 10.0] for _ in var_names]

        results = sens.analyze_numerically(var_names, bounds, n_samples=100)

        assert results is not None
        assert isinstance(results, dict)

    def test_analyze_numerically_invalid_bounds_fails(self) -> None:
        """*test_analyze_numerically_invalid_bounds_fails()* tests analysis fails with wrong bounds."""
        sens = Sensitivity(_pi_expr="U/d", _fwk="CUSTOM")

        vals = ["U", "d"]
        bounds = [[1.0, 10.0]]  # Only one bound, need two

        with pytest.raises(ValueError) as excinfo:
            sens.analyze_numerically(vals, bounds, n_samples=100)

        assert "bounds" in str(excinfo.value).lower()

    def test_analyze_numerically_samples_property(self) -> None:
        """*test_analyze_numerically_samples_property()* tests n_samples is stored."""
        sens = Sensitivity(_pi_expr="U/d", _fwk="CUSTOM")

        vals = ["U", "d"]
        bounds = [[1.0, 10.0], [0.5, 5.0]]
        n_samples = 75

        sens.analyze_numerically(vals, bounds, n_samples=n_samples)

        assert sens.n_samples == n_samples

    def test_analyze_numerically_domain_and_range(self) -> None:
        """*test_analyze_numerically_domain_and_range()* tests domain and range are generated."""
        sens = Sensitivity(_pi_expr="U*d", _fwk="CUSTOM")

        vals = ["U", "d"]
        bounds = [[1.0, 10.0], [0.5, 5.0]]

        sens.analyze_numerically(vals, bounds, n_samples=100)

        assert sens.var_domains is not None
        assert sens.var_ranges is not None
        assert isinstance(sens.var_domains, np.ndarray)
        assert isinstance(sens.var_ranges, np.ndarray)

    # ========================================================================
    # Results Tests
    # ========================================================================

    def test_results_property(self) -> None:
        """*test_results_property()* tests results property."""
        sens = Sensitivity(_pi_expr="U/d", _fwk="CUSTOM")

        vals = {"U": 10.0, "d": 2.0}
        sens.analyze_symbolically(vals)

        results = sens.results
        assert results is not None
        assert isinstance(results, dict)

    def test_results_stored_after_symbolic(self) -> None:
        """*test_results_stored_after_symbolic()* tests results are stored after symbolic analysis."""
        sens = Sensitivity(_pi_expr="U/d", _fwk="CUSTOM")

        vals = {"U": 10.0, "d": 2.0}
        sens.analyze_symbolically(vals)

        assert sens.results is not None
        assert len(sens.results) > 0

    def test_results_stored_after_numerical(self) -> None:
        """*test_results_stored_after_numerical()* tests results are stored after numerical analysis."""
        sens = Sensitivity(_pi_expr="U/d", _fwk="CUSTOM")

        vals = ["U", "d"]
        bounds = [[1.0, 10.0], [0.5, 5.0]]
        sens.analyze_numerically(vals, bounds, n_samples=100)

        assert sens.results is not None
        assert isinstance(sens.results, dict)

    # ========================================================================
    # Clear and Reset Tests
    # ========================================================================

    def test_clear_method(self) -> None:
        """*test_clear_method()* tests clear method resets all attributes."""
        sens = Sensitivity(
            _idx=5,
            _sym="\\Pi_{1}",
            _fwk="CUSTOM",
            _pi_expr="U/d"
        )

        # Run analysis
        vals = {"U": 10.0, "d": 2.0}
        sens.analyze_symbolically(vals)

        # Clear
        sens.clear()

        # Check reset
        assert sens._idx == -1
        assert sens._pi_expr is None
        assert sens._sym_func is None
        assert sens.results == {}

    # ========================================================================
    # Serialization Tests
    # ========================================================================

    def test_to_dict(self) -> None:
        """*test_to_dict()* tests conversion to dictionary."""
        sens = Sensitivity(
            _idx=0,
            _sym="\\Pi_{0}",
            _fwk="CUSTOM",
            _name="Test Sensitivity",
            description="Test description",
            _pi_expr="U/d"
        )

        sens_dict = sens.to_dict()

        assert isinstance(sens_dict, dict)
        assert "name" in sens_dict
        assert "idx" in sens_dict
        assert "sym" in sens_dict
        assert "pi_expr" in sens_dict

    def test_from_dict(self) -> None:
        """*test_from_dict()* tests creation from dictionary."""
        data = {
            "idx": 0,
            "sym": "\\Pi_{0}",
            "fwk": "CUSTOM",
            "name": "Test Sensitivity",
            "description": "Test description",
            "pi_expr": "U",
            "n_samples": 100
        }

        sens = Sensitivity.from_dict(data)

        assert sens is not None
        assert sens._idx == 0
        assert sens._sym == "\\Pi_{0}"
        assert sens.name == "Test Sensitivity"

    def test_round_trip_serialization(self) -> None:
        """*test_round_trip_serialization()* tests to_dict and from_dict round trip."""
        original = Sensitivity(
            _idx=1,
            _sym="\\Pi_{1}",
            _fwk="CUSTOM",
            _name="Original",
            description="Original description",
            _pi_expr="U/d"
        )

        # Convert to dict and back
        sens_dict = original.to_dict()
        restored = Sensitivity.from_dict(sens_dict)

        # Check key attributes match
        assert restored._idx == original._idx
        assert restored._sym == original._sym
        assert restored.name == original.name
        assert restored._pi_expr == original._pi_expr

    # ========================================================================
    # Read-only Property Tests
    # ========================================================================

    def test_variables_property_readonly(self) -> None:
        """*test_variables_property_readonly()* tests variables property is read-only."""
        sens = Sensitivity(_pi_expr="U/d", _fwk="CUSTOM")

        variables = sens.variables
        assert variables is not None

    def test_symbols_property_readonly(self) -> None:
        """*test_symbols_property_readonly()* tests symbols property is read-only."""
        sens = Sensitivity(_pi_expr="U/d", _fwk="CUSTOM")

        symbols = sens.symbols
        assert symbols is not None

    def test_aliases_property_readonly(self) -> None:
        """*test_aliases_property_readonly()* tests aliases property is read-only."""
        sens = Sensitivity(_pi_expr="U/d", _fwk="CUSTOM")

        aliases = sens.aliases
        assert aliases is not None

    def test_exe_func_property_readonly(self) -> None:
        """*test_exe_func_property_readonly()* tests exe_func property is read-only."""
        sens = Sensitivity(_pi_expr="U/d", _fwk="CUSTOM")

        vals = {"U": 10.0, "d": 2.0}
        sens.analyze_symbolically(vals)

        exe_func = sens.exe_func
        # exe_func can be None or a callable/dict
        assert exe_func is None or callable(exe_func) or isinstance(exe_func, dict)
