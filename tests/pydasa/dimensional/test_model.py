# -*- coding: utf-8 -*-
"""
Module test_model.py
===========================================

Unit tests for DimMatrix class in PyDASA.

This module provides comprehensive test coverage for the DimMatrix class,
which implements matrix-based dimensional analysis following the 
Buckingham Pi theorem methodology.

Classes:
    TestDimMatrix: Test cases for DimMatrix class.
"""

# native python modules
import unittest
from typing import Dict

# python third-party modules
import pytest
import numpy as np

# Import the module to test
from pydasa.dimensional.model import DimMatrix
from pydasa.core.parameter import Variable
from pydasa.dimensional.framework import DimScheme
from pydasa.buckingham.vashchy import Coefficient

# Import test data
from tests.pydasa.data.test_data import get_model_test_data

# asserting module imports
assert DimMatrix
assert Variable
assert DimScheme
assert Coefficient
assert get_model_test_data


class TestDimMatrix(unittest.TestCase):
    """Test cases for DimMatrix class.

    This test class provides comprehensive coverage for dimensional matrix
    operations including matrix creation, solving, and coefficient generation.
    """

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """Inject test data fixture.

        Sets up test variables and dimensional framework for all tests.
        Creates a comprehensive set of fluid dynamics variables for testing
        dimensional analysis operations.
        """
        # Get test data
        self.test_data = get_model_test_data()

        # Setup dimensional framework
        self.test_framework = DimScheme(_fwk="PHYSICAL")
        self.test_framework.update_global_config()

        # Create test variables from test data
        self.test_variables = {}
        for var_sym, var_data in self.test_data["TEST_VARIABLES"].items():
            self.test_variables[var_sym] = Variable(**var_data)

    # ========================================================================
    # Helper methods
    # ========================================================================

    def get_relevant_variables(self) -> Dict[str, Variable]:
        """Get only relevant variables from test variables.

        Returns:
            Dict[str, Variable]: Dictionary of relevant variables.
        """
        return {k: v for k, v in self.test_variables.items() if v.relevant}

    def create_test_model(self,
                          name: str = "Test Model",
                          description: str = "Test model for DA.",
                          variables: Dict[str, Variable] = None,
                          framework: DimScheme = None) -> DimMatrix:
        """Create a test DimMatrix model with default or custom parameters.

        Args:
            name (str): Model name. Defaults to "Test Model".
            description (str): Model description.
            variables (Dict[str, Variable]): Variables dictionary. If None, uses relevant variables.
            framework (DimScheme): Dimensional framework. If None, uses test framework.

        Returns:
            DimMatrix: Configured test model.
        """
        if variables is None:
            variables = self.get_relevant_variables()
        if framework is None:
            framework = self.test_framework

        matrix = DimMatrix(name=name,
                           description=description,
                           _variables=variables,
                           _framework=framework)
        return matrix

    # ========================================================================
    # Initialization tests
    # ========================================================================

    def test_default_initialization(self) -> None:
        """Test creating DimMatrix with default values."""
        model = DimMatrix()

        assert model is not None
        assert model.name == "Dimensional Matrix"
        assert model.description == ""
        assert isinstance(model._framework, DimScheme)
        assert isinstance(model._variables, dict)
        assert len(model._variables) == 0
        assert isinstance(model._relevant_lt, dict)
        assert len(model._relevant_lt) == 0
        assert model._dim_mtx is None
        assert model._n_var == 0
        assert model._n_relevant == 0
        assert model._n_in == 0
        assert model._n_out == 0
        assert model._n_ctrl == 0

    def test_initialization_with_variables(self) -> None:
        """Test creating DimMatrix with variables."""
        model = self.create_test_model(
            name="Fluid Dynamics Model",
            description="Test model for fluid flow"
        )

        assert model is not None
        assert model.name == "Fluid Dynamics Model"
        assert len(model._variables) == 7  # All relevant variables from test data
        assert len(model._relevant_lt) == 7
        assert model._n_relevant == 7

    # ========================================================================
    # Variable management tests
    # ========================================================================

    def test_variables_property_getter(self) -> None:
        """Test variables property getter."""
        model = DimMatrix()
        variables = model.variables

        assert isinstance(variables, dict)
        assert len(variables) == 0

    def test_variables_property_setter(self) -> None:
        """Test variables property setter."""
        model = DimMatrix()
        relevant_vars = self.get_relevant_variables()
        
        model.variables = relevant_vars
        
        assert len(model.variables) == 7
        assert model._n_var == 7
        assert model._n_relevant == 7

    def test_variables_setter_invalid_type(self) -> None:
        """Test variables setter with invalid type."""
        model = DimMatrix()
        
        with pytest.raises(ValueError) as excinfo:
            model.variables = ["not", "a", "dict"]
        assert "must be in non-empty dictionary" in str(excinfo.value)

    def test_variables_setter_invalid_values(self) -> None:
        """Test variables setter with invalid variable types."""
        model = DimMatrix()
        
        with pytest.raises(ValueError) as excinfo:
            model.variables = {"v": "not a variable"}
        assert "Variable instances" in str(excinfo.value)

    # ========================================================================
    # Framework management tests
    # ========================================================================

    def test_framework_property_getter(self) -> None:
        """Test framework property getter."""
        model = DimMatrix()
        framework = model.framework
        
        assert isinstance(framework, DimScheme)
        assert framework._fwk == "PHYSICAL"

    def test_framework_property_setter(self) -> None:
        """Test framework property setter."""
        model = DimMatrix()
        comp_framework = DimScheme(_fwk="COMPUTATION")
        
        model.framework = comp_framework
        
        assert model.framework._fwk == "COMPUTATION"

    def test_framework_setter_invalid_type(self) -> None:
        """Test framework setter with invalid type."""
        model = DimMatrix()
        
        with pytest.raises(ValueError) as excinfo:
            model.framework = "not a framework"
        assert "DimScheme instance" in str(excinfo.value)

    # ========================================================================
    # Variable statistics tests
    # ========================================================================

    def test_update_variable_stats(self) -> None:
        """Test updating variable statistics."""
        model = self.create_test_model()
        
        assert model._n_var == 7
        assert model._n_relevant == 7
        assert model._n_in == 4  # v, L, rho, g
        assert model._n_out == 1  # P
        assert model._n_ctrl == 2  # mu, nu

    def test_validation_no_output_variable(self) -> None:
        """Test validation fails when no output variable exists."""
        vars_no_output = {
            k: v for k, v in self.test_data["VARIABLES_NO_OUTPUT"].items()
        }
        no_out_vars = {k: Variable(**v) for k, v in vars_no_output.items()}
        
        with pytest.raises(ValueError) as excinfo:
            DimMatrix(_variables=no_out_vars)
        assert "No output variable" in str(excinfo.value)

    def test_validation_too_many_outputs(self) -> None:
        """Test validation fails with too many output variables."""
        vars_multi_output = {
            k: v for k, v in self.test_data["VARIABLES_MULTI_OUTPUT"].items()
        }
        multi_out_vars = {k: Variable(**v) for k, v in vars_multi_output.items()}
        
        with pytest.raises(ValueError) as excinfo:
            DimMatrix(_variables=multi_out_vars)
        assert "Invalid number of outputs" in str(excinfo.value)

    # ========================================================================
    # Variable sorting tests
    # ========================================================================

    def test_sort_by_category(self) -> None:
        """Test sorting variables by category."""
        model = self.create_test_model()
        
        # Get sorted variables
        sorted_vars = list(model._relevant_lt.values())
        
        # First should be OUT
        assert sorted_vars[0].cat == "OUT"  # P

    def test_find_output_variable(self) -> None:
        """Test finding the output variable."""
        model = self.create_test_model()
        
        assert model._output is not None
        assert model._output._sym == "P"
        assert model._output.cat == "OUT"

    # ========================================================================
    # FDU extraction tests
    # ========================================================================

    def test_extract_fdus(self) -> None:
        """Test extracting FDUs from variables."""
        model = self.create_test_model()
        
        working_fdus = model._extract_fdus()
        
        assert isinstance(working_fdus, list)
        # Should contain M, L, T from the variable dimensions
        assert "M" in working_fdus
        assert "L" in working_fdus
        assert "T" in working_fdus

    # ========================================================================
    # Matrix creation tests
    # ========================================================================

    def test_create_matrix(self) -> None:
        """Test creating dimensional matrix."""
        model = self.create_test_model()
        
        model.create_matrix()
        
        assert model._dim_mtx is not None
        assert isinstance(model._dim_mtx, np.ndarray)
        assert model._dim_mtx.shape[1] == 7  # 7 relevant variables
        assert model._dim_mtx_trans is not None

    def test_matrix_dimensions(self) -> None:
        """Test dimensional matrix has correct dimensions."""
        model = self.create_test_model()
        
        model.create_matrix()
        
        n_fdu = len(model._framework.fdu_symbols)
        n_var = len(model._relevant_lt)
        
        assert model._dim_mtx.shape == (n_fdu, n_var)

    # ========================================================================
    # Matrix solving tests
    # ========================================================================

    def test_solve_matrix(self) -> None:
        """Test solving dimensional matrix."""
        model = self.create_test_model()
        
        model.solve_matrix()
        
        assert model._rref_mtx is not None
        assert isinstance(model._rref_mtx, np.ndarray)
        assert isinstance(model._pivot_cols, list)
        assert len(model._coefficients) > 0

    def test_generate_coefficients(self) -> None:
        """Test generating dimensionless coefficients."""
        model = self.create_test_model()
        
        model.solve_matrix()
        
        assert len(model._coefficients) > 0
        # Check first coefficient
        first_coef = list(model._coefficients.values())[0]
        assert isinstance(first_coef, Coefficient)
        assert first_coef._idx >= 0
        assert first_coef._sym.startswith("\\Pi_")

    # ========================================================================
    # Coefficient derivation tests
    # ========================================================================

    def test_derive_coefficient(self) -> None:
        """Test deriving new coefficient from existing ones."""
        model = self.create_test_model()
        
        model.solve_matrix()
        
        # Get two base coefficients
        coef_syms = list(model._coefficients.keys())
        if len(coef_syms) >= 2:
            expr = f"{coef_syms[0]} * {coef_syms[1]}"
            
            derived = model.derive_coefficient(
                expr=expr,
                name="Test Derived",
                description="Test derived coefficient"
            )
            
            assert derived is not None
            assert derived.cat == "DERIVED"
            assert derived.name == "Test Derived"
            assert derived._sym in model._coefficients

    def test_derive_coefficient_invalid_expression(self) -> None:
        """Test deriving coefficient with invalid expression."""
        model = self.create_test_model()
        
        model.solve_matrix()
        
        with pytest.raises(ValueError) as excinfo:
            model.derive_coefficient(expr="invalid * expression")
        assert "does not contain any valid" in str(excinfo.value)

    def test_derive_coefficient_nonexistent_reference(self) -> None:
        """Test deriving coefficient with nonexistent reference."""
        model = self.create_test_model()
        
        model.solve_matrix()
        
        with pytest.raises(ValueError) as excinfo:
            model.derive_coefficient(expr="\\Pi_{999} * \\Pi_{1000}")
        assert "does not exist" in str(excinfo.value)

    # ========================================================================
    # Complete analysis tests
    # ========================================================================

    def test_analyze_complete_workflow(self) -> None:
        """Test complete dimensional analysis workflow."""
        model = self.create_test_model()
        
        model.analyze()
        
        # Check all stages completed
        assert model._dim_mtx is not None
        assert model._rref_mtx is not None
        assert len(model._coefficients) > 0
        assert model._output is not None

    # ========================================================================
    # Property getter tests
    # ========================================================================

    def test_relevant_lt_property(self) -> None:
        """Test relevant_lt property getter."""
        model = self.create_test_model()
        
        relevant_list = model.relevant_lt
        
        assert isinstance(relevant_list, dict)
        assert len(relevant_list) == 7

    def test_coefficients_property(self) -> None:
        """Test coefficients property getter."""
        model = self.create_test_model()
        
        model.analyze()
        coefficients = model.coefficients
        
        assert isinstance(coefficients, dict)
        assert all(isinstance(c, Coefficient) for c in coefficients.values())

    def test_output_property(self) -> None:
        """Test output property getter."""
        model = self.create_test_model()
        
        output = model.output
        
        assert output is not None
        assert isinstance(output, Variable)
        assert output.cat == "OUT"

    def test_dim_mtx_property(self) -> None:
        """Test dim_mtx property getter."""
        model = self.create_test_model()
        
        model.create_matrix()
        dim_mtx = model.dim_mtx
        
        assert dim_mtx is not None
        assert isinstance(dim_mtx, np.ndarray)

    def test_rref_mtx_property(self) -> None:
        """Test rref_mtx property getter."""
        model = self.create_test_model()
        
        model.solve_matrix()
        rref = model.rref_mtx
        
        assert rref is not None
        assert isinstance(rref, np.ndarray)

    def test_pivot_cols_property(self) -> None:
        """Test pivot_cols property getter."""
        model = self.create_test_model()
        
        model.solve_matrix()
        pivots = model.pivot_cols
        
        assert isinstance(pivots, list)
        assert all(isinstance(p, int) for p in pivots)

    # ========================================================================
    # Clear method tests
    # ========================================================================

    def test_clear_resets_all(self) -> None:
        """Test clear method resets all attributes."""
        model = self.create_test_model()
        
        model.analyze()
        model.clear()
        
        assert len(model._variables) == 0
        assert len(model._relevant_lt) == 0
        assert model._n_var == 0
        assert model._n_relevant == 0
        assert model._dim_mtx is None
        assert model._rref_mtx is None
        assert len(model._coefficients) == 0

    # ========================================================================
    # Serialization tests
    # ========================================================================

    def test_to_dict_structure(self) -> None:
        """Test to_dict method returns correct structure."""
        model = self.create_test_model(name="Test Model")
        
        model.analyze()
        result = model.to_dict()
        
        assert isinstance(result, dict)
        assert "name" in result
        assert "variables" in result
        assert "coefficients" in result
        assert "n_var" in result
        assert result["name"] == "Test Model"

    # ========================================================================
    # Edge case tests
    # ========================================================================

    def test_empty_model(self) -> None:
        """Test operations on empty model."""
        model = DimMatrix()
        
        assert model._dim_mtx is None
        assert len(model._coefficients) == 0

    def test_minimal_model(self) -> None:
        """Test model with minimum required variables."""
        minimal_data = self.test_data["MINIMAL_VARIABLES"]
        minimal_vars = {k: Variable(**v) for k, v in minimal_data.items()}
        
        model = DimMatrix(_variables=minimal_vars)
        
        assert model._n_in == 1
        assert model._n_out == 1
        assert model._n_relevant == 2

    def test_large_variable_set(self) -> None:
        """Test model with many variables."""
        # Create 10 input variables
        large_vars = {}
        for i in range(10):
            large_vars[f"v{i}"] = Variable(
                _sym=f"v{i}",
                _cat="IN",
                _dims="L*T^-1",
                relevant=True,
                _idx=i
            )
        
        # Add one output
        large_vars["P"] = Variable(
            _sym="P",
            _cat="OUT",
            _dims="M*L^-1*T^-2",
            relevant=True,
            _idx=10
        )
        
        # Should raise error for too many inputs
        with pytest.raises(ValueError) as excinfo:
            DimMatrix(_variables=large_vars)
        assert "Too many input variables" in str(excinfo.value)
