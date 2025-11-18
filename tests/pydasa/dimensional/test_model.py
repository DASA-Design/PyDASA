# -*- coding: utf-8 -*-
"""
Module test_model.py
===========================================

Unit tests for DimMatrix class in PyDASA.

This module provides comprehensive test coverage for the DimMatrix class, which implements matrix-based dimensional analysis following the Buckingham Pi theorem methodology.

Classes:
    TestDimMatrix: Test cases for DimMatrix class.
"""

# native python modules
import unittest
from typing import Dict
import json

# python third-party modules
import pytest
import numpy as np

# Import the module to test
from pydasa.dimensional.model import DimMatrix
from pydasa.core.parameter import Variable
from pydasa.dimensional.framework import DimSchema
from pydasa.buckingham.vashchy import Coefficient

# Import test data
from tests.pydasa.data.test_data import get_model_test_data

# asserting module imports
assert DimMatrix
assert Variable
assert DimSchema
assert Coefficient
assert get_model_test_data


class TestDimMatrix(unittest.TestCase):
    """Test cases for DimMatrix class.

    This test class provides comprehensive coverage for dimensional matrix  operations including matrix creation, solving, and coefficient generation.
    """

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """Inject test data fixture.

        Sets up test variables and dimensional framework for all tests. Creates a comprehensive set of fluid dynamics variables for testing dimensional analysis operations.
        """
        # Get test data
        self.test_data = get_model_test_data()

        # Setup dimensional framework
        self.test_framework = DimSchema(_fwk="PHYSICAL")
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
                          framework: DimSchema,
                          variables: Dict[str, Variable],
                          name: str = "Test Model",
                          description: str = "Test model for PyDASA",) -> DimMatrix:
        """Create a test DimMatrix model with default or custom parameters.

        Args:
            framework (DimSchema): Dimensional framework. If None, uses test framework.
            variables (Dict[str, Variable]): Variables dictionary. If None, uses relevant variables.
            name (str): Model name. Defaults to "Test Model".
            description (str): Model description.

        Returns:
            DimMatrix: Configured test model.
        """
        variables = self.get_relevant_variables()
        framework = self.test_framework

        return DimMatrix(name=name,
                         description=description,
                         _variables=variables,
                         _framework=framework)

    # ========================================================================
    # Initialization tests
    # ========================================================================

    def test_default_initialization(self) -> None:
        """Test creating DimMatrix with default values."""
        model = DimMatrix()

        assert model is not None
        assert model.name == "Dimensional Matrix"
        assert model.description == ""
        assert isinstance(model._framework, DimSchema)
        assert isinstance(model._variables, dict)
        assert len(model._variables) == 0
        assert isinstance(model._relevant_lt, dict)
        assert len(model._relevant_lt) == 0
        assert model._n_var == 0
        assert model._n_relevant == 0
        assert model._n_in == 0
        assert model._n_out == 0
        assert model._n_ctrl == 0

        expected_mtx = np.array([])
        if model._dim_mtx is None:
            assert expected_mtx is None or expected_mtx.size == 0
        else:
            assert np.array_equal(model._dim_mtx, expected_mtx)

    def test_initialization_with_variables(self) -> None:
        """Test creating DimMatrix with variables."""
        model = self.create_test_model(self.test_framework,
                                       self.test_variables,
                                       name="Fluid Dynamics Model",
                                       description="Test model for fluid flow")

        assert model is not None
        assert model.name == "Fluid Dynamics Model"
        assert model.description == "Test model for fluid flow"
        values = self.test_variables.values()
        relevant_count = len([v for v in values if v.relevant])
        assert len(model._variables) == relevant_count
        assert len(model._relevant_lt) == relevant_count
        assert model._n_relevant == relevant_count

    def test_initialization_with_custom_framework(self) -> None:
        """Test creating DimMatrix with custom framework."""
        comp_framework = DimSchema(_fwk="COMPUTATION")
        comp_framework.update_global_config()

        model = DimMatrix(name="Computation Model",
                          _framework=comp_framework)

        assert model._framework._fwk == "COMPUTATION"
        assert model._framework.fdu_symbols == comp_framework.fdu_symbols

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

        relevant_count = len(relevant_vars)
        assert len(model.variables) == relevant_count
        assert model._n_var == relevant_count
        assert model._n_relevant == relevant_count

    def test_variables_setter_invalid_type(self) -> None:
        """Test variables setter with invalid type."""
        model = DimMatrix()

        with pytest.raises(ValueError) as excinfo:
            model.variables = ["not", "a", "dict"]  # type: ignore
        assert "must be in non-empty dictionary" in str(excinfo.value)

    def test_variables_setter_invalid_values(self) -> None:
        """Test variables setter with invalid variable types."""
        model = DimMatrix()

        with pytest.raises(ValueError) as excinfo:
            model.variables = {"v": "not a variable"}  # type: ignore
        assert "Variable instances" in str(excinfo.value)

    def test_variables_setter_empty_dict(self) -> None:
        """Test variables setter with empty dictionary."""
        model = DimMatrix()

        with pytest.raises(ValueError) as excinfo:
            model.variables = {}
        assert "must be in non-empty dictionary" in str(excinfo.value)

    # ========================================================================
    # Framework management tests
    # ========================================================================

    def test_framework_property_getter(self) -> None:
        """Test framework property getter."""
        model = DimMatrix()
        framework = model.framework

        assert isinstance(framework, DimSchema)
        assert framework._fwk == "PHYSICAL"

    def test_framework_property_setter(self) -> None:
        """Test framework property setter."""
        model = DimMatrix()
        comp_framework = DimSchema(_fwk="COMPUTATION")

        model.framework = comp_framework

        assert model.framework._fwk == "COMPUTATION"

    def test_framework_setter_invalid_type(self) -> None:
        """Test framework setter with invalid type."""
        model = DimMatrix()

        with pytest.raises(ValueError) as excinfo:
            model.framework = "not a framework"     # type: ignore
        assert "DimSchema instance" in str(excinfo.value)

    # ========================================================================
    # Variable statistics tests
    # ========================================================================

    def test_update_variable_stats(self) -> None:
        """Test updating variable statistics."""
        model = self.create_test_model(self.test_framework,
                                       self.test_variables)

        # Count expected values from test data
        relevant_vars = [v for v in self.test_variables.values() if v.relevant]
        expected_in = sum(1 for v in relevant_vars if v.cat == "IN")
        expected_out = sum(1 for v in relevant_vars if v.cat == "OUT")
        expected_ctrl = sum(1 for v in relevant_vars if v.cat == "CTRL")

        assert model._n_var == len(relevant_vars)
        assert model._n_relevant == len(relevant_vars)
        assert model._n_in == expected_in
        assert model._n_out == expected_out
        assert model._n_ctrl == expected_ctrl

    def test_validation_no_output_variable(self) -> None:
        """Test validation fails when no output variable exists."""
        vars_no_out = self.test_data["VARIABLES_NO_OUTPUT"]
        no_out_vars = {k: Variable(**v) for k, v in vars_no_out.items()}

        with pytest.raises(ValueError) as excinfo:
            DimMatrix(_variables=no_out_vars)
        assert "No output variable" in str(excinfo.value)

    def test_validation_too_many_outputs(self) -> None:
        """Test validation fails with too many output variables."""
        vars_multi_out = self.test_data["VARIABLES_MULTI_OUTPUT"]
        multi_out_vars = {k: Variable(**v) for k, v in vars_multi_out.items()}

        with pytest.raises(ValueError) as excinfo:
            DimMatrix(_variables=multi_out_vars)
        assert "Invalid number of outputs" in str(excinfo.value)

    def test_validation_no_input_variables(self) -> None:
        """Test validation fails when no input variables exist."""
        vars_no_input = {
            "P": Variable(
                _sym="P",
                _cat="OUT",
                _dims="M*L^-1*T^-2",
                relevant=True),
        }

        with pytest.raises(ValueError) as excinfo:
            DimMatrix(_variables=vars_no_input)
        assert "No input variables" in str(excinfo.value)

    # ========================================================================
    # Variable sorting tests
    # ========================================================================

    def test_sort_by_category(self) -> None:
        """Test sorting variables by category."""
        model = self.create_test_model(self.test_framework,
                                       self.test_variables)
        # Get sorted variables
        sorted_vars = list(model._relevant_lt.values())

        # Check indices are sequential
        for i, var in enumerate(sorted_vars):
            assert var._idx == i
        # OUT should be the first after all IN,
        assert model.output is not None
        idx = model.output.idx
        assert sorted_vars[idx].cat == "OUT"

    def test_find_output_variable(self) -> None:
        """Test finding the output variable."""
        model = self.create_test_model(self.test_framework,
                                       self.test_variables)

        assert model._output is not None
        assert model._output._sym == "P"
        assert model._output.cat == "OUT"

    # ========================================================================
    # FDU extraction tests
    # ========================================================================

    def test_extract_fdus(self) -> None:
        """Test extracting FDUs from variables."""
        model = self.create_test_model(self.test_framework,
                                       self.test_variables)

        working_fdus = model._extract_fdus()

        assert isinstance(working_fdus, list)
        expected_fdus = self.test_data["EXPECTED_FDU_SYMBOLS"]
        for fdu in expected_fdus:
            assert fdu in working_fdus

    def test_extract_fdus_empty_model(self) -> None:
        """Test extracting FDUs from empty model."""
        model = DimMatrix()

        working_fdus = model._extract_fdus()

        assert isinstance(working_fdus, list)
        assert len(working_fdus) == 0

    # ========================================================================
    # Matrix creation tests
    # ========================================================================

    def test_create_matrix(self) -> None:
        """Test creating dimensional matrix."""
        model = self.create_test_model(self.test_framework,
                                       self.test_variables)

        model.create_matrix()

        assert model._dim_mtx is not None
        assert isinstance(model._dim_mtx, np.ndarray)
        assert model._dim_mtx_trans is not None

        # Check matrix shape
        n_vars = len(model._relevant_lt)
        assert model._dim_mtx.shape[1] == n_vars

    def test_matrix_dimensions(self) -> None:
        """Test dimensional matrix has correct dimensions."""
        model = self.create_test_model(self.test_framework,
                                       self.test_variables)

        model.create_matrix()

        n_fdu = len(model._framework.fdu_symbols)
        n_var = len(model._relevant_lt)

        assert model._dim_mtx.shape == (n_fdu, n_var)           # type: ignore
        assert model._dim_mtx_trans.shape == (n_var, n_fdu)     # type: ignore

    def test_create_matrix_empty_model(self) -> None:
        """Test creating matrix with empty model raises error."""
        model = DimMatrix()

        with pytest.raises(ValueError) as excinfo:
            model.create_matrix()
        assert "No relevant variables" in str(excinfo.value)

    # ========================================================================
    # Matrix solving tests
    # ========================================================================

    def test_solve_matrix(self) -> None:
        """Test solving dimensional matrix."""
        model = self.create_test_model(self.test_framework,
                                       self.test_variables)

        model.create_matrix()
        model.solve_matrix()

        assert model._rref_mtx is not None
        assert isinstance(model._rref_mtx, np.ndarray)
        assert isinstance(model._pivot_cols, list)
        assert len(model._coefficients) > 0

    # ========================================================================
    # Coefficient derivation tests
    # ========================================================================

    def test_derive_coefficient(self) -> None:
        """Test deriving new coefficient from existing ones."""
        model = self.create_test_model(self.test_framework,
                                       self.test_variables)
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
        model = self.create_test_model(self.test_framework,
                                       self.test_variables)
        model.solve_matrix()

        with pytest.raises(ValueError) as excinfo:
            model.derive_coefficient(expr="invalid * expression")
        assert "does not contain any valid" in str(excinfo.value)

    def test_derive_coefficient_nonexistent_reference(self) -> None:
        """Test deriving coefficient with nonexistent reference."""
        model = self.create_test_model(self.test_framework,
                                       self.test_variables)
        model.solve_matrix()

        with pytest.raises(ValueError) as excinfo:
            model.derive_coefficient(expr="\\Pi_{999} * \\Pi_{1000}")
        assert "does not exist" in str(excinfo.value)

    def test_derive_coefficient_no_base_coefficients(self) -> None:
        """Test deriving coefficient when no base coefficients exist."""
        model = self.create_test_model(self.test_framework,
                                       self.test_variables)
        # Don't solve matrix first
        with pytest.raises(ValueError) as excinfo:
            model.derive_coefficient(expr="\\Pi_{0} * \\Pi_{1}")
        assert "No base coefficients exist" in str(excinfo.value)

    # ========================================================================
    # Complete analysis tests
    # ========================================================================

    def test_analyze_complete_workflow(self) -> None:
        """Test complete dimensional analysis workflow."""
        model = self.create_test_model(self.test_framework,
                                       self.test_variables)
        model.analyze()

        # Check all stages completed
        assert model._dim_mtx is not None
        assert model._rref_mtx is not None
        assert len(model._coefficients) > 0
        assert model._output is not None

    def test_analyze_empty_model(self) -> None:
        """Test analyze fails on empty model."""
        model = DimMatrix()
        
        with pytest.raises(ValueError):
            model.analyze()

    # ========================================================================
    # Property getter tests
    # ========================================================================

    def test_relevant_lt_property(self) -> None:
        """Test relevant_lt property getter."""
        model = self.create_test_model(self.test_framework,
                                       self.test_variables)
        relevant_list = model.relevant_lt

        assert isinstance(relevant_list, dict)
        relevant_count = len([v for v in self.test_variables.values() if v.relevant])
        assert len(relevant_list) == relevant_count

    def test_coefficients_property(self) -> None:
        """Test coefficients property getter."""
        model = self.create_test_model(self.test_framework,
                                       self.test_variables)
        model.analyze()
        coefficients = model.coefficients

        assert isinstance(coefficients, dict)
        assert all(isinstance(c, Coefficient) for c in coefficients.values())

    def test_output_property(self) -> None:
        """Test output property getter."""
        model = self.create_test_model(self.test_framework,
                                       self.test_variables)
        output = model.output

        assert output is not None
        assert isinstance(output, Variable)
        assert output.cat == "OUT"

    def test_dim_mtx_property(self) -> None:
        """Test dim_mtx property getter."""
        model = self.create_test_model(self.test_framework,
                                       self.test_variables)
        model.create_matrix()
        dim_mtx = model.dim_mtx

        assert dim_mtx is not None
        assert isinstance(dim_mtx, np.ndarray)

    def test_rref_mtx_property(self) -> None:
        """Test rref_mtx property getter."""
        model = self.create_test_model(self.test_framework,
                                       self.test_variables)
        model.solve_matrix()
        rref = model.rref_mtx

        assert rref is not None
        assert isinstance(rref, np.ndarray)

    def test_pivot_cols_property(self) -> None:
        """Test pivot_cols property getter."""
        model = self.create_test_model(self.test_framework,
                                       self.test_variables)
        model.solve_matrix()
        pivots = model.pivot_cols

        assert isinstance(pivots, list)
        assert all(isinstance(p, int) for p in pivots)

    # ========================================================================
    # Clear method tests
    # ========================================================================

    def test_clear_resets_all(self) -> None:
        """Test clear method resets all attributes."""
        model = self.create_test_model(self.test_framework,
                                       self.test_variables)
        model.analyze()
        model.clear()

        assert len(model._variables) == 0
        assert len(model._relevant_lt) == 0
        assert model._n_var == 0
        assert model._n_relevant == 0
        assert model._dim_mtx is None
        assert model._rref_mtx is None
        assert len(model._coefficients) == 0

    def test_clear_preserves_framework(self) -> None:
        """Test clear preserves framework."""
        model = self.create_test_model(self.test_framework,
                                       self.test_variables)
        original_fwk = model._framework._fwk

        model.analyze()
        model.clear()

        assert model._framework._fwk == original_fwk

    # ========================================================================
    # Serialization tests
    # ========================================================================

    def test_to_dict_structure(self) -> None:
        """Test to_dict method returns correct structure."""
        model = self.create_test_model(self.test_framework,
                                       self.test_variables,
                                       name="Fluid Test Model",
                                       description="Test model for fluid flow")

        model.analyze()
        result = model.to_dict()

        assert isinstance(result, dict)
        assert "name" in result
        assert "variables" in result
        assert "coefficients" in result
        assert "n_var" in result
        assert result["name"] == "Fluid Test Model"

    # def test_to_dict_json_serializable(self) -> None:
    #     """Test to_dict result is JSON serializable."""
    #     # TODO not priorityt, do it later!
    #     model = self.create_test_model(self.test_framework,
    #                                    self.test_variables)
    #     model.analyze()
    #     result = model.to_dict()

    #     # Should not raise exception
    #     json_str = json.dumps(result, indent=2)
    #     assert isinstance(json_str, str)
    #     assert len(json_str) > 0

    def test_from_dict_roundtrip(self) -> None:
        """Test from_dict restores model correctly."""
        model1 = self.create_test_model(self.test_framework,
                                        self.test_variables,
                                        name="Original Model")

        # Convert to dict
        model_dict = model1.to_dict()

        # Recreate from dict
        model2 = DimMatrix.from_dict(model_dict)

        # Verify basic properties
        assert model2.name == model1.name
        assert model2._fwk == model1._fwk
        assert len(model2._variables) == len(model1._variables)

    def test_from_dict_with_underscore_keys(self) -> None:
        """Test from_dict handles underscore prefixes correctly."""
        data = {
            "_idx": 5,
            "_sym": "DM",
            "name": "Test",
            "_fwk": "PHYSICAL"
        }

        model = DimMatrix.from_dict(data)

        assert model._idx == 5
        assert model._sym == "DM"
        assert model.name == "Test"
        assert model._fwk == "PHYSICAL"

    # ========================================================================
    # Edge case tests
    # ========================================================================

    def test_empty_model(self) -> None:
        """Test operations on empty model."""
        model = DimMatrix()

        expected_mtx = np.array([])
        if model._dim_mtx is None:
            assert expected_mtx is None or expected_mtx.size == 0
        else:
            assert np.array_equal(model._dim_mtx, expected_mtx)
        assert len(model._coefficients) == 0
        assert model._output is None

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

    def test_model_with_irrelevant_variables(self) -> None:
        """Test model correctly filters irrelevant variables."""
        model = DimMatrix(_variables=self.test_variables)

        # Should only count relevant variables
        relevant_count = len([v for v in self.test_variables.values() if v.relevant])
        assert model._n_relevant == relevant_count
        assert len(model._relevant_lt) == relevant_count

        # All variables in relevant_lt should have relevant=True
        for var in model._relevant_lt.values():
            assert var.relevant is True
