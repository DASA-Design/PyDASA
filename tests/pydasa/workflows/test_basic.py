# -*- coding: utf-8 -*-
"""
Module test_basic.py
===========================================

Tests for **WorkflowBase** in *PyDASA*.

This module provides unit tests for the base workflow class that provides common functionality across all workflow orchestrators.
"""
# import testing package
import unittest
import pytest

# import the module to test
from pydasa.workflows.basic import WorkflowBase

# import required classes
from pydasa.elements.parameter import Variable
from pydasa.dimensional.buckingham import Coefficient
from pydasa.dimensional.vaschy import Schema

# import test data
from tests.pydasa.data.test_data import get_simulation_test_data

# asserting module imports
assert WorkflowBase
assert get_simulation_test_data


class TestWorkflowBase(unittest.TestCase):
    """**TestWorkflowBase** implements unit tests for workflow base class.

    Args:
        unittest (TestCase): unittest.TestCase class for Python unit testing.
    """

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """*inject_fixtures()* injects global test parameters as fixture."""
        # Get test data
        self.test_data = get_simulation_test_data()
        assert self.test_data is not None, "Test data loading failed"
        assert "FDU_LIST" in self.test_data, "Test data missing FDU_LIST"
        assert "CHANNEL_FLOW_VARIABLES" in self.test_data, "Test data missing CHANNEL_FLOW_VARIABLES"

        # Setup dimensional schema
        self.dim_schema = Schema(_fwk="CUSTOM",
                                 _fdu_lt=self.test_data["FDU_LIST"],
                                 _idx=0)
        assert self.dim_schema is not None, "Schema initialization failed"
        self.dim_schema._setup_fdus()

        # Setup variables
        self.test_variables = {}
        for sym, var_data in self.test_data["CHANNEL_FLOW_VARIABLES"].items():
            var = Variable(**var_data)
            assert var is not None, f"Variable creation failed for {sym}"
            # Set schema and prepare dimensions for CUSTOM framework variables
            var._schema = self.dim_schema
            var._prepare_dims()
            self.test_variables[sym] = var

        # Setup mock coefficients (for testing)
        self.test_coefficients = {
            "\\Pi_{0}": Coefficient(
                _idx=0,
                _sym="\\Pi_{0}",
                _alias="Pi_0",
                _fwk="CUSTOM",
                _name="Reynolds Number",
                description="Dimensionless Reynolds coefficient"
            ),
            "\\Pi_{1}": Coefficient(
                _idx=1,
                _sym="\\Pi_{1}",
                _alias="Pi_1",
                _fwk="CUSTOM",
                _name="Geometric Ratio",
                description="Dimensionless geometric coefficient"
            )
        }
        assert len(self.test_coefficients) > 0, "Coefficients initialization failed"

    def test_schema_string_assignment(self) -> None:
        """*test_schema_string_assignment()* tests setting schema from string."""
        # Create workflow base instance
        workflow = WorkflowBase()

        # Set schema from string
        workflow.schema = "COMPUTATION"

        # Test schema converted
        assert workflow.schema is not None

        # Store in local variable for type narrowing
        schema = workflow.schema
        assert isinstance(schema, Schema)
        assert schema._fwk == "COMPUTATION"

    def test_schema_dict_assignment(self) -> None:
        """*test_schema_dict_assignment()* tests setting schema from dict."""
        # Create workflow base instance
        workflow = WorkflowBase()

        # Create a simple schema dict using PHYSICAL framework (has default FDUs)
        schema_dict = {
            "_fwk": "PHYSICAL",
            "_idx": 0
        }

        # Set schema from dict
        workflow.schema = schema_dict

        # Test schema converted
        assert workflow.schema is not None

        # Test schema converted
        schema = workflow.schema
        assert isinstance(schema, Schema)
        assert schema._fwk == "PHYSICAL"

    def test_schema_object_assignment(self) -> None:
        """*test_schema_object_assignment()* tests setting schema from Schema object."""
        # Create workflow base instance
        workflow = WorkflowBase()

        # Set schema from object
        workflow.schema = self.dim_schema

        # Test schema converted
        assert workflow.schema is not None

        # Test schema assigned
        schema = workflow.schema
        assert isinstance(schema, Schema)
        assert schema._fwk == "CUSTOM"

    def test_variables_dict_assignment(self) -> None:
        """*test_variables_dict_assignment()* tests setting variables from dict."""
        # Create workflow base instance
        workflow = WorkflowBase()

        # Set variables
        workflow.variables = self.test_variables

        # Test variables set
        assert len(workflow.variables) == len(self.test_variables)
        for sym in self.test_variables.keys():
            assert sym in workflow.variables
            assert isinstance(workflow.variables[sym], Variable)

    def test_variables_conversion_from_dicts(self) -> None:
        """*test_variables_conversion_from_dicts()* tests converting dict values to Variables."""
        # Create workflow base instance
        workflow = WorkflowBase()

        # Create dict with dict values
        var_dicts = {}
        for sym, var in self.test_variables.items():
            var_dicts[sym] = var.to_dict()

        # Set variables from dicts
        workflow.variables = var_dicts

        # Test conversion happened
        assert len(workflow.variables) == len(var_dicts)
        for sym in var_dicts.keys():
            assert isinstance(workflow.variables[sym], Variable)

    def test_variables_invalid_type(self) -> None:
        """*test_variables_invalid_type()* tests error on invalid variable type."""
        # Create workflow base instance
        workflow = WorkflowBase()

        # Try to set invalid variables
        with pytest.raises(ValueError) as excinfo:
            workflow.variables = {"key": "invalid_string"}
        assert "values must be Variable or dict" in str(excinfo.value)

    def test_coefficients_property_setter(self) -> None:
        """*test_coefficients_property_setter()* tests coefficients property setter."""
        # Create workflow base instance
        workflow = WorkflowBase()

        # Set coefficients
        workflow.coefficients = self.test_coefficients

        # Test coefficients set
        assert len(workflow.coefficients) == len(self.test_coefficients)
        for sym in self.test_coefficients.keys():
            assert sym in workflow.coefficients

    def test_coefficients_conversion_from_dicts(self) -> None:
        """*test_coefficients_conversion_from_dicts()* tests converting dict values to Coefficients."""
        # Create workflow base instance
        workflow = WorkflowBase()

        # Create dict with dict values
        coef_dicts = {}
        for sym, coef in self.test_coefficients.items():
            coef_dicts[sym] = coef.to_dict()

        # Set coefficients from dicts
        workflow.coefficients = coef_dicts

        # Test conversion happened
        assert len(workflow.coefficients) == len(coef_dicts)
        for sym in coef_dicts.keys():
            assert isinstance(workflow.coefficients[sym], Coefficient)

    def test_reset(self) -> None:
        """*test_reset()* tests resetting workflow state."""
        # Create workflow with data
        workflow = WorkflowBase()
        workflow.variables = self.test_variables
        workflow._coefficients = self.test_coefficients
        workflow._results = {"key": {"data": "value"}}
        workflow._is_solved = True

        # Reset workflow
        workflow.reset()

        # Test state cleared
        assert len(workflow.coefficients) == 0
        assert len(workflow.results) == 0
        assert workflow.is_solved is False
        # Test variables preserved
        assert len(workflow.variables) > 0

    def test_clear(self) -> None:
        """*test_clear()* tests clearing all workflow data."""
        # Create workflow with data
        workflow = WorkflowBase()
        workflow.variables = self.test_variables
        workflow._coefficients = self.test_coefficients
        workflow._results = {"key": {"data": "value"}}
        workflow._is_solved = True

        # Clear workflow
        workflow.clear()

        # Test everything cleared
        assert len(workflow.variables) == 0
        assert len(workflow.coefficients) == 0
        assert len(workflow.results) == 0
        assert workflow.is_solved is False
        # Note: _schema is intentionally not cleared in WorkflowBase.clear()

    def test_properties_are_copies(self) -> None:
        """*test_properties_are_copies()* tests that properties return copies."""
        # Create workflow
        workflow = WorkflowBase()
        workflow.variables = self.test_variables

        # Get variables
        vars1 = workflow.variables
        vars2 = workflow.variables

        # Test they are different objects (copies)
        assert vars1 is not vars2
        assert id(vars1) != id(vars2)

    def test_to_dict(self) -> None:
        """*test_to_dict()* tests converting workflow to dictionary."""
        # Create workflow
        workflow = WorkflowBase()
        workflow.variables = self.test_variables
        workflow.schema = self.dim_schema
        workflow._coefficients = self.test_coefficients

        # Convert to dict
        data = workflow.to_dict()

        # Test dictionary structure
        assert isinstance(data, dict)
        assert "variables" in data
        assert "coefficients" in data
        assert "is_solved" in data
        assert "schema" in data
        # Test variables converted to dicts
        assert isinstance(data["variables"], dict)
        # Test schema converted to dict
        assert isinstance(data["schema"], dict)

    def test_from_dict(self) -> None:
        """*test_from_dict()* tests creating workflow from dictionary."""
        # Create workflow and convert to dict
        workflow1 = WorkflowBase()
        workflow1.variables = self.test_variables
        workflow1.schema = self.dim_schema
        workflow1._coefficients = self.test_coefficients
        data = workflow1.to_dict()

        # Create new workflow from dict
        workflow2 = WorkflowBase.from_dict(data)

        # Test workflow created correctly
        assert workflow2.schema is not None
        assert workflow1.schema is not None
        assert len(workflow2.variables) == len(workflow1.variables)
        assert len(workflow2.coefficients) == len(workflow1.coefficients)

        schema2 = workflow2.schema
        schema1 = workflow1.schema
        assert isinstance(schema2, Schema) and isinstance(schema1, Schema)
        assert schema2._fwk == schema1._fwk

    def test_is_solved_property(self) -> None:
        """*test_is_solved_property()* tests is_solved property."""
        # Create workflow
        workflow = WorkflowBase()

        # Test initial state
        assert workflow.is_solved is False

        # Set solved state
        workflow._is_solved = True
        assert workflow.is_solved is True

    def test_variables_setter_resets_solved_state(self) -> None:
        """*test_variables_setter_resets_solved_state()* tests setting variables resets solved state."""
        # Create workflow
        workflow = WorkflowBase()
        workflow._is_solved = True

        # Set variables
        workflow.variables = self.test_variables

        # Test solved state reset
        assert workflow.is_solved is False

    def test_schema_setter_resets_solved_state(self) -> None:
        """*test_schema_setter_resets_solved_state()* tests setting schema resets solved state."""
        # Create workflow
        workflow = WorkflowBase()
        workflow._is_solved = True

        # Set schema
        workflow.schema = "COMPUTATION"

        # Test solved state reset
        assert workflow.is_solved is False

    def test_results_property(self) -> None:
        """*test_results_property()* tests results property getter."""
        # Create workflow
        workflow = WorkflowBase()
        workflow._results = {"test": {"value": 123}}

        # Get results
        results = workflow.results

        # Test results returned correctly
        assert isinstance(results, dict)
        assert "test" in results
        assert results["test"]["value"] == 123

    def test_convert_to_objects_with_variables(self) -> None:
        """*test_convert_to_objects_with_variables()* tests _convert_to_objects helper for Variables."""
        # Create workflow
        workflow = WorkflowBase()

        # Create mixed dict (Variable objects and dicts)
        mixed_data = {}
        for i, (sym, var) in enumerate(self.test_variables.items()):
            if i % 2 == 0:
                mixed_data[sym] = var  # Variable object
            else:
                mixed_data[sym] = var.to_dict()  # dict

        # Convert to objects
        result = workflow._convert_to_objects(mixed_data, Variable)

        # Test all values are Variable instances
        assert len(result) == len(mixed_data)
        for val in result.values():
            assert isinstance(val, Variable)

    def test_convert_to_schema_from_string(self) -> None:
        """*test_convert_to_schema_from_string()* tests _convert_to_schema helper with string input."""
        # Create workflow
        workflow = WorkflowBase()

        # Convert string to schema
        schema = workflow._convert_to_schema("PHYSICAL")

        # Test schema created
        assert isinstance(schema, Schema)
        assert schema._fwk == "PHYSICAL"

    def test_convert_to_schema_from_dict(self) -> None:
        """*test_convert_to_schema_from_dict()* tests _convert_to_schema helper with dict input."""
        # Create workflow
        workflow = WorkflowBase()

        # Create schema dict
        schema_dict = {"_fwk": "COMPUTATION", "_idx": 0}

        # Convert dict to schema
        schema = workflow._convert_to_schema(schema_dict)

        # Test schema created
        assert isinstance(schema, Schema)
        assert schema._fwk == "COMPUTATION"

    def test_convert_to_schema_from_list(self) -> None:
        """*test_convert_to_schema_from_list()* tests _convert_to_schema helper with list input."""
        # Create workflow
        workflow = WorkflowBase()

        # Convert FDU list to schema
        schema = workflow._convert_to_schema(self.test_data["FDU_LIST"])

        # Test schema created
        assert isinstance(schema, Schema)
        assert schema._fwk == "CUSTOM"

    def test_convert_to_schema_invalid_type(self) -> None:
        """*test_convert_to_schema_invalid_type()* tests _convert_to_schema error handling."""
        # Create workflow
        workflow = WorkflowBase()

        # Try to convert invalid type
        with pytest.raises(TypeError) as excinfo:
            workflow._convert_to_schema(123)  # type: ignore
        assert "Schema input must be" in str(excinfo.value)

    def test_coefficients_setter_resets_solved_state(self) -> None:
        """*test_coefficients_setter_resets_solved_state()* tests coefficients setter clears is_solved flag."""
        # Create workflow and mark as solved
        workflow = WorkflowBase()
        workflow._is_solved = True

        # Set coefficients
        workflow.coefficients = self.test_coefficients

        # Test is_solved reset
        assert workflow.is_solved is False
        assert len(workflow.coefficients) == 2
