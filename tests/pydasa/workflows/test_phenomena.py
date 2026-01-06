# -*- coding: utf-8 -*-
"""
Module test_phenomena.py
===========================================

Tests for **AnalysisEngine** in *PyDASA*.

This module provides unit tests for managing dimensional analysis workflows.
"""
# import testing package
import unittest
import pytest

# import the module to test
from pydasa.workflows.phenomena import AnalysisEngine

# import required classes
from pydasa.elements.parameter import Variable
from pydasa.dimensional.buckingham import Coefficient
from pydasa.dimensional.model import Matrix
from pydasa.dimensional.vaschy import Schema

# import test data
from tests.pydasa.data.test_data import get_simulation_test_data, get_model_test_data

# asserting module imports
assert AnalysisEngine
assert get_simulation_test_data
assert get_model_test_data


class TestAnalysisEngine(unittest.TestCase):
    """**TestAnalysisEngine** implements unit tests for dimensional analysis engine.

    Args:
        unittest (TestCase): unittest.TestCase class for Python unit testing.
    """

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """*inject_fixtures()* injects global test parameters as fixture."""
        # Get test data
        self.test_data = get_simulation_test_data()
        self.model_data = get_model_test_data()

        # Setup dimensional schema
        self.dim_schema = Schema(_fwk="CUSTOM",
                                 _fdu_lt=self.test_data["FDU_LIST"],
                                 _idx=0)
        self.dim_schema._setup_fdus()

        # Setup variables
        self.test_variables = {}
        for sym, var_data in self.test_data["CHANNEL_FLOW_VARIABLES"].items():
            self.test_variables[sym] = Variable(**var_data)

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

    def test_default_engine(self) -> None:
        """*test_default_engine()* tests creating engine with defaults."""
        # Create engine with defaults
        engine = AnalysisEngine()

        # Test if engine is not None
        assert engine is not None
        # Test default schema initialized
        assert engine.schema is not None
        assert isinstance(engine.schema, Schema)
        # Test empty collections
        assert len(engine.variables) == 0
        assert len(engine.coefficients) == 0
        # Test solve state
        assert engine.is_solved is False
        # Test if engine is instance
        assert isinstance(engine, AnalysisEngine)

    def test_custom_engine(self) -> None:
        """*test_custom_engine()* tests creating engine with custom values."""
        # Create engine with custom data
        engine = AnalysisEngine(
            _idx=0,
            _fwk="CUSTOM",
            _name="Test Engine",
            description="Test Dimensional Analysis Engine"
        )

        # Set variables after initialization
        engine.variables = self.test_variables

        # Test if engine is not None
        assert engine is not None
        # Test variables set
        assert len(engine.variables) == len(self.test_variables)
        # Test name and description
        assert engine.name == "Test Engine"
        assert engine.description == "Test Dimensional Analysis Engine"
        # Test framework
        assert engine._fwk == "CUSTOM"

    def test_schema_initialization(self) -> None:
        """*test_schema_initialization()* tests schema is initialized properly."""
        # Create engine
        engine = AnalysisEngine()

        # Test schema exists
        assert engine.schema is not None
        assert isinstance(engine.schema, Schema)
        # Test default framework
        assert engine.schema._fwk == "PHYSICAL"

    def test_schema_string_assignment(self) -> None:
        """*test_schema_string_assignment()* tests setting schema from string."""
        # Create engine
        engine = AnalysisEngine()

        # Set schema from string
        engine.schema = "COMPUTATION"

        # Test schema converted
        assert isinstance(engine.schema, Schema)
        assert engine.schema._fwk == "COMPUTATION"

    def test_schema_dict_assignment(self) -> None:
        """*test_schema_dict_assignment()* tests setting schema from dict."""
        # Create engine
        engine = AnalysisEngine()

        # Create a simple schema dict using PHYSICAL framework (has default FDUs)
        schema_dict = {
            "_fwk": "PHYSICAL",
            "_idx": 0
        }

        # Set schema from dict
        engine.schema = schema_dict

        # Test schema converted
        assert isinstance(engine.schema, Schema)
        assert engine.schema._fwk == "PHYSICAL"

    def test_schema_object_assignment(self) -> None:
        """*test_schema_object_assignment()* tests setting schema from Schema object."""
        # Create engine
        engine = AnalysisEngine()

        # Set schema from object
        engine.schema = self.dim_schema

        # Test schema assigned
        assert engine.schema is self.dim_schema
        assert engine.schema._fwk == "CUSTOM"

    def test_variables_dict_assignment(self) -> None:
        """*test_variables_dict_assignment()* tests setting variables from dict."""
        # Create engine
        engine = AnalysisEngine()

        # Set variables
        engine.variables = self.test_variables

        # Test variables set
        assert len(engine.variables) == len(self.test_variables)
        for sym in self.test_variables.keys():
            assert sym in engine.variables
            assert isinstance(engine.variables[sym], Variable)

    def test_variables_conversion_from_dicts(self) -> None:
        """*test_variables_conversion_from_dicts()* tests converting dict values to Variables."""
        # Create engine
        engine = AnalysisEngine()

        # Create dict with dict values
        var_dicts = {}
        for sym, var in self.test_variables.items():
            var_dicts[sym] = var.to_dict()

        # Set variables from dicts
        engine.variables = var_dicts

        # Test conversion happened
        assert len(engine.variables) == len(var_dicts)
        for sym in var_dicts.keys():
            assert isinstance(engine.variables[sym], Variable)

    def test_variables_invalid_type(self) -> None:
        """*test_variables_invalid_type()* tests error on invalid variable type."""
        # Create engine
        engine = AnalysisEngine()

        # Try to set invalid variables
        with pytest.raises(ValueError) as excinfo:
            engine.variables = {"key": "invalid_string"}
        assert "must be type 'Variable' or 'dict'" in str(excinfo.value)

    def test_create_matrix(self) -> None:
        """*test_create_matrix()* tests creating dimensional matrix."""
        # Create engine with variables
        engine = AnalysisEngine(_fwk="CUSTOM")
        engine.variables = self.test_variables
        engine.schema = self.dim_schema

        # Create matrix
        engine.create_matrix()

        # Test matrix created
        assert engine.matrix is not None
        assert isinstance(engine.matrix, Matrix)
        # Test solve state reset
        assert engine.is_solved is False

    def test_create_matrix_without_variables(self) -> None:
        """*test_create_matrix_without_variables()* tests error when creating matrix without variables."""
        # Create engine without variables
        engine = AnalysisEngine()

        # Try to create matrix
        with pytest.raises(ValueError) as excinfo:
            engine.create_matrix()
        assert "Variables must be set" in str(excinfo.value)

    def test_solve_without_matrix(self) -> None:
        """*test_solve_without_matrix()* tests error when solving without matrix."""
        # Create engine
        engine = AnalysisEngine()
        engine.variables = self.test_variables

        # Try to solve without creating matrix
        with pytest.raises(ValueError) as excinfo:
            engine.solve()
        assert "Matrix must be created before solving" in str(excinfo.value)

    def test_run_analysis_complete_workflow(self) -> None:
        """*test_run_analysis_complete_workflow()* tests complete analysis workflow."""
        # Create engine with minimal setup
        engine = AnalysisEngine(_fwk="CUSTOM")
        engine.variables = self.test_variables
        engine.schema = self.dim_schema

        # Run complete workflow
        # try:
        result = engine.run_analysis()

        # Test workflow completed
        assert engine.matrix is not None
        assert isinstance(result, dict)
        # Test solve state updated
        assert engine.is_solved is True
        # except Exception as e:
        #     # If Matrix.solve() not fully implemented, that's okay for now
        #     pytest.skip(f"Matrix solving not fully implemented: {str(e)}")

    def test_reset(self) -> None:
        """*test_reset()* tests resetting engine state."""
        # Create engine with data
        engine = AnalysisEngine(_fwk="CUSTOM")
        engine.variables = self.test_variables
        engine._coefficients = self.test_coefficients
        engine._is_solved = True

        # Reset engine
        engine.reset()

        # Test state cleared
        assert engine.matrix is None
        assert len(engine.coefficients) == 0
        assert engine.is_solved is False
        # Test variables preserved
        assert len(engine.variables) > 0

    def test_clear(self) -> None:
        """*test_clear()* tests clearing all engine data."""
        # Create engine with data
        engine = AnalysisEngine(_fwk="CUSTOM")
        engine.variables = self.test_variables
        engine._coefficients = self.test_coefficients
        engine._is_solved = True

        # Clear engine
        engine.clear()

        # Test everything cleared
        assert len(engine.variables) == 0
        assert len(engine.coefficients) == 0
        assert engine.matrix is None
        assert engine.is_solved is False
        # Test schema reset to default
        assert engine.schema is not None
        assert engine.schema._fwk == "PHYSICAL"

    def test_properties_are_copies(self) -> None:
        """*test_properties_are_copies()* tests that properties return copies."""
        # Create engine
        engine = AnalysisEngine()
        engine.variables = self.test_variables

        # Get variables
        vars1 = engine.variables
        vars2 = engine.variables

        # Test they are different objects (copies)
        assert vars1 is not vars2
        assert id(vars1) != id(vars2)

    def test_to_dict(self) -> None:
        """*test_to_dict()* tests converting engine to dictionary."""
        # Create engine
        engine = AnalysisEngine(
            _idx=0,
            _fwk="CUSTOM",
            _name="Test",
            description="Test engine"
        )
        engine.variables = self.test_variables

        # Convert to dict
        data = engine.to_dict()

        # Test dictionary structure
        assert isinstance(data, dict)
        assert "name" in data
        assert "description" in data
        assert "idx" in data
        assert "fwk" in data
        assert "variables" in data
        assert "coefficients" in data
        assert "is_solved" in data
        assert data["name"] == "Test"
        assert data["description"] == "Test engine"

    def test_from_dict(self) -> None:
        """*test_from_dict()* tests creating engine from dictionary."""
        # Create engine and convert to dict
        engine1 = AnalysisEngine(
            _idx=0,
            _fwk="CUSTOM",
            _name="Test",
            description="Test engine"
        )
        engine1.variables = self.test_variables
        data = engine1.to_dict()

        # Create new engine from dict
        engine2 = AnalysisEngine.from_dict(data)

        # Test engine created correctly
        assert engine2 is not None
        assert engine2.name == engine1.name
        assert engine2.description == engine1.description
        assert engine2._idx == engine1._idx
        assert len(engine2.variables) == len(engine1.variables)

    def test_repr(self) -> None:
        """*test_repr()* tests string representation."""
        # Create engine
        engine = AnalysisEngine(_name="Test Engine")
        engine.variables = self.test_variables

        # Get repr
        repr_str = repr(engine)

        # Test repr contains key info
        assert "AnalysisEngine" in repr_str
        assert "Test Engine" in repr_str
        assert "variables=" in repr_str
        assert "coefficients=" in repr_str
        assert "status=" in repr_str

    def test_repr_solved_state(self) -> None:
        """*test_repr_solved_state()* tests repr shows solved state."""
        # Create engine
        engine = AnalysisEngine(_name="Test")
        engine._is_solved = True

        # Get repr
        repr_str = repr(engine)

        # Test solved status shown
        assert "solved" in repr_str

    def test_validate_dict_helper(self) -> None:
        """*test_validate_dict_helper()* tests dictionary validation method."""
        # Create engine
        engine = AnalysisEngine()

        # Test valid dictionary
        valid_dict = {"key1": Variable(), "key2": Variable()}
        assert engine._validate_dict(valid_dict, Variable)

        # Test invalid type
        with pytest.raises(ValueError) as excinfo:
            engine._validate_dict("not a dict", Variable)   # type: ignore
        assert "must be a dictionary" in str(excinfo.value)

        # Test empty dictionary
        with pytest.raises(ValueError) as excinfo:
            engine._validate_dict({}, Variable)
        assert "cannot be empty" in str(excinfo.value)

        # Test wrong value types
        with pytest.raises(ValueError) as excinfo:
            engine._validate_dict({"key": "not_variable"}, Variable)
        assert "must contain" in str(excinfo.value)

    def test_is_solved_property(self) -> None:
        """*test_is_solved_property()* tests is_solved property."""
        # Create engine
        engine = AnalysisEngine()

        # Test initial state
        assert engine.is_solved is False

        # Set solved state
        engine._is_solved = True
        assert engine.is_solved is True

    def test_coefficients_property_setter(self) -> None:
        """*test_coefficients_property_setter()* tests coefficients property setter."""
        # Create engine
        engine = AnalysisEngine()

        # Set coefficients
        engine.coefficients = self.test_coefficients

        # Test coefficients set
        assert len(engine.coefficients) == len(self.test_coefficients)
        for sym in self.test_coefficients.keys():
            assert sym in engine.coefficients

    def test_coefficients_conversion_from_dicts(self) -> None:
        """*test_coefficients_conversion_from_dicts()* tests converting dict values to Coefficients."""
        # Create engine
        engine = AnalysisEngine()

        # Create dict with dict values
        coef_dicts = {}
        for sym, coef in self.test_coefficients.items():
            coef_dicts[sym] = coef.to_dict()

        # Set coefficients from dicts
        engine.coefficients = coef_dicts

        # Test conversion happened
        assert len(engine.coefficients) == len(coef_dicts)
        for sym in coef_dicts.keys():
            assert isinstance(engine.coefficients[sym], Coefficient)

    def test_matrix_property(self) -> None:
        """*test_matrix_property()* tests matrix property getter/setter."""
        # Create engine
        engine = AnalysisEngine()

        # Test initial state
        assert engine.matrix is None

        # Create and set matrix
        engine.variables = self.test_variables
        engine.schema = self.dim_schema
        engine.create_matrix()

        # Test matrix set
        assert engine.matrix is not None
        assert isinstance(engine.matrix, Matrix)

    def test_variables_setter_resets_solved_state(self) -> None:
        """*test_variables_setter_resets_solved_state()* tests setting variables resets solved state."""
        # Create engine
        engine = AnalysisEngine()
        engine._is_solved = True

        # Set variables
        engine.variables = self.test_variables

        # Test solved state reset
        assert engine.is_solved is False

    def test_schema_type_error(self) -> None:
        """*test_schema_type_error()* tests error on invalid schema type."""
        # Create engine
        engine = AnalysisEngine()

        # Try to set invalid schema
        with pytest.raises(ValueError) as excinfo:
            engine.schema = 123     # type: ignore
        assert "must be str or dict or Schema" in str(excinfo.value)
