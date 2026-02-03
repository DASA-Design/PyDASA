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
            var = Variable(**var_data)
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
        # CUSTOM framework requires _schema parameter with FDU definitions
        engine = AnalysisEngine(
            _idx=0,
            _fwk="CUSTOM",
            _schema=self.test_data["FDU_LIST"],  # Provide custom FDU definitions
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

    def test_create_matrix(self) -> None:
        """*test_create_matrix()* tests creating dimensional matrix."""
        # Create engine with variables
        engine = AnalysisEngine(_fwk="CUSTOM", _schema=self.test_data["FDU_LIST"])
        engine.variables = self.test_variables

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
        engine = AnalysisEngine(_fwk="CUSTOM", _schema=self.test_data["FDU_LIST"])
        engine.variables = self.test_variables

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
        engine = AnalysisEngine(_fwk="CUSTOM", _schema=self.test_data["FDU_LIST"])
        engine.variables = self.test_variables
        engine._coefficients = self.test_coefficients
        engine._is_solved = True

        # Reset engine
        engine.reset()

        # Test state cleared (including Matrix-specific _model)
        assert engine.matrix is None
        assert len(engine.coefficients) == 0
        assert engine.is_solved is False
        # Test variables preserved
        assert len(engine.variables) > 0

    def test_clear(self) -> None:
        """*test_clear()* tests clearing all engine data."""
        # Create engine with data
        engine = AnalysisEngine(_fwk="CUSTOM", _schema=self.test_data["FDU_LIST"])
        engine.variables = self.test_variables
        engine._coefficients = self.test_coefficients
        engine._is_solved = True

        # Clear engine
        engine.clear()

        # Test everything cleared (including Matrix-specific _model)
        assert len(engine.variables) == 0
        assert len(engine.coefficients) == 0
        assert engine.matrix is None
        assert engine.is_solved is False
        # Test schema reset to default
        assert engine.schema is not None
        assert engine.schema._fwk == "PHYSICAL"

    def test_to_dict(self) -> None:
        """*test_to_dict()* tests converting engine to dictionary."""
        # Create engine
        engine = AnalysisEngine(
            _idx=0,
            _fwk="CUSTOM",
            _schema=self.test_data["FDU_LIST"],
            _name="Test",
            description="Test engine"
        )
        engine.variables = self.test_variables

        # Convert to dict
        data = engine.to_dict()

        # Test dictionary structure (including Foundation and WorkflowBase fields)
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
        # Matrix-specific: model field should not be present if matrix not created
        assert "model" not in data or data.get("model") is None

    def test_from_dict(self) -> None:
        """*test_from_dict()* tests creating engine from dictionary."""
        # Create engine and convert to dict
        engine1 = AnalysisEngine(
            _idx=0,
            _fwk="CUSTOM",
            _schema=self.test_data["FDU_LIST"],
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

    def test_to_dict_with_matrix(self) -> None:
        """*test_to_dict_with_matrix()* tests serialization includes Matrix when present."""
        # Create engine with matrix
        engine = AnalysisEngine(_fwk="CUSTOM", _schema=self.test_data["FDU_LIST"])
        engine.variables = self.test_variables
        engine.create_matrix()

        # Convert to dict
        data = engine.to_dict()

        # Test Matrix serialized
        assert "model" in data
        assert data["model"] is not None
        assert isinstance(data["model"], dict)

    def test_from_dict_with_matrix(self) -> None:
        """*test_from_dict_with_matrix()* tests deserialization restores Matrix."""
        # Create engine with matrix
        engine1 = AnalysisEngine(_fwk="CUSTOM", _schema=self.test_data["FDU_LIST"])
        engine1.variables = self.test_variables
        engine1.create_matrix()
        data = engine1.to_dict()

        # Create new engine from dict
        engine2 = AnalysisEngine.from_dict(data)

        # Test Matrix restored
        assert engine2.matrix is not None
        assert isinstance(engine2.matrix, Matrix)

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
        assert "is_solved=" in repr_str

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
        assert "must be str or dict or list or Schema" in str(excinfo.value)

    def test_derive_coefficient_without_matrix(self) -> None:
        """*test_derive_coefficient_without_matrix()* tests error when deriving without matrix."""
        # Create engine without matrix
        engine = AnalysisEngine()
        engine.variables = self.test_variables

        # Try to derive coefficient without creating matrix
        with pytest.raises(ValueError) as excinfo:
            engine.derive_coefficient(expr="\\Pi_{0}**(-1)", symbol="\\Pi_{4}")
        assert "Matrix must be created before deriving coefficients" in str(excinfo.value)

    def test_derive_coefficient_without_solving(self) -> None:
        """*test_derive_coefficient_without_solving()* tests error when deriving without solving."""
        # Create engine and matrix but don't solve
        engine = AnalysisEngine(_fwk="CUSTOM", _schema=self.test_data["FDU_LIST"])
        engine.variables = self.test_variables
        engine.create_matrix()

        # Try to derive coefficient without solving
        with pytest.raises(ValueError) as excinfo:
            engine.derive_coefficient(expr="\\Pi_{0}**(-1)", symbol="\\Pi_{4}")
        assert "Matrix must be solved before deriving coefficients" in str(excinfo.value)

    def test_derive_coefficient_success(self) -> None:
        """*test_derive_coefficient_success()* tests successful coefficient derivation."""
        # Create and solve engine
        engine = AnalysisEngine(_fwk="CUSTOM", _schema=self.test_data["FDU_LIST"])
        engine.variables = self.test_variables
        engine.run_analysis()

        # Derive a new coefficient (inverse of Pi_0)
        derived = engine.derive_coefficient(
            expr="\\Pi_{0}**(-1)",
            symbol="\\Pi_{4}",
            name="Derived Coefficient",
            description="Test derived coefficient"
        )

        # Test coefficient derived
        assert derived is not None
        assert isinstance(derived, Coefficient)
        assert derived.name == "Derived Coefficient"
        assert derived.description == "Test derived coefficient"

    def test_derive_coefficient_with_multiplication(self) -> None:
        """*test_derive_coefficient_with_multiplication()* tests deriving coefficient with multiplication."""
        # Create and solve engine
        engine = AnalysisEngine(_fwk="CUSTOM", _schema=self.test_data["FDU_LIST"])
        engine.variables = self.test_variables
        engine.run_analysis()

        # Get number of existing coefficients
        n_coeffs = len(engine.coefficients)

        # Only test if we have at least 2 coefficients
        if n_coeffs >= 2:
            # Derive a combined coefficient
            derived = engine.derive_coefficient(
                expr="\\Pi_{0} * \\Pi_{1}",
                symbol="\\Pi_{4}",
                name="Combined Coefficient"
            )

            # Test coefficient derived
            assert derived is not None
            assert isinstance(derived, Coefficient)
            assert derived.name == "Combined Coefficient"

    def test_derive_coefficient_delegates_to_matrix(self) -> None:
        """*test_derive_coefficient_delegates_to_matrix()* tests that derive_coefficient delegates to Matrix."""
        # Create and solve engine
        engine = AnalysisEngine(_fwk="CUSTOM", _schema=self.test_data["FDU_LIST"])
        engine.variables = self.test_variables
        engine.run_analysis()

        # Derive coefficient
        derived = engine.derive_coefficient(
            expr="\\Pi_{0}**(-1)",
            symbol="\\Pi_{99}",
            name="Test",
            description="Test description",
            idx=99
        )

        # Test parameters passed correctly
        assert derived.name == "Test"
        assert derived.description == "Test description"
        assert derived.idx == 99

    def test_calculate_coefficients_without_matrix(self) -> None:
        """*test_calculate_coefficients_without_matrix()* tests error when calculating without matrix."""
        # Create engine without matrix
        engine = AnalysisEngine()
        engine.variables = self.test_variables

        # Try to calculate without creating matrix
        with pytest.raises(ValueError) as excinfo:
            engine.calculate_coefficients()
        assert "Matrix must be created before calculating coefficients" in str(excinfo.value)

    def test_calculate_coefficients_without_solving(self) -> None:
        """*test_calculate_coefficients_without_solving()* tests error when calculating without solving."""
        # Create engine and matrix but don't solve
        engine = AnalysisEngine(_fwk="CUSTOM", _schema=self.test_data["FDU_LIST"])
        engine.variables = self.test_variables
        engine.create_matrix()

        # Try to calculate without solving
        with pytest.raises(ValueError) as excinfo:
            engine.calculate_coefficients()
        assert "Matrix must be solved before calculating coefficients" in str(excinfo.value)

    def test_calculate_coefficients_with_defaults(self) -> None:
        """*test_calculate_coefficients_with_defaults()* tests calculating with default setpoints."""
        # Create and solve engine
        engine = AnalysisEngine(_fwk="CUSTOM",
                                _schema=self.test_data["FDU_LIST"])
        engine.variables = self.test_variables
        
        # Set std_setpoint on all variables to enable default calculation
        for var in engine.variables.values():
            var.std_setpoint = 1.0
        
        engine.run_analysis()

        # Calculate coefficients using default setpoints from variables
        results = engine.calculate_coefficients()

        # Test results returned
        assert isinstance(results, dict)
        # Should have results for coefficients that can be calculated

    def test_calculate_coefficients_with_custom_setpoints(self) -> None:
        """*test_calculate_coefficients_with_custom_setpoints()* tests calculating with custom setpoints."""
        # Create and solve engine
        engine = AnalysisEngine(_fwk="CUSTOM",
                                _schema=self.test_data["FDU_LIST"])
        engine.variables = self.test_variables
        engine.run_analysis()

        # Prepare custom setpoints for all variables
        custom_setpoints = {}
        for var_sym, var in self.test_variables.items():
            custom_setpoints[var_sym] = 1.0  # Use simple value for all

        # Calculate coefficients with custom setpoints
        results = engine.calculate_coefficients(setpoints=custom_setpoints)

        # Test results returned
        assert isinstance(results, dict)
        # Should have results for coefficients that can be calculated

    def test_calculate_coefficients_with_partial_setpoints(self) -> None:
        """*test_calculate_coefficients_with_partial_setpoints()* tests with incomplete setpoints."""
        # Create and solve engine
        engine = AnalysisEngine(_fwk="CUSTOM",
                                _schema=self.test_data["FDU_LIST"])
        engine.variables = self.test_variables
        engine.run_analysis()

        # Provide only partial setpoints (not all variables)
        partial_setpoints = {"U": 1.0, "h": 0.1}

        # Calculate coefficients with partial setpoints
        results = engine.calculate_coefficients(setpoints=partial_setpoints)

        # Test results returned (may be partial or empty)
        assert isinstance(results, dict)
        # Method should not fail, but may return limited results

    def test_calculate_coefficients_updates_coefficient_setpoints(self) -> None:
        """*test_calculate_coefficients_updates_coefficient_setpoints()* tests that coefficient setpoints are updated."""
        # Create and solve engine
        engine = AnalysisEngine(_fwk="CUSTOM", _schema=self.test_data["FDU_LIST"])
        engine.variables = self.test_variables
        engine.run_analysis()

        # Get a coefficient before calculation
        if len(engine.coefficients) > 0:
            first_coef_sym = list(engine.coefficients.keys())[0]
            # first_coef = engine.coefficients[first_coef_sym]

            # Prepare setpoints
            custom_setpoints = {var_sym: 1.0 for var_sym in self.test_variables.keys()}

            # Calculate coefficients
            results = engine.calculate_coefficients(setpoints=custom_setpoints)

            # If this coefficient was calculated, check it's in results
            if first_coef_sym in results:
                # Verify the result is a valid number
                assert isinstance(results[first_coef_sym], (int, float))
