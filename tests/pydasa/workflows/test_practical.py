# -*- coding: utf-8 -*-
"""
Module test_practical.py
===========================================

Tests for **MonteCarloSimulation** in *PyDASA*.

This module provides unit tests for managing Monte Carlo simulations.
"""
# import testing package
import unittest
import pytest
import numpy as np

# import the module to test
from pydasa.workflows.practical import MonteCarloSimulation

# import required classes
from pydasa.elements.parameter import Variable
from pydasa.dimensional.buckingham import Coefficient   # vashchy
from pydasa.dimensional.vaschy import Schema

# import test data
from tests.pydasa.data.test_data import get_simulation_test_data

# asserting module imports
assert MonteCarloSimulation
assert get_simulation_test_data


class TestMonteCarloSimulation(unittest.TestCase):
    """**TestMonteCarloSimulation** implements unit tests for Monte Carlo handler.

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
            # Set schema and prepare dimensions for CUSTOM framework variables
            var._schema = self.dim_schema
            var._prepare_dims()
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
        handler = MonteCarloSimulation()

        # Test if handler is not None
        assert handler is not None
        # Test default category
        assert handler._cat == "DIST"
        # Test default experiments
        assert handler._experiments == -1
        # Test empty collections
        assert len(handler._variables) == 0
        assert len(handler._coefficients) == 0
        assert len(handler._distributions) == 0
        assert len(handler._simulations) == 0
        assert len(handler._results) == 0
        # Test if handler is instance
        assert isinstance(handler, MonteCarloSimulation)

    def test_custom_handler(self) -> None:
        """*test_custom_handler()* tests creating handler with custom values."""
        # Create handler with custom data
        handler = MonteCarloSimulation(
            _idx=0,
            _fwk="CUSTOM",
            _cat="DIST",
            _experiments=500,
            _variables=self.test_variables,
            _coefficients=self.test_coefficients,
            _name="Test Handler",
            description="Test Monte Carlo Handler"
        )

        # Test if handler is not None
        assert handler is not None
        # Test custom category
        assert handler._cat == "DIST"
        # Test custom experiments
        assert handler._experiments == 500
        # Test variables set
        assert len(handler._variables) == len(self.test_variables)
        # Test coefficients set
        assert len(handler._coefficients) == len(self.test_coefficients)
        # Test name and description
        assert handler.name == "Test Handler"
        assert handler.description == "Test Monte Carlo Handler"

    def test_configure_distributions(self) -> None:
        """*test_configure_distributions()* tests distribution configuration."""
        # Create handler
        handler = MonteCarloSimulation(
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )

        # Configure distributions
        handler.configure_distributions()

        # Test distributions created
        assert len(handler._distributions) > 0
        # Test each variable has distribution
        for sym in self.test_variables.keys():
            assert sym in handler._distributions
            dist = handler._distributions[sym]
            assert "dtype" in dist
            assert "params" in dist

        # Test error: no variables defined
        handler_no_vars = MonteCarloSimulation()
        with pytest.raises(ValueError) as excinfo:
            handler_no_vars.configure_distributions()
        assert "no variables defined" in str(excinfo.value)

    def test_config_simulations(self) -> None:
        """*test_config_simulations()* tests simulation configuration."""
        # Create handler
        handler = MonteCarloSimulation(
            _fwk="CUSTOM",
            _experiments=100,
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )

        # Configure distributions first
        handler.configure_distributions()

        # Configure simulations
        handler.configure_simulations()

        # Test simulations created
        assert len(handler._simulations) > 0
        # Test one simulation per coefficient
        assert len(handler._simulations) == len(self.test_coefficients)

        # Test error: no coefficients
        handler_no_coefs = MonteCarloSimulation(
            _fwk="CUSTOM",
            _experiments=100,
            _variables=self.test_variables
        )
        handler_no_coefs.configure_distributions()
        with pytest.raises(ValueError) as excinfo:
            handler_no_coefs.configure_simulations()
        assert "no coefficients defined" in str(excinfo.value)

        # Test error: no variables
        handler_no_vars = MonteCarloSimulation(
            _fwk="CUSTOM",
            _experiments=100,
            _coefficients=self.test_coefficients
        )
        with pytest.raises(ValueError) as excinfo:
            handler_no_vars.configure_simulations()
        assert "no variables defined" in str(excinfo.value)

        # Test error: no distributions
        handler_no_dist = MonteCarloSimulation(
            _fwk="CUSTOM",
            _experiments=100,
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )
        with pytest.raises(ValueError) as excinfo:
            handler_no_dist.configure_simulations()
        assert "distributions not defined" in str(excinfo.value)

    def test_clear(self) -> None:
        """*test_clear()* tests clearing handler state completely."""
        # Create handler with data
        handler = MonteCarloSimulation(
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )

        # Configure distributions
        handler.configure_distributions()

        # Clear handler
        handler.clear()

        # Test all collections cleared including variables and coefficients
        assert len(handler._simulations) == 0
        assert len(handler._distributions) == 0
        assert len(handler._results) == 0
        assert len(handler._shared_cache) == 0
        assert len(handler._variables) == 0
        assert len(handler._coefficients) == 0

    def test_reset(self) -> None:
        """*test_reset()* tests resetting simulation state while preserving configuration."""
        # Create handler with data
        handler = MonteCarloSimulation(
            _experiments=100,
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )

        # Configure distributions and simulations
        handler.create_simulations()

        # Store original counts
        original_var_count = len(handler._variables)
        original_coef_count = len(handler._coefficients)

        # Reset handler
        handler.reset()

        # Test simulation state cleared
        assert len(handler._simulations) == 0
        assert len(handler._distributions) == 0
        assert len(handler._results) == 0
        assert len(handler._shared_cache) == 0

        # Test configuration preserved
        assert len(handler._variables) == original_var_count
        assert len(handler._coefficients) == original_coef_count

    def test_properties(self) -> None:
        """*test_properties()* tests property getters and setters."""
        # Create handler
        handler = MonteCarloSimulation()

        # Test cat property
        assert handler.cat == "DIST"
        handler.cat = "DATA"
        assert handler.cat == "DATA"

        # Test invalid cat value
        with pytest.raises(ValueError):
            handler.cat = "INVALID"

        # Reset to DIST for testing simulations
        handler.cat = "DIST"

        # Test experiments property
        assert handler.experiments == -1
        handler.experiments = 2000
        assert handler.experiments == 2000

        # Test invalid experiments
        with pytest.raises(ValueError) as excinfo:
            handler.experiments = 0
        assert "must be >= 1" in str(excinfo.value)

        # Test is_configured property (initially False)
        assert handler.is_configured is False

        # Configure handler and test is_configured becomes True
        handler._variables = self.test_variables
        handler._coefficients = self.test_coefficients
        handler.experiments = 5  # Set experiments before creating simulations
        handler.create_simulations()
        assert handler.is_configured is True

        # Test has_results property
        assert handler.has_results is False
        handler.run_simulation(iters=5)
        assert handler.has_results is True

        # Test simulations property returns copy
        sims_copy = handler.simulations
        assert isinstance(sims_copy, dict)
        assert len(sims_copy) == len(handler._simulations)
        # Verify it's a copy, not the same object
        assert id(sims_copy) != id(handler._simulations)

    def test_get_simulation(self) -> None:
        """*test_get_simulation()* tests getting simulation by name."""
        # Create handler
        handler = MonteCarloSimulation(
            _fwk="CUSTOM",
            _experiments=100,
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )

        # Configure
        handler.create_simulations()

        # Get existing simulation
        sim = handler.get_simulation("\\Pi_{0}")
        assert sim is not None

        # Test non-existing simulation
        with pytest.raises(ValueError) as excinfo:
            handler.get_simulation("invalid")
        assert "does not exist" in str(excinfo.value)

    def test_get_distribution(self) -> None:
        """*test_get_distribution()* tests getting distribution by name."""
        # Create handler
        handler = MonteCarloSimulation(_variables=self.test_variables)

        # Configure distributions
        handler.configure_distributions()

        # Get existing distribution
        dist = handler.get_distribution("U")
        assert dist is not None

        # Test non-existing distribution
        with pytest.raises(ValueError) as excinfo:
            handler.get_distribution("invalid")
        assert "does not exist" in str(excinfo.value)

    def test_to_dict(self) -> None:
        """*test_to_dict()* tests converting handler to dictionary."""
        # Create handler
        handler = MonteCarloSimulation(
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

    def test_shared_cache_initialization(self) -> None:
        """*test_shared_cache_initialization()* tests that shared cache is properly initialized."""
        handler = MonteCarloSimulation(
            _fwk="CUSTOM",
            _experiments=100,
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )
        handler.create_simulations()

        # Verify shared cache exists
        assert hasattr(handler, '_shared_cache')
        assert isinstance(handler._shared_cache, dict)

        # Verify cache for all variables
        for var_sym in handler._variables.keys():
            assert var_sym in handler._shared_cache
            assert handler._shared_cache[var_sym].shape == (handler._experiments, 1)

    def test_simulations_share_same_cache(self) -> None:
        """*test_simulations_share_same_cache()* tests all simulations reference same cache object."""
        handler = MonteCarloSimulation(
            _fwk="CUSTOM",
            _experiments=100,
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )
        handler.create_simulations()

        # Get cache IDs from all simulations
        cache_ids = [id(sim._simul_cache) for sim in handler._simulations.values()]

        # All should reference the same object
        assert len(set(cache_ids)) == 1
        assert cache_ids[0] == id(handler._shared_cache)

    def test_cache_values_consistent_across_coefficients(self) -> None:
        """*test_cache_values_consistent_across_coefficients()* tests same variable has identical values across coefficients."""
        handler = MonteCarloSimulation(
            _fwk="CUSTOM",
            _variables=self.test_variables,
            _coefficients=self.test_coefficients,
            _experiments=5
        )
        handler.create_simulations()
        handler.run_simulation(iters=5)

        # Extract results
        all_results = {pi: sim.extract_results() for pi, sim in handler._simulations.items()}

        # Check variable 'U' appears in multiple coefficients
        u_values = []
        for pi_sym, results in all_results.items():
            for key in results.keys():
                if key.startswith("U@"):
                    u_values.append(results[key])

        # All U values should be identical
        if len(u_values) > 1:
            reference = u_values[0]
            for values in u_values[1:]:
                np.testing.assert_array_equal(reference, values)

    def test_cache_modification_visible_everywhere(self) -> None:
        """*test_cache_modification_visible_everywhere()* tests cache changes are visible to all simulations."""
        handler = MonteCarloSimulation(
            _fwk="CUSTOM",
            _experiments=100,
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )
        handler.create_simulations()

        # Modify cache through handler
        test_var = 'd'
        test_idx = 2
        test_value = 99.99
        handler._shared_cache[test_var][test_idx, 0] = test_value

        # Verify all simulations see the change
        for sim in handler._simulations.values():
            cached = sim._get_cached_value(test_var, test_idx)
            assert cached == test_value

    def test_run_simulation_convenience_method(self) -> None:
        """*test_run_simulation_convenience_method()* tests run_simulation() combines create and run."""
        handler = MonteCarloSimulation(
            _fwk="CUSTOM",
            _experiments=5,
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )

        # Run simulation with all setup in one call
        handler.run_simulation(iters=5)

        # Verify simulations were created
        assert len(handler._simulations) > 0

        # Verify results were generated
        assert len(handler._results) > 0

        # Verify experiments were set
        assert handler._experiments == 5

        # Test with explicit mode parameter (still DIST since we use distributions)
        handler2 = MonteCarloSimulation(
            _fwk="CUSTOM",
            _cat="DIST",
            _experiments=5,
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )
        # Change mode before running to test mode parameter
        original_mode = handler2._cat
        handler2.run_simulation(iters=5, mode="DIST")
        assert handler2._cat == "DIST"
        # Verify mode parameter can change the category
        assert original_mode == "DIST"

    def test_create_simulations_method(self) -> None:
        """*test_create_simulations_method()* tests new create_simulations() method name."""
        handler = MonteCarloSimulation(
            _fwk="CUSTOM",
            _experiments=100,
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )

        # Use new method name
        handler.create_simulations()

        # Verify distributions and simulations created
        assert len(handler._distributions) > 0
        assert len(handler._simulations) > 0

    def test_run_method(self) -> None:
        """*test_run_method()* tests run_simulation() method with iters parameter."""
        handler = MonteCarloSimulation(
            _fwk="CUSTOM",
            _experiments=5,
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )

        # Setup and run with method names
        handler.create_simulations()
        handler.run_simulation(iters=5)

        # Verify results generated
        assert len(handler._results) > 0
        assert handler._experiments == 5

    def test_from_dict_method(self) -> None:
        """*test_from_dict_method()* tests deserialization from dictionary."""
        # Create original handler
        handler_original = MonteCarloSimulation(
            _idx=3,
            _fwk="CUSTOM",
            _name="Test Handler",
            description="Test for serialization",
            _cat="DIST",
            _experiments=50,
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )

        # Configure and run simulations
        handler_original.create_simulations()
        handler_original.run_simulation(iters=50)

        # Serialize
        handler_dict = handler_original.to_dict()

        # Deserialize
        handler_restored = MonteCarloSimulation.from_dict(handler_dict)

        # Verify restoration
        assert handler_restored is not None
        assert handler_restored._idx == handler_original._idx
        assert handler_restored._fwk == handler_original._fwk
        assert handler_restored._name == handler_original._name
        assert handler_restored.description == handler_original.description
        assert handler_restored._cat == handler_original._cat
        assert handler_restored._experiments == handler_original._experiments

        # Verify variables are restored
        assert len(handler_restored._variables) == len(handler_original._variables)

        # Verify coefficients are restored
        assert len(handler_restored._coefficients) == len(handler_original._coefficients)

        # Verify simulations are restored
        assert len(handler_restored._simulations) == len(handler_original._simulations)

        # Verify distributions are restored
        assert len(handler_restored._distributions) == len(handler_original._distributions)

        # Verify results are restored
        assert len(handler_restored._results) == len(handler_original._results)

    def test_serialization_round_trip(self) -> None:
        """*test_serialization_round_trip()* tests complete round-trip serialization."""
        # Create and configure original handler
        handler_original = MonteCarloSimulation(
            _idx=7,
            _fwk="CUSTOM",
            _name="Round Trip Handler",
            _cat="DIST",
            _experiments=10,
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )

        handler_original.create_simulations()
        handler_original.run_simulation(iters=10)

        # Get original results count
        original_results_count = len(handler_original._results)

        # Round-trip: serialize and deserialize
        handler_dict = handler_original.to_dict()
        handler_restored = MonteCarloSimulation.from_dict(handler_dict)

        # Verify results count matches
        assert len(handler_restored._results) == original_results_count

        # Verify key attributes match
        assert handler_restored._idx == handler_original._idx
        assert handler_restored._name == handler_original._name
        assert handler_restored._cat == handler_original._cat
        assert handler_restored._experiments == handler_original._experiments
        assert handler_restored._is_solved == handler_original._is_solved

    def test_from_dict_with_minimal_data(self) -> None:
        """*test_from_dict_with_minimal_data()* tests deserialization with minimal required data."""
        # Create minimal dictionary
        minimal_dict = {
            "_idx": 0,
            "_fwk": "CUSTOM",
            "cat": "DIST",
            "experiments": 100
        }

        # Should create a valid MonteCarloSimulation instance
        handler = MonteCarloSimulation.from_dict(minimal_dict)

        assert handler is not None
        assert handler._idx == 0
        assert handler._fwk == "CUSTOM"
        assert handler._cat == "DIST"
        assert handler._experiments == 100

    def test_from_dict_preserves_configuration(self) -> None:
        """*test_from_dict_preserves_configuration()* tests that configuration is preserved during deserialization."""
        # Create handler with full configuration
        handler_original = MonteCarloSimulation(
            _idx=5,
            _fwk="CUSTOM",
            _cat="DATA",
            _experiments=200,
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )

        # Configure without running
        handler_original.create_simulations()

        # Serialize and deserialize
        handler_dict = handler_original.to_dict()
        handler_restored = MonteCarloSimulation.from_dict(handler_dict)

        # Verify configuration preserved
        assert handler_restored.is_configured == handler_original.is_configured
        assert len(handler_restored._simulations) == len(handler_original._simulations)
        assert len(handler_restored._distributions) == len(handler_original._distributions)

    def test_get_results(self) -> None:
        """*test_get_results()* tests getting results by simulation name."""
        # Create and run simulation
        handler = MonteCarloSimulation(
            _fwk="CUSTOM",
            _experiments=5,
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )
        handler.run_simulation(iters=5)

        # Get results for existing simulation
        results = handler.get_results("\\Pi_{0}")
        assert results is not None
        assert isinstance(results, dict)
        assert "inputs" in results
        assert "results" in results
        assert "statistics" in results

        # Test error: non-existing results
        with pytest.raises(ValueError) as excinfo:
            handler.get_results("invalid_name")
        assert "do not exist" in str(excinfo.value)

    def test_validation_errors(self) -> None:
        """*test_validation_errors()* tests various validation error paths."""
        # Test _validate_coefficient_vars with missing var_dims attribute
        handler = MonteCarloSimulation(
            _fwk="CUSTOM",
            _variables=self.test_variables
        )

        # Create coefficient without var_dims
        bad_coef = Coefficient(
            _idx=0,
            _sym="\\Pi_{bad}",
            _fwk="CUSTOM"
        )
        # Remove var_dims if it exists
        if hasattr(bad_coef, 'var_dims'):
            delattr(bad_coef, 'var_dims')

        with pytest.raises(ValueError) as excinfo:
            handler._validate_coefficient_vars(bad_coef, "\\Pi_{bad}")
        assert "missing var_dims attribute" in str(excinfo.value)

        # Test _validate_coefficient_vars with None var_dims
        bad_coef2 = Coefficient(
            _idx=1,
            _sym="\\Pi_{bad2}",
            _fwk="CUSTOM"
        )
        bad_coef2.var_dims = None   # type: ignore

        with pytest.raises(ValueError) as excinfo:
            handler._validate_coefficient_vars(bad_coef2, "\\Pi_{bad2}")
        assert "has None var_dims" in str(excinfo.value)

        # Test _validate_coefficient_vars with wrong type
        bad_coef3 = Coefficient(
            _idx=2,
            _sym="\\Pi_{bad3}",
            _fwk="CUSTOM"
        )
        bad_coef3.var_dims = [1, 2, 3]  # type: ignore # List instead of dict

        with pytest.raises(TypeError) as excinfo:
            handler._validate_coefficient_vars(bad_coef3, "\\Pi_{bad3}")
        assert "must be a dictionary" in str(excinfo.value)

        # Test _init_shared_cache with negative experiments
        handler_neg = MonteCarloSimulation(
            _fwk="CUSTOM",
            _experiments=-5,
            _variables=self.test_variables
        )

        with pytest.raises(ValueError) as excinfo:
            handler_neg._init_shared_cache()
        assert "must be positive" in str(excinfo.value)

        # Test _get_distributions with missing distributions
        handler2 = MonteCarloSimulation(
            _fwk="CUSTOM",
            _variables=self.test_variables
        )
        handler2.configure_distributions()

        with pytest.raises(ValueError) as excinfo:
            handler2._get_distributions(["nonexistent_var"])
        assert "Missing distributions" in str(excinfo.value)
