# -*- coding: utf-8 -*-
"""
Module test_practical.py
===========================================

Tests for **MonteCarloHandler** in *PyDASA*.

This module provides unit tests for managing Monte Carlo simulations.
"""
# import testing package
import unittest
import pytest
import numpy as np

# import the module to test
from pydasa.handlers.practical import MonteCarloHandler

# import required classes
from pydasa.core.parameter import Variable
from pydasa.buckingham.vashchy import Coefficient
from pydasa.dimensional.framework import DimSchema

# import test data
from tests.pydasa.data.test_data import get_simulation_test_data

# asserting module imports
assert MonteCarloHandler
assert get_simulation_test_data


class TestMonteCarloHandler(unittest.TestCase):
    """**TestMonteCarloHandler** implements unit tests for Monte Carlo handler.

    Args:
        unittest (TestCase): unittest.TestCase class for Python unit testing.
    """

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """*inject_fixtures()* injects global test parameters as fixture."""
        # Get test data
        self.test_data = get_simulation_test_data()

        # Setup dimensional schema
        self.dim_schema = DimSchema(_fwk="CUSTOM",
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
                name="Test Coefficient 0",
                description="Test dimensionless coefficient"
            ),
            "\\Pi_{1}": Coefficient(
                _idx=1,
                _sym="\\Pi_{1}",
                _alias="Pi_1",
                _fwk="CUSTOM",
                _cat="COMPUTED",
                name="Test Coefficient 1",
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
        handler = MonteCarloHandler()

        # Test if handler is not None
        assert handler is not None
        # Test default category
        assert handler._cat == "NUM"
        # Test default experiments
        assert handler._experiments == 1000
        # Test empty collections
        assert len(handler._variables) == 0
        assert len(handler._coefficients) == 0
        assert len(handler._distributions) == 0
        assert len(handler._simulations) == 0
        assert len(handler._results) == 0
        # Test if handler is instance
        assert isinstance(handler, MonteCarloHandler)

    def test_custom_handler(self) -> None:
        """*test_custom_handler()* tests creating handler with custom values."""
        # Create handler with custom data
        handler = MonteCarloHandler(
            _idx=0,
            _fwk="CUSTOM",
            _cat="NUM",
            _experiments=500,
            _variables=self.test_variables,
            _coefficients=self.test_coefficients,
            name="Test Handler",
            description="Test Monte Carlo Handler"
        )

        # Test if handler is not None
        assert handler is not None
        # Test custom category
        assert handler._cat == "NUM"
        # Test custom experiments
        assert handler._experiments == 500
        # Test variables set
        assert len(handler._variables) == len(self.test_variables)
        # Test coefficients set
        assert len(handler._coefficients) == len(self.test_coefficients)
        # Test name and description
        assert handler.name == "Test Handler"
        assert handler.description == "Test Monte Carlo Handler"

    def test_config_distributions(self) -> None:
        """*test_config_distributions()* tests distribution configuration."""
        # Create handler
        handler = MonteCarloHandler(
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )

        # Configure distributions
        handler._config_distributions()

        # Test distributions created
        assert len(handler._distributions) > 0
        # Test each variable has distribution
        for sym in self.test_variables.keys():
            assert sym in handler._distributions
            dist = handler._distributions[sym]
            assert "dtype" in dist
            assert "params" in dist

    def test_config_simulations(self) -> None:
        """*test_config_simulations()* tests simulation configuration."""
        # Create handler
        handler = MonteCarloHandler(
            _fwk="CUSTOM",
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )

        # Configure distributions first
        handler._config_distributions()

        # Configure simulations
        handler._config_simulations()

        # Test simulations created
        assert len(handler._simulations) > 0
        # Test one simulation per coefficient
        assert len(handler._simulations) == len(self.test_coefficients)

    def test_validate_dict(self) -> None:
        """*test_validate_dict()* tests dictionary validation."""
        # Create handler
        handler = MonteCarloHandler()

        # Test valid dictionary
        valid_dict = {"key1": Variable(), "key2": Variable()}
        assert handler._validate_dict(valid_dict, Variable)

        # Test invalid type
        with pytest.raises(ValueError) as excinfo:
            handler._validate_dict("not a dict", Variable)      # type: ignore #
        assert "must be a dictionary" in str(excinfo.value)

        # Test empty dictionary
        with pytest.raises(ValueError) as excinfo:
            handler._validate_dict({}, Variable)
        assert "cannot be empty" in str(excinfo.value)

    def test_clear(self) -> None:
        """*test_clear()* tests clearing handler state."""
        # Create handler with data
        handler = MonteCarloHandler(
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )

        # Configure distributions
        handler._config_distributions()

        # Clear handler
        handler.clear()

        # Test collections cleared
        assert len(handler._simulations) == 0
        assert len(handler._distributions) == 0
        assert len(handler._results) == 0
        assert len(handler._shared_cache) == 0

    def test_properties(self) -> None:
        """*test_properties()* tests property getters and setters."""
        # Create handler
        handler = MonteCarloHandler()

        # Test cat property
        assert handler.cat == "NUM"
        handler.cat = "SYM"
        assert handler.cat == "SYM"

        # Test experiments property
        assert handler.experiments == 1000
        handler.experiments = 2000
        assert handler.experiments == 2000

        # Test invalid experiments
        with pytest.raises(ValueError) as excinfo:
            handler.experiments = 0
        assert "must be positive" in str(excinfo.value)

    def test_get_simulation(self) -> None:
        """*test_get_simulation()* tests getting simulation by name."""
        # Create handler
        handler = MonteCarloHandler(
            _fwk="CUSTOM",
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )

        # Configure
        handler.config_simulations()

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
        handler = MonteCarloHandler(_variables=self.test_variables)

        # Configure distributions
        handler._config_distributions()

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
        handler = MonteCarloHandler(
            _idx=0,
            _fwk="CUSTOM",
            name="Test",
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
        handler = MonteCarloHandler(
            _fwk="CUSTOM",
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )
        handler.config_simulations()

        # Verify shared cache exists
        assert hasattr(handler, '_shared_cache')
        assert isinstance(handler._shared_cache, dict)

        # Verify cache for all variables
        for var_sym in handler._variables.keys():
            assert var_sym in handler._shared_cache
            assert handler._shared_cache[var_sym].shape == (handler._experiments, 1)

    def test_simulations_share_same_cache(self) -> None:
        """*test_simulations_share_same_cache()* tests all simulations reference same cache object."""
        handler = MonteCarloHandler(
            _fwk="CUSTOM",
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )
        handler.config_simulations()

        # Get cache IDs from all simulations
        cache_ids = [id(sim._simul_cache) for sim in handler._simulations.values()]

        # All should reference the same object
        assert len(set(cache_ids)) == 1
        assert cache_ids[0] == id(handler._shared_cache)

    def test_cache_values_consistent_across_coefficients(self) -> None:
        """*test_cache_values_consistent_across_coefficients()* tests same variable has identical values across coefficients."""
        handler = MonteCarloHandler(
            _fwk="CUSTOM",
            _variables=self.test_variables,
            _coefficients=self.test_coefficients,
            _experiments=5
        )
        handler.config_simulations()
        handler.simulate(n_samples=5)

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
        handler = MonteCarloHandler(
            _fwk="CUSTOM",
            _variables=self.test_variables,
            _coefficients=self.test_coefficients
        )
        handler.config_simulations()

        # Modify cache through handler
        test_var = 'd'
        test_idx = 2
        test_value = 99.99
        handler._shared_cache[test_var][test_idx, 0] = test_value

        # Verify all simulations see the change
        for sim in handler._simulations.values():
            cached = sim._get_cached_value(test_var, test_idx)
            assert cached == test_value
