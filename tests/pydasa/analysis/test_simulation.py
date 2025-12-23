# -*- coding: utf-8 -*-
"""
Module test_simulation.py
===========================================

Unit tests for MonteCarloSim class in PyDASA.

This module provides test cases for Monte Carlo simulation functionality
following the complete dimensional analysis workflow:
1. Create dimensional framework and schema
2. Define variables with distributions
3. Create and solve dimensional matrix
4. Run Monte Carlo simulations on each coefficient
"""

# Import testing packages
import unittest
import pytest
import random

# typying imports
from typing import Dict, Any

# Import numpy for numerical operations
import numpy as np

# Import core PyDASA modules
# from pydasa.core.fundamental import Dimension
from pydasa.core.parameter import Variable
from pydasa.dimensional.framework import DimSchema
from pydasa.dimensional.model import DimMatrix

# Import the module to test
from pydasa.analysis.simulation import MonteCarloSim

# Import related classes
from pydasa.buckingham.vashchy import Coefficient

# Import test data
from tests.pydasa.data.test_data import get_simulation_test_data

# Asserting module imports
assert MonteCarloSim
assert Coefficient
assert Variable
assert DimMatrix
assert DimSchema
assert get_simulation_test_data

# Number of experiments for simulations
N_EXP: int = 50


# ============================================================================
# Distribution Functions
# ============================================================================

def dist_uniform(a: float, b: float) -> float:
    """Generate uniform random value between a and b."""
    return random.uniform(a, b)


def dist_dependent(a: float, U: float) -> float:
    """Generate dependent random value based on U."""
    return random.uniform(a, 2 * U)


# ============================================================================
# Test Class
# ============================================================================

class TestMonteCarloSim(unittest.TestCase):
    """**TestMonteCarloSim** implements unit tests for the MonteCarloSim class.

    Args:
        unittest (TestCase): unittest.TestCase class for Python unit testing.
    """

    # ========================================================================
    # Type hints for class attributes
    # ========================================================================

    # Add type hints at class level
    dim_schema: DimSchema
    variables: Dict[str, Variable]
    dim_model: DimMatrix
    coefficients: Dict[str, Coefficient]
    dist_specs: Dict[str, Dict[str, Any]]
    test_data: Dict[str, Any]

    # ========================================================================
    # Fixtures and Setup
    # ========================================================================

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """*inject_fixtures()* injects global test parameters as a fixture."""
        # Load test data
        self.test_data = get_simulation_test_data()

        # Setup dimensional framework
        self._setup_dimensional_framework()

        # Setup variables
        self._setup_channel_flow_variables()

        # Setup dimensional model and solve
        self._setup_dimensional_model()

        # Setup distribution specs
        self._setup_distribution_specs()

    def _setup_dimensional_framework(self) -> None:
        """*_setup_dimensional_framework()* sets up custom dimensional framework."""
        # Get FDU list from test data
        self.fdu_list = self.test_data["FDU_LIST"]

        # Create and configure dimensional schema
        self.dim_schema = DimSchema(_fdu_lt=self.fdu_list, _fwk="CUSTOM")
        self.dim_schema.update_global_config()

    def _setup_channel_flow_variables(self) -> None:
        """*_setup_channel_flow_variables()* creates variables for planar channel flow."""
        # Get variable data from test data
        var_data = self.test_data["CHANNEL_FLOW_VARIABLES"]

        # Create Variable objects with distribution functions
        self.variables = {}
        for var_sym, var_config in var_data.items():
            # Create a copy to avoid modifying test data
            config = var_config.copy()

            # Add distribution functions based on variable
            if var_sym == "U":
                config["_dist_func"] = lambda: dist_uniform(0.0, 15.0)
            elif var_sym == "\\mu_{1}":
                config["_dist_func"] = lambda U, a=0.0, b=0.75: float(U) + dist_dependent(a, b)
            elif var_sym == "y_{2}":
                config["_dist_func"] = lambda: dist_uniform(0.0, 10.0)
            elif var_sym == "d":
                config["_dist_func"] = lambda: dist_uniform(0.0, 5.0)
            elif var_sym == "P":
                config["_dist_func"] = lambda: dist_uniform(0.0, 100000.0)
            elif var_sym == "v":
                config["_dist_func"] = lambda: dist_uniform(0.0, 1.0)

            # Create Variable object
            self.variables[var_sym] = Variable(**config)

    def _setup_dimensional_model(self) -> None:
        """*_setup_dimensional_model()* creates and solves dimensional model."""
        # Create dimensional model
        self.dim_model = DimMatrix(
            _fwk="CUSTOM",
            _idx=0,
            _framework=self.dim_schema
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
                "Check that solve_matrix() completed successfully."
            )

        if not isinstance(coefficients, dict):
            raise TypeError(
                f"Expected coefficients to be dict, got {type(coefficients)}"
            )

        if len(coefficients) == 0:
            raise ValueError(
                "Dimensional model produced empty coefficients dictionary. "
                "The model may not have enough variables or proper dimensional setup."
            )

        self.coefficients = coefficients

    def _setup_distribution_specs(self) -> None:
        """*_setup_distribution_specs()* creates distribution specifications."""
        U_var = self.variables["U"]
        mu_var = self.variables["\\mu_{1}"]

        self.dist_specs = {
            "U": {
                "depends": U_var.depends,
                "dtype": U_var.dist_type,
                "params": U_var.dist_params,
                "func": U_var.dist_func
            },
            "\\mu_{1}": {
                "depends": mu_var.depends,
                "dtype": mu_var.dist_type,
                "params": mu_var.dist_params,
                "func": mu_var.dist_func
            }
        }

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

        # Test dependent variable
        mu_var = self.variables["\\mu_{1}"]
        assert mu_var.depends == ["U"]
        assert mu_var._dist_func is not None

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

        # Now type checker knows coefficients is not None
        for pi_sym, coef in self.coefficients.items():
            assert coef is not None
            assert coef.pi_expr is not None
            assert coef._pi_expr is not None

    # ========================================================================
    # Monte Carlo Simulation Initialization Tests
    # ========================================================================

    def test_monte_carlo_creation_with_coefficient(self) -> None:
        """*test_monte_carlo_creation_with_coefficient()* tests MC sim creation."""
        # Get first coefficient
        assert self.coefficients is not None
        assert len(self.coefficients) > 0

        # Get first coefficient - type checker now knows this is valid
        first_pi = list(self.coefficients.keys())[0]
        coef = self.coefficients[first_pi]

        # Create Monte Carlo simulation
        mc_sim = MonteCarloSim(
            _idx=0,
            _sym=f"MC_{first_pi}",
            _fwk="CUSTOM",
            name=f"Monte Carlo for {first_pi}",
            description=f"Monte Carlo simulation for {first_pi}",
            _coefficient=coef,
            _experiments=N_EXP
        )

        assert mc_sim is not None
        assert mc_sim._coefficient == coef
        assert mc_sim._pi_expr == coef._pi_expr

    def test_monte_carlo_without_coefficient_fails(self) -> None:
        """*test_monte_carlo_without_coefficient_fails()* tests creation fails without coefficient."""
        with pytest.raises((ValueError, TypeError)):
            MonteCarloSim(
                _idx=0,
                _sym="MC_Test",
                _fwk="CUSTOM",
                _experiments=N_EXP
            )

    def test_monte_carlo_set_coefficient(self) -> None:
        """*test_monte_carlo_set_coefficient()* tests setting coefficient."""
        pi_keys = list(self.coefficients.keys())
        if len(pi_keys) < 2:
            pytest.skip("Need at least 2 coefficients for this test")

        coef1 = self.coefficients[pi_keys[0]]
        coef2 = self.coefficients[pi_keys[1]]

        # Create with first coefficient
        mc_sim = MonteCarloSim(_coefficient=coef1,
                               _experiments=N_EXP)

        # Change to second coefficient
        mc_sim.set_coefficient(coef2)

        assert mc_sim._coefficient == coef2
        assert mc_sim._pi_expr == coef2._pi_expr

    # ========================================================================
    # Expression Parsing Tests
    # ========================================================================

    def test_expression_parsing_from_coefficient(self) -> None:
        """*test_expression_parsing_from_coefficient()* tests parsing coefficient expressions."""
        for pi_sym, coef in self.coefficients.items():
            mc_sim = MonteCarloSim(_coefficient=coef,
                                   _experiments=N_EXP)

            assert mc_sim._sym_func is not None
            assert mc_sim._var_symbols is not None
            assert len(mc_sim._var_symbols) > 0

    def test_variable_extraction_from_expression(self) -> None:
        """*test_variable_extraction_from_expression()* tests extracting variables."""
        # Test with Pi_1 if it exists
        if "\\Pi_{1}" in self.coefficients:
            coef = self.coefficients["\\Pi_{1}"]
            mc_sim = MonteCarloSim(_coefficient=coef,
                                   _experiments=N_EXP)

            # Get variables from coefficient
            vars_in_coef = []
            if mc_sim.coefficient is not None:
                vars_in_coef = list(mc_sim.coefficient.var_dims.keys())

            assert len(vars_in_coef) > 0
            assert all(v in self.variables for v in vars_in_coef)

    # ========================================================================
    # Distribution Configuration Tests
    # ========================================================================

    def test_distribution_setup(self) -> None:
        """*test_distribution_setup()* tests setting up distributions."""
        if "\\Pi_{1}" not in self.coefficients:
            pytest.skip("Pi_1 coefficient not found")

        coef = self.coefficients["\\Pi_{1}"]
        mc_sim = MonteCarloSim(_coefficient=coef,
                               _experiments=N_EXP)

        # Get variables in this coefficient
        vars_in_coef = list(coef.var_dims.keys())

        # Set up distributions only for variables in this coefficient
        mc_sim._distributions = {
            k: v for k, v in self.dist_specs.items() if k in vars_in_coef
        }

        assert mc_sim._distributions is not None
        assert len(mc_sim._distributions) > 0

    def test_missing_distributions_raises_error(self) -> None:
        """*test_missing_distributions_raises_error()* tests validation fails without distributions."""
        if "\\Pi_{1}" not in self.coefficients:
            pytest.skip("Pi_1 coefficient not found")

        coef = self.coefficients["\\Pi_{1}"]
        mc_sim = MonteCarloSim(_coefficient=coef,
                               _variables=self.variables,
                               _experiments=N_EXP)

        # Don't set distributions
        with pytest.raises(ValueError) as excinfo:
            mc_sim.run()

        assert "Missing distributions" in str(excinfo.value)

    # ========================================================================
    # Monte Carlo Simulation Execution Tests
    # ========================================================================

    def test_run_simulation_on_each_coefficient(self) -> None:
        """*test_run_simulation_on_each_coefficient()* tests running MC on all coefficients."""
        assert self.coefficients is not None

        for pi_sym, coef in self.coefficients.items():
            mc_sim = MonteCarloSim(
                _idx=0,
                _sym=f"MC_{pi_sym}",
                _fwk="CUSTOM",
                name=f"Monte Carlo {pi_sym}",
                _coefficient=coef,
                _variables=self.variables,
                _experiments=N_EXP
            )

            # Get variables in this coefficient
            vars_in_coef = list(coef.var_dims.keys())

            # Set up distributions only for variables in this coefficient
            mc_sim._distributions = {
                k: v for k, v in self.dist_specs.items() if k in vars_in_coef
            }

            # Skip if not all variables have distributions
            missing = [v for v in vars_in_coef if v not in mc_sim._distributions]
            if missing:
                continue

            try:
                mc_sim.run()

                # Verify results
                assert mc_sim._results.size > 0
                assert len(mc_sim._results) == N_EXP
                assert not all(np.isnan(mc_sim._results.flatten()))

            except ValueError as e:
                pytest.fail(f"Simulation failed for {pi_sym}: {str(e)}")

    def test_run_simulation_pi_1(self) -> None:
        """*test_run_simulation_pi_1()* tests running MC simulation on Pi_1."""
        if "\\Pi_{1}" not in self.coefficients:
            pytest.skip("Pi_1 coefficient not found")

        coef = self.coefficients["\\Pi_{1}"]

        mc_sim = MonteCarloSim(
            _idx=0,
            _sym="MC_Pi_1",
            _fwk="CUSTOM",
            name="Monte Carlo Pi_1",
            _coefficient=coef,
            _variables=self.variables,
            _experiments=N_EXP
        )

        # Get variables in coefficient
        vars_in_coef = list(coef.var_dims.keys())

        # Set distributions
        mc_sim._distributions = {
            k: v for k, v in self.dist_specs.items()
            if k in vars_in_coef
        }

        # Validate distributions
        missing = [v for v in vars_in_coef if v not in mc_sim._distributions]
        if missing:
            pytest.skip(f"Missing distributions for: {missing}")

        mc_sim.run()

        # Verify results
        assert mc_sim._results.size == N_EXP
        assert not all(np.isnan(mc_sim._results.flatten()))

    def test_run_simulation_with_dependencies(self) -> None:
        """*test_run_simulation_with_dependencies()* tests simulation with dependent variables."""
        # Find a coefficient that uses both U and mu_1
        target_coef = None
        for pi_sym, coef in self.coefficients.items():
            vars_in_coef = list(coef.var_dims.keys())
            if "U" in vars_in_coef and "\\mu_{1}" in vars_in_coef:
                target_coef = coef
                break

        if target_coef is None:
            pytest.skip("No coefficient found with both U and mu_1")

        mc_sim = MonteCarloSim(
            _coefficient=target_coef,
            _variables=self.variables,
            _experiments=N_EXP
        )

        # Set distributions
        vars_in_coef = list(target_coef.var_dims.keys())
        mc_sim._distributions = {
            k: v for k, v in self.dist_specs.items()
            if k in vars_in_coef
        }

        # Set dependencies
        mc_sim._dependencies = {
            k: v.depends for k, v in self.variables.items()
            if k in vars_in_coef
        }

        # Skip if missing distributions
        if not all(v in mc_sim._distributions for v in vars_in_coef):
            pytest.skip("Missing required distributions")

        mc_sim.run()

        assert mc_sim._results.size == N_EXP
        assert not all(np.isnan(mc_sim._results.flatten()))

    # ========================================================================
    # Statistics Tests
    # ========================================================================

    def test_calculate_statistics(self) -> None:
        """*test_calculate_statistics()* tests statistical calculations."""
        if "\\Pi_{1}" not in self.coefficients:
            pytest.skip("Pi_1 coefficient not found")

        coef = self.coefficients["\\Pi_{1}"]
        mc_sim = MonteCarloSim(_coefficient=coef,
                               _variables=self.variables,
                               _experiments=N_EXP)

        # Setup distributions
        vars_in_coef = list(coef.var_dims.keys())
        mc_sim._distributions = {
            k: v for k, v in self.dist_specs.items()
            if k in vars_in_coef
        }

        if not all(v in mc_sim._distributions for v in vars_in_coef):
            pytest.skip("Missing distributions")

        mc_sim.run()

        # Test statistics
        assert not np.isnan(mc_sim.mean)
        assert not np.isnan(mc_sim.std_dev)
        assert not np.isnan(mc_sim.variance)
        assert not np.isnan(mc_sim.median)
        assert mc_sim._count == N_EXP

    def test_statistics_property(self) -> None:
        """*test_statistics_property()* tests statistics dictionary property."""
        if "\\Pi_{1}" not in self.coefficients:
            pytest.skip("Pi_1 coefficient not found")

        coef = self.coefficients["\\Pi_{1}"]
        mc_sim = MonteCarloSim(_coefficient=coef,
                               _variables=self.variables,
                               _experiments=N_EXP)
        # Setup and run
        vars_in_coef = list(coef.var_dims.keys())
        mc_sim._distributions = {
            k: v for k, v in self.dist_specs.items()
            if k in vars_in_coef
        }

        if not all(v in mc_sim._distributions for v in vars_in_coef):
            pytest.skip("Missing distributions")

        mc_sim.run()

        # Get statistics
        stats = mc_sim.statistics

        assert isinstance(stats, dict)
        assert "mean" in stats
        assert "std_dev" in stats
        assert "variance" in stats
        assert "median" in stats
        assert "min" in stats
        assert "max" in stats
        assert "count" in stats

    def test_confidence_intervals(self) -> None:
        """*test_confidence_intervals()* tests confidence interval calculation."""
        if "\\Pi_{1}" not in self.coefficients:
            pytest.skip("Pi_1 coefficient not found")

        coef = self.coefficients["\\Pi_{1}"]
        mc_sim = MonteCarloSim(_coefficient=coef,
                               _variables=self.variables,
                               _experiments=N_EXP)
        # Setup and run
        vars_in_coef = list(coef.var_dims.keys())
        mc_sim._distributions = {
            k: v for k, v in self.dist_specs.items()
            if k in vars_in_coef
        }

        if not all(v in mc_sim._distributions for v in vars_in_coef):
            pytest.skip("Missing distributions")

        mc_sim.run()

        # Test different confidence levels
        for conf_level in [0.90, 0.95, 0.99]:
            lower, upper = mc_sim.get_confidence_interval(conf=conf_level)
            assert lower < upper
            assert isinstance(lower, float)
            assert isinstance(upper, float)

    # ========================================================================
    # Results Extraction Tests
    # ========================================================================

    def test_extract_results(self) -> None:
        """*test_extract_results()* tests extracting simulation results."""
        if "\\Pi_{1}" not in self.coefficients:
            pytest.skip("Pi_1 coefficient not found")

        coef = self.coefficients["\\Pi_{1}"]
        mc_sim = MonteCarloSim(_coefficient=coef,
                               _variables=self.variables,
                               _experiments=N_EXP)
        # Setup and run
        vars_in_coef = list(coef.var_dims.keys())
        mc_sim._distributions = {
            k: v for k, v in self.dist_specs.items()
            if k in vars_in_coef
        }

        if not all(v in mc_sim._distributions for v in vars_in_coef):
            pytest.skip("Missing distributions")

        mc_sim.run()

        # Extract results
        results = mc_sim.extract_results()

        assert isinstance(results, dict)
        assert len(results) > 0

        # Should have coefficient results
        assert coef.sym in results
        assert len(results[coef.sym]) == N_EXP

    def test_extract_results_keys(self) -> None:
        """*test_extract_results_keys()* tests result dictionary keys."""
        if "\\Pi_{1}" not in self.coefficients:
            pytest.skip("Pi_1 coefficient not found")

        coef = self.coefficients["\\Pi_{1}"]
        mc_sim = MonteCarloSim(_coefficient=coef,
                               _variables=self.variables,
                               _experiments=N_EXP)

        # Setup and run
        vars_in_coef = list(coef.var_dims.keys())
        mc_sim._distributions = {
            k: v for k, v in self.dist_specs.items()
            if k in vars_in_coef
        }

        if not all(v in mc_sim._distributions for v in vars_in_coef):
            pytest.skip("Missing distributions")

        mc_sim.run()

        results = mc_sim.extract_results()
        keys = list(results.keys())

        # Coefficient symbol should be in keys
        assert coef.sym in keys

        # Variable samples should be in keys with format "var@coef"
        for var in vars_in_coef:
            expected_key = f"{var}@{coef.sym}"
            assert expected_key in keys

    # ========================================================================
    # Property Tests
    # ========================================================================

    def test_experiments_property(self) -> None:
        """*test_experiments_property()* tests experiments property."""
        coef = list(self.coefficients.values())[0]
        mc_sim = MonteCarloSim(_coefficient=coef)

        assert mc_sim.experiments == 1000

        mc_sim.experiments = 500
        assert mc_sim.experiments == 500

    def test_experiments_invalid_value(self) -> None:
        """*test_experiments_invalid_value()* tests invalid experiments value."""
        coef = list(self.coefficients.values())[0]
        mc_sim = MonteCarloSim(_coefficient=coef)

        with pytest.raises(ValueError):
            mc_sim.experiments = -1

        with pytest.raises(ValueError):
            mc_sim.experiments = 0

    def test_coefficient_property(self) -> None:
        """*test_coefficient_property()* tests coefficient property."""
        coef = list(self.coefficients.values())[0]
        mc_sim = MonteCarloSim(_coefficient=coef)

        retrieved_coef = mc_sim.coefficient
        assert retrieved_coef == coef

    def test_variables_property(self) -> None:
        """*test_variables_property()* tests variables property."""
        coef = list(self.coefficients.values())[0]
        mc_sim = MonteCarloSim(_coefficient=coef)
        
        vars_dict = mc_sim.variables
        assert isinstance(vars_dict, dict)

    # ========================================================================
    # Cache Management Tests
    # ========================================================================

    def test_cache_initialization(self) -> None:
        """*test_cache_initialization()* tests cache is initialized."""
        coef = list(self.coefficients.values())[0]
        mc_sim = MonteCarloSim(_coefficient=coef,
                               _experiments=N_EXP)

        assert mc_sim._simul_cache is not None
        assert isinstance(mc_sim._simul_cache, dict)

    def test_cache_operations(self) -> None:
        """*test_cache_operations()* tests cache get/set operations."""
        coef = list(self.coefficients.values())[0]
        mc_sim = MonteCarloSim(_coefficient=coef,
                               _variables=self.variables,
                               _experiments=N_EXP)

        if len(mc_sim._variables) == 0:
            pytest.skip("No variables in coefficient")

        var_sym = list(mc_sim._variables.keys())[0]

        # Test setting value
        mc_sim._set_cached_value(var_sym, 0, 5.0)

        # Test getting value
        cached = mc_sim._get_cached_value(var_sym, 0)
        assert cached == 5.0

    # ========================================================================
    # Memory Management Tests
    # ========================================================================

    def test_reset_memory(self) -> None:
        """*test_reset_memory()* tests resetting simulation memory."""
        if "\\Pi_{1}" not in self.coefficients:
            pytest.skip("Pi_1 coefficient not found")

        coef = self.coefficients["\\Pi_{1}"]
        mc_sim = MonteCarloSim(_coefficient=coef,
                               _variables=self.variables,
                               _experiments=N_EXP)

        # Setup and run
        vars_in_coef = list(coef.var_dims.keys())
        mc_sim._distributions = {
            k: v for k, v in self.dist_specs.items()
            if k in vars_in_coef
        }

        if not all(v in mc_sim._distributions for v in vars_in_coef):
            pytest.skip("Missing distributions")

        mc_sim.run()

        # Reset
        mc_sim._reset_memory()

        # Check results are NaN
        assert all(np.isnan(mc_sim._results.flatten()))
