# -*- coding: utf-8 -*-
"""
Module test_simulation.py
===========================================

Unit tests for MonteCarlo class in PyDASA.

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
from pydasa.elements.parameter import Variable
from pydasa.dimensional.vaschy import Schema
from pydasa.dimensional.model import Matrix

# Import the module to test
from pydasa.analysis.simulation import MonteCarlo, SimulationMode

# Import related classes
from pydasa.dimensional.buckingham import Coefficient

# Import test data
from tests.pydasa.data.test_data import get_simulation_test_data

# Asserting module imports
assert MonteCarlo
assert SimulationMode
assert Coefficient
assert Variable
assert Matrix
assert Schema
assert get_simulation_test_data

# Number of experiments for simulations
N_EXP: int = 100


# ============================================================================
# Distribution Functions
# ============================================================================

def dist_uniform(a: float, b: float) -> float:
    """Generate uniform random value between a and b."""
    return random.uniform(a, b)


def dist_dependent(a: float, U: float) -> float:
    """Generate dependent random value based on U."""
    return random.uniform(a, 2 * U)


def _is_none_or_nan(value: float | int | None) -> bool:
    """Helper to check if value is None or NaN.

    Args:
        value: Value to check (can be float, int, or None)

    Returns:
        True if value is None or NaN, False otherwise
    """
    return value is None or (isinstance(value, (float, int)) and np.isnan(value))


# ============================================================================
# Test Class
# ============================================================================

class TestMonteCarlo(unittest.TestCase):
    """**TestMonteCarlo** implements unit tests for the MonteCarlo class.

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
        self.dim_schema = Schema(_fdu_lt=self.fdu_list, _fwk="CUSTOM")

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
            var = Variable(**config)
            # Set schema and prepare dimensions for CUSTOM framework variables
            var._schema = self.dim_schema
            var._prepare_dims()
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
        self.dim_model.relevance_lt = self.variables

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
        # Create distribution specs for all variables
        self.dist_specs = {}
        for var_sym, var_obj in self.variables.items():
            self.dist_specs[var_sym] = {
                "depends": var_obj.depends,
                "dtype": var_obj.dist_type,
                "params": var_obj.dist_params,
                "func": var_obj.dist_func
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
        assert len(self.dim_model.relevance_lt) == 6

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
        mc_sim = MonteCarlo(
            _idx=0,
            _sym=f"MC_{first_pi}",
            _fwk="CUSTOM",
            _name=f"Monte Carlo for {first_pi}",
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
            MonteCarlo(
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
        mc_sim = MonteCarlo(_coefficient=coef1,
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
            mc_sim = MonteCarlo(_coefficient=coef,
                                _experiments=N_EXP)

            assert mc_sim._sym_func is not None
            assert mc_sim._var_symbols is not None
            assert len(mc_sim._var_symbols) > 0

    def test_variable_extraction_from_expression(self) -> None:
        """*test_variable_extraction_from_expression()* tests extracting variables."""
        # Test with Pi_1 if it exists
        if "\\Pi_{1}" in self.coefficients:
            coef = self.coefficients["\\Pi_{1}"]
            mc_sim = MonteCarlo(_coefficient=coef,
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
        mc_sim = MonteCarlo(_coefficient=coef,
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
        mc_sim = MonteCarlo(_coefficient=coef,
                            _variables=self.variables,
                            _experiments=N_EXP)

        # Don't set distributions
        with pytest.raises(ValueError) as excinfo:
            mc_sim.run()

        assert "Distributions must be provided" in str(excinfo.value) or "Missing distributions" in str(excinfo.value)

    # ========================================================================
    # Monte Carlo Simulation Execution Tests
    # ========================================================================

    def test_run_simulation_on_each_coefficient(self) -> None:
        """*test_run_simulation_on_each_coefficient()* tests running MC on all coefficients."""
        assert self.coefficients is not None

        for pi_sym, coef in self.coefficients.items():
            mc_sim = MonteCarlo(
                _idx=0,
                _sym=f"MC_{pi_sym}",
                _fwk="CUSTOM",
                _name=f"Monte Carlo {pi_sym}",
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

        mc_sim = MonteCarlo(
            _idx=0,
            _sym="MC_Pi_1",
            _fwk="CUSTOM",
            _name="Monte Carlo Pi_1",
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
        # Find a coefficient that uses mu_1 (which depends on U)
        # Since U and mu_1 have the same dimensions, they get separate coefficients
        # We test with mu_1's coefficient since mu_1 depends on U
        target_coef = None
        for pi_sym, coef in self.coefficients.items():
            vars_in_coef = list(coef.var_dims.keys())
            if "\\mu_{1}" in vars_in_coef:
                target_coef = coef
                break

        if target_coef is None:
            pytest.skip("No coefficient found with mu_1 (dependent variable)")

        mc_sim = MonteCarlo(
            _coefficient=target_coef,
            _variables=self.variables,
            _experiments=N_EXP
        )

        # Set distributions for variables in coefficient
        vars_in_coef = list(target_coef.var_dims.keys())
        mc_sim._distributions = {
            k: v for k, v in self.dist_specs.items()
            if k in vars_in_coef
        }

        # Also need distribution for U since mu_1 depends on it
        if "U" not in mc_sim._distributions and "U" in self.dist_specs:
            mc_sim._distributions["U"] = self.dist_specs["U"]

        # Set dependencies
        mc_sim._dependencies = {
            k: v.depends for k, v in self.variables.items()
            if k in vars_in_coef or k == "U"  # Include U as it's a dependency
        }

        # Skip if missing distributions
        required_vars = vars_in_coef + ["U"]  # U is required for mu_1
        if not all(v in mc_sim._distributions for v in required_vars):
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
        mc_sim = MonteCarlo(_coefficient=coef,
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
        assert mc_sim.mean is not None
        assert mc_sim.dev is not None
        assert mc_sim.median is not None
        assert not np.isnan(mc_sim.mean)
        assert not np.isnan(mc_sim.dev)
        assert not np.isnan(mc_sim.median)
        assert mc_sim.count == N_EXP

    def test_statistics_property(self) -> None:
        """*test_statistics_property()* tests statistics dictionary property."""
        if "\\Pi_{1}" not in self.coefficients:
            pytest.skip("Pi_1 coefficient not found")

        coef = self.coefficients["\\Pi_{1}"]
        mc_sim = MonteCarlo(_coefficient=coef,
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
        assert "dev" in stats

        assert "median" in stats
        assert "min" in stats
        assert "max" in stats
        assert "count" in stats

    def test_confidence_intervals(self) -> None:
        """*test_confidence_intervals()* tests confidence interval calculation."""
        if "\\Pi_{1}" not in self.coefficients:
            pytest.skip("Pi_1 coefficient not found")

        coef = self.coefficients["\\Pi_{1}"]
        mc_sim = MonteCarlo(_coefficient=coef,
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
        mc_sim = MonteCarlo(_coefficient=coef,
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
        mc_sim = MonteCarlo(_coefficient=coef,
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
        mc_sim = MonteCarlo(_coefficient=coef)

        assert mc_sim.experiments == -1  # Default value

        mc_sim.experiments = 500
        assert mc_sim.experiments == 500

    def test_experiments_invalid_value(self) -> None:
        """*test_experiments_invalid_value()* tests invalid experiments value."""
        coef = list(self.coefficients.values())[0]
        mc_sim = MonteCarlo(_coefficient=coef)

        with pytest.raises(ValueError):
            mc_sim.experiments = -1

        with pytest.raises(ValueError):
            mc_sim.experiments = 0

    def test_coefficient_property(self) -> None:
        """*test_coefficient_property()* tests coefficient property."""
        coef = list(self.coefficients.values())[0]
        mc_sim = MonteCarlo(_coefficient=coef)

        retrieved_coef = mc_sim.coefficient
        assert retrieved_coef == coef

    def test_variables_property(self) -> None:
        """*test_variables_property()* tests variables property."""
        coef = list(self.coefficients.values())[0]
        mc_sim = MonteCarlo(_coefficient=coef)

        vars_dict = mc_sim.variables
        assert isinstance(vars_dict, dict)

    def test_results_property(self) -> None:
        """*test_results_property()* tests results property getter."""
        coef = list(self.coefficients.values())[0]
        mc_sim = MonteCarlo(_coefficient=coef,
                            _variables=self.variables,
                            _experiments=N_EXP)
        mc_sim.distributions = self.dist_specs

        # Before running simulation, should raise error
        with pytest.raises(ValueError):
            _ = mc_sim.results

        # After running simulation, should return results
        mc_sim.run(iters=N_EXP, mode="DIST")
        results = mc_sim.results
        assert isinstance(results, np.ndarray)
        assert len(results) == N_EXP

    def test_data_property_getter(self) -> None:
        """*test_data_property_getter()* tests data property getter."""
        coef = list(self.coefficients.values())[0]
        mc_sim = MonteCarlo(_coefficient=coef,
                            _variables=self.variables,
                            _experiments=N_EXP)
        mc_sim.distributions = self.dist_specs

        # Before running simulation, should raise error
        with pytest.raises(ValueError):
            _ = mc_sim.data

        # After running simulation, should return data
        mc_sim.run(iters=N_EXP, mode="DIST")
        data = mc_sim.data
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_data_property_setter(self) -> None:
        """*test_data_property_setter()* tests data property setter."""
        coef = list(self.coefficients.values())[0]
        mc_sim = MonteCarlo(_coefficient=coef)

        # Set data with lists
        test_data = {
            "U": [1.0, 2.0, 3.0],
            "h": [0.1, 0.2, 0.3]
        }
        mc_sim.data = test_data     # type: ignore

        # Verify data was converted to numpy arrays
        retrieved_data = mc_sim.data
        for key, values in retrieved_data.items():
            assert isinstance(values, np.ndarray)

    def test_distributions_property(self) -> None:
        """*test_distributions_property()* tests distributions property."""
        coef = list(self.coefficients.values())[0]
        mc_sim = MonteCarlo(_coefficient=coef)

        # Test getter with empty distributions
        dists = mc_sim.distributions
        assert isinstance(dists, dict)

        # Test setter
        mc_sim.distributions = self.dist_specs
        dists = mc_sim.distributions
        assert len(dists) > 0

    def test_dependencies_property(self) -> None:
        """*test_dependencies_property()* tests dependencies property."""
        coef = list(self.coefficients.values())[0]
        mc_sim = MonteCarlo(_coefficient=coef)

        # Test getter
        deps = mc_sim.dependencies
        assert isinstance(deps, dict)

        # Test setter
        test_deps = {"\\mu_{1}": ["U"]}
        mc_sim.dependencies = test_deps
        deps = mc_sim.dependencies
        assert "\\mu_{1}" in deps
        assert deps["\\mu_{1}"] == ["U"]

    def test_cat_property(self) -> None:
        """*test_cat_property()* tests cat property."""
        coef = list(self.coefficients.values())[0]
        mc_sim = MonteCarlo(_coefficient=coef)

        # Test getter
        cat = mc_sim.cat
        assert isinstance(cat, str)

        # Test setter with valid values
        mc_sim.cat = "DIST"
        assert mc_sim.cat == "DIST"

        mc_sim.cat = "DATA"
        assert mc_sim.cat == "DATA"

        # Test setter with invalid value
        with pytest.raises(ValueError):
            mc_sim.cat = "INVALID"

    def test_count_property_getter(self) -> None:
        """*test_count_property_getter()* tests count property getter."""
        coef = list(self.coefficients.values())[0]
        mc_sim = MonteCarlo(_coefficient=coef,
                            _variables=self.variables,
                            _experiments=N_EXP)
        mc_sim.distributions = self.dist_specs

        # Before running simulation, should raise error
        with pytest.raises(ValueError):
            _ = mc_sim.count

        # After running simulation, should return count
        mc_sim.run(iters=N_EXP, mode="DIST")
        count = mc_sim.count
        assert count == N_EXP

    def test_count_property_setter(self) -> None:
        """*test_count_property_setter()* tests count property setter."""
        coef = list(self.coefficients.values())[0]
        mc_sim = MonteCarlo(_coefficient=coef)

        # Test setting valid count
        mc_sim.count = 100
        # Note: _count is set, but count property getter requires results

        # Test setting invalid count
        with pytest.raises(ValueError):
            mc_sim.count = -1

    def test_inherited_statistics_properties(self) -> None:
        """*test_inherited_statistics_properties()* tests inherited BoundsSpecs properties."""
        coef = list(self.coefficients.values())[0]
        mc_sim = MonteCarlo(_coefficient=coef,
                            _variables=self.variables,
                            _experiments=N_EXP)
        mc_sim.distributions = self.dist_specs

        # Run simulation
        mc_sim.run(iters=N_EXP, mode="DIST")

        # Test inherited properties from BoundsSpecs
        assert isinstance(mc_sim.mean, (float, np.floating))
        assert isinstance(mc_sim.median, (float, np.floating))
        assert isinstance(mc_sim.dev, (float, np.floating))
        assert isinstance(mc_sim.min, (float, np.floating))
        assert isinstance(mc_sim.max, (float, np.floating))

        # Verify values are not NaN
        assert not np.isnan(mc_sim.mean)
        assert not np.isnan(mc_sim.median)
        assert not np.isnan(mc_sim.dev)
        assert not np.isnan(mc_sim.min)
        assert not np.isnan(mc_sim.max)

    # ========================================================================
    # Cache Management Tests
    # ========================================================================

    def test_cache_initialization(self) -> None:
        """*test_cache_initialization()* tests cache is initialized."""
        coef = list(self.coefficients.values())[0]
        mc_sim = MonteCarlo(_coefficient=coef,
                            _experiments=N_EXP)

        assert mc_sim._simul_cache is not None
        assert isinstance(mc_sim._simul_cache, dict)

    def test_cache_operations(self) -> None:
        """*test_cache_operations()* tests cache get/set operations."""
        coef = list(self.coefficients.values())[0]
        mc_sim = MonteCarlo(_coefficient=coef,
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
        mc_sim = MonteCarlo(_coefficient=coef,
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

    # ========================================================================
    # Simulation Mode Tests
    # ========================================================================

    def test_simulation_mode_enum(self) -> None:
        """*test_simulation_mode_enum()* tests SimulationMode enum values."""
        assert SimulationMode.DIST.value == "DIST"
        assert SimulationMode.DATA.value == "DATA"

    def test_run_with_generate_mode_explicit(self) -> None:
        """*test_run_with_generate_mode_explicit()* tests running with explicit generate mode."""
        if "\\Pi_{1}" not in self.coefficients:
            pytest.skip("Pi_1 coefficient not found")

        coef = self.coefficients["\\Pi_{1}"]
        mc_sim = MonteCarlo(_coefficient=coef,
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

        # Run with explicit DIST mode
        mc_sim.run(mode=SimulationMode.DIST)

        assert mc_sim._results.size == N_EXP
        assert not all(np.isnan(mc_sim._results.flatten()))

    def test_run_with_generate_mode_string(self) -> None:
        """*test_run_with_generate_mode_string()* tests running with mode='generate' string."""
        if "\\Pi_{1}" not in self.coefficients:
            pytest.skip("Pi_1 coefficient not found")

        coef = self.coefficients["\\Pi_{1}"]
        mc_sim = MonteCarlo(_coefficient=coef,
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

        # Run with string mode
        mc_sim.run(mode="dist")

        assert mc_sim._results.size == N_EXP
        assert not all(np.isnan(mc_sim._results.flatten()))

    def test_run_with_data_mode(self) -> None:
        """*test_run_with_data_mode()* tests running simulation with pre-existing data."""
        if "\\Pi_{1}" not in self.coefficients:
            pytest.skip("Pi_1 coefficient not found")

        coef = self.coefficients["\\Pi_{1}"]
        vars_in_coef = list(coef.var_dims.keys())

        # Generate sample data for variables
        n_samples = 50
        for var_sym in vars_in_coef:
            if var_sym in self.variables:
                var = self.variables[var_sym]
                # Create sample data
                var._data = np.random.uniform(0.1, 10.0, n_samples)

        mc_sim = MonteCarlo(_coefficient=coef,
                            _variables=self.variables,
                            _experiments=n_samples)

        # Run with DATA mode
        mc_sim.run(mode=SimulationMode.DATA)

        assert mc_sim._results.size == n_samples
        assert not all(np.isnan(mc_sim._results.flatten()))
        assert mc_sim._experiments == n_samples

    def test_run_with_data_mode_string(self) -> None:
        """*test_run_with_data_mode_string()* tests running with mode='data' string."""
        if "\\Pi_{1}" not in self.coefficients:
            pytest.skip("Pi_1 coefficient not found")

        coef = self.coefficients["\\Pi_{1}"]
        vars_in_coef = list(coef.var_dims.keys())

        # Generate sample data for variables
        n_samples = 50
        for var_sym in vars_in_coef:
            if var_sym in self.variables:
                var = self.variables[var_sym]
                var._data = np.random.uniform(0.1, 10.0, n_samples)

        mc_sim = MonteCarlo(_coefficient=coef,
                            _variables=self.variables,
                            _experiments=n_samples)

        # Run with string mode
        mc_sim.run(mode="data")

        assert mc_sim._results.size == n_samples
        assert not all(np.isnan(mc_sim._results.flatten()))

    def test_run_data_mode_no_data_raises_error(self) -> None:
        """*test_run_data_mode_no_data_raises_error()* tests error when variables have no data."""
        if "\\Pi_{1}" not in self.coefficients:
            pytest.skip("Pi_1 coefficient not found")

        coef = self.coefficients["\\Pi_{1}"]
        mc_sim = MonteCarlo(_coefficient=coef,
                            _variables=self.variables,
                            _experiments=N_EXP)

        # Don't set any variable data
        with pytest.raises(ValueError) as excinfo:
            mc_sim.run(mode="data")

        assert "has no data" in str(excinfo.value)

    def test_run_data_mode_inconsistent_lengths_raises_error(self) -> None:
        """*test_run_data_mode_inconsistent_lengths_raises_error()* tests error with inconsistent data lengths."""
        if "\\Pi_{1}" not in self.coefficients:
            pytest.skip("Pi_1 coefficient not found")

        coef = self.coefficients["\\Pi_{1}"]
        vars_in_coef = list(coef.var_dims.keys())

        if len(vars_in_coef) < 2:
            pytest.skip("Need at least 2 variables for this test")

        # Set different data lengths for variables
        for idx, var_sym in enumerate(vars_in_coef):
            if var_sym in self.variables:
                var = self.variables[var_sym]
                # Create different length arrays
                length = 50 + idx * 10
                var._data = np.random.uniform(0.1, 10.0, length)

        mc_sim = MonteCarlo(_coefficient=coef,
                            _variables=self.variables,
                            _experiments=100)

        with pytest.raises(ValueError) as excinfo:
            mc_sim.run(mode="data")

        assert "data points" in str(excinfo.value) or "inconsistent" in str(excinfo.value)

    def test_run_data_mode_fewer_points_than_experiments(self) -> None:
        """*test_run_data_mode_fewer_points_than_experiments()* tests adjustment when data < experiments."""
        if "\\Pi_{1}" not in self.coefficients:
            pytest.skip("Pi_1 coefficient not found")

        coef = self.coefficients["\\Pi_{1}"]
        vars_in_coef = list(coef.var_dims.keys())

        # Set data with fewer points than requested experiments
        n_samples = 30
        for var_sym in vars_in_coef:
            if var_sym in self.variables:
                var = self.variables[var_sym]
                var._data = np.random.uniform(0.1, 10.0, n_samples)

        mc_sim = MonteCarlo(_coefficient=coef,
                            _variables=self.variables,
                            _experiments=100)  # Request more than available

        # Should run with available data (adjust experiments to match data)
        mc_sim.run(iters=n_samples, mode="data")

        # Should have run with available data points
        assert mc_sim._experiments == n_samples
        assert mc_sim._results.size == n_samples

    def test_invalid_mode_raises_error(self) -> None:
        """*test_invalid_mode_raises_error()* tests error with invalid mode string."""
        if "\\Pi_{1}" not in self.coefficients:
            pytest.skip("Pi_1 coefficient not found")

        coef = self.coefficients["\\Pi_{1}"]
        mc_sim = MonteCarlo(_coefficient=coef,
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

        with pytest.raises(ValueError) as excinfo:
            mc_sim.run(mode="invalid_mode")

        assert "Invalid cat" in str(excinfo.value) or "Invalid mode" in str(excinfo.value)

    def test_data_mode_results_match_input_order(self) -> None:
        """*test_data_mode_results_match_input_order()* tests that data mode preserves input order."""
        if "\\Pi_{1}" not in self.coefficients:
            pytest.skip("Pi_1 coefficient not found")

        coef = self.coefficients["\\Pi_{1}"]
        vars_in_coef = list(coef.var_dims.keys())

        # Set specific data values
        n_samples = 10
        for var_sym in vars_in_coef:
            if var_sym in self.variables:
                var = self.variables[var_sym]
                # Create sequential data for verification
                var._data = np.linspace(1.0, 10.0, n_samples)

        mc_sim = MonteCarlo(_coefficient=coef,
                            _variables=self.variables,
                            _experiments=n_samples)

        mc_sim.run(mode="data")

        # Verify that input data was used in order
        assert len(mc_sim._data) == len(vars_in_coef)
        for var_sym in vars_in_coef:
            assert var_sym in mc_sim._data
            assert len(mc_sim._data[var_sym]) == n_samples
            # Check that data was used
            assert not np.isnan(mc_sim._data[var_sym][0])

    # ========================================================================
    # Summary Property Tests
    # ========================================================================

    def test_summary_property(self) -> None:
        """*test_summary_property()* tests summary property returns correct statistics."""
        if "\\Pi_{1}" not in self.coefficients:
            pytest.skip("Pi_1 coefficient not found")

        coef = self.coefficients["\\Pi_{1}"]
        vars_in_coef = list(coef.var_dims.keys())

        mc_sim = MonteCarlo(
            _coefficient=coef,
            _variables=self.variables,
            _experiments=N_EXP
        )

        # Set up distributions
        mc_sim._distributions = {
            k: v for k, v in self.dist_specs.items() if k in vars_in_coef
        }

        mc_sim.run()

        # Test summary property
        summary = mc_sim.summary
        assert summary is not None
        assert isinstance(summary, dict)
        assert "mean" in summary
        assert "median" in summary
        assert "dev" in summary
        assert "min" in summary
        assert "max" in summary
        assert "count" in summary
        assert summary["count"] == N_EXP

    def test_summary_property_without_results(self) -> None:
        """*test_summary_property_without_results()* tests summary with no results."""
        if "\\Pi_{1}" not in self.coefficients:
            pytest.skip("Pi_1 coefficient not found")

        coef = self.coefficients["\\Pi_{1}"]
        mc_sim = MonteCarlo(_coefficient=coef,
                            _experiments=N_EXP)

        # Access summary without running simulation - should have NaN values
        summary = mc_sim.summary
        assert summary is not None
        assert isinstance(summary, dict)
        # Statistics should be NaN or default values before running
        assert _is_none_or_nan(summary["mean"])
        assert _is_none_or_nan(summary["median"])
        assert _is_none_or_nan(summary["dev"])
        assert _is_none_or_nan(summary["min"])
        assert _is_none_or_nan(summary["max"])

    # ========================================================================
    # Clear Method Tests
    # ========================================================================

    def test_clear_method(self) -> None:
        """*test_clear_method()* tests clear method resets all attributes."""
        if "\\Pi_{1}" not in self.coefficients:
            pytest.skip("Pi_1 coefficient not found")

        coef = self.coefficients["\\Pi_{1}"]
        vars_in_coef = list(coef.var_dims.keys())

        mc_sim = MonteCarlo(
            _idx=5,
            _sym="MC_Test",
            _fwk="CUSTOM",
            _name="Test Simulation",
            _coefficient=coef,
            _variables=self.variables,
            _experiments=N_EXP
        )

        # Set up and run simulation
        mc_sim._distributions = {
            k: v for k, v in self.dist_specs.items() if k in vars_in_coef
        }
        mc_sim.run()

        # Verify simulation has results
        assert mc_sim._results.size > 0
        assert len(mc_sim._data) > 0

        # Clear the simulation
        mc_sim.clear()

        # Verify all attributes are reset
        assert mc_sim._coefficient is not None  # Should be default Coefficient()
        assert mc_sim._pi_expr is None
        assert mc_sim._sym_func is None
        assert mc_sim._exe_func is None
        assert len(mc_sim._variables) == 0
        assert len(mc_sim._symbols) == 0
        assert len(mc_sim._aliases) == 0
        assert len(mc_sim._latex_to_py) == 0
        assert len(mc_sim._py_to_latex) == 0
        assert len(mc_sim._var_symbols) == 0
        assert mc_sim._experiments == -1
        assert len(mc_sim._distributions) == 0
        assert len(mc_sim._dependencies) == 0
        assert len(mc_sim._simul_cache) == 0
        assert len(mc_sim._data) == 0

        # Use .size for NumPy arrays, not len()
        assert mc_sim._results.size == 0

        # Verify statistics are reset to NaN or None
        assert _is_none_or_nan(mc_sim._mean)
        assert _is_none_or_nan(mc_sim._median)
        assert _is_none_or_nan(mc_sim._dev)
        assert _is_none_or_nan(mc_sim._min)
        assert _is_none_or_nan(mc_sim._max)
        assert mc_sim._count == -1

    # ========================================================================
    # Serialization Tests
    # ========================================================================

    def test_to_dict_method(self) -> None:
        """*test_to_dict_method()* tests serialization to dictionary."""
        if "\\Pi_{1}" not in self.coefficients:
            pytest.skip("Pi_1 coefficient not found")

        coef = self.coefficients["\\Pi_{1}"]
        vars_in_coef = list(coef.var_dims.keys())

        mc_sim = MonteCarlo(
            _idx=3,
            _sym="MC_Pi_1",
            _fwk="CUSTOM",
            _name="Test MC Simulation",
            description="Test simulation for serialization",
            _coefficient=coef,
            _variables=self.variables,
            _experiments=N_EXP
        )

        # Set up and run simulation
        mc_sim._distributions = {
            k: v for k, v in self.dist_specs.items() if k in vars_in_coef
        }
        mc_sim.run()

        # Convert to dictionary
        sim_dict = mc_sim.to_dict()

        # Verify dictionary structure
        assert isinstance(sim_dict, dict)
        assert "idx" in sim_dict or "_idx" in sim_dict
        assert "sym" in sim_dict or "_sym" in sim_dict
        assert "fwk" in sim_dict or "_fwk" in sim_dict
        assert "name" in sim_dict or "_name" in sim_dict
        assert "description" in sim_dict
        assert "coefficient" in sim_dict
        assert "variables" in sim_dict
        assert "experiments" in sim_dict or "iterations" in sim_dict
        assert "results" in sim_dict

        # Verify coefficient is serialized
        coef_data = sim_dict.get("coefficient")
        assert coef_data is not None
        assert isinstance(coef_data, dict)

        # Verify variables are serialized
        vars_data = sim_dict.get("variables")
        assert vars_data is not None
        assert isinstance(vars_data, dict)

        # Verify results are converted to list
        results_data = sim_dict.get("results")
        assert results_data is not None
        assert isinstance(results_data, list)

    def test_from_dict_method(self) -> None:
        """*test_from_dict_method()* tests deserialization from dictionary."""
        if "\\Pi_{1}" not in self.coefficients:
            pytest.skip("Pi_1 coefficient not found")

        coef = self.coefficients["\\Pi_{1}"]
        vars_in_coef = list(coef.var_dims.keys())

        # Create original simulation
        mc_sim_original = MonteCarlo(
            _idx=7,
            _sym="MC_Original",
            _fwk="CUSTOM",
            _name="Original Simulation",
            description="Test for round-trip serialization",
            _coefficient=coef,
            _variables=self.variables,
            _experiments=N_EXP
        )

        # Set up and run simulation
        mc_sim_original._distributions = {
            k: v for k, v in self.dist_specs.items() if k in vars_in_coef
        }
        mc_sim_original.run()

        # Serialize
        sim_dict = mc_sim_original.to_dict()

        # Deserialize
        mc_sim_restored = MonteCarlo.from_dict(sim_dict)

        # Verify restoration
        assert mc_sim_restored is not None
        assert mc_sim_restored._idx == mc_sim_original._idx
        assert mc_sim_restored._sym == mc_sim_original._sym
        assert mc_sim_restored._fwk == mc_sim_original._fwk
        assert mc_sim_restored._name == mc_sim_original._name
        assert mc_sim_restored.description == mc_sim_original.description
        assert mc_sim_restored._experiments == mc_sim_original._experiments

        # Verify coefficient is restored
        assert mc_sim_restored._coefficient is not None
        assert mc_sim_restored._coefficient.sym == mc_sim_original._coefficient.sym

        # Verify variables are restored
        assert len(mc_sim_restored._variables) == len(mc_sim_original._variables)

        # Verify results are restored
        assert mc_sim_restored._results.size == mc_sim_original._results.size
        np.testing.assert_array_equal(mc_sim_restored._results, mc_sim_original._results)

    def test_serialization_round_trip(self) -> None:
        """*test_serialization_round_trip()* tests complete round-trip serialization."""
        if "\\Pi_{1}" not in self.coefficients:
            pytest.skip("Pi_1 coefficient not found")

        coef = self.coefficients["\\Pi_{1}"]
        vars_in_coef = list(coef.var_dims.keys())

        # Create and run original simulation
        mc_sim_original = MonteCarlo(
            _idx=9,
            _sym="MC_RoundTrip",
            _fwk="CUSTOM",
            _coefficient=coef,
            _variables=self.variables,
            _experiments=50
        )

        mc_sim_original._distributions = {
            k: v for k, v in self.dist_specs.items() if k in vars_in_coef
        }
        mc_sim_original.run()

        # Get original statistics
        original_stats = mc_sim_original.statistics

        # Round-trip: serialize and deserialize
        sim_dict = mc_sim_original.to_dict()
        mc_sim_restored = MonteCarlo.from_dict(sim_dict)

        # Get restored statistics
        restored_stats = mc_sim_restored.statistics

        # Compare statistics
        assert restored_stats["mean"] == original_stats["mean"]
        assert restored_stats["median"] == original_stats["median"]
        assert restored_stats["dev"] == original_stats["dev"]
        assert restored_stats["min"] == original_stats["min"]
        assert restored_stats["max"] == original_stats["max"]
        assert restored_stats["count"] == original_stats["count"]

    def test_from_dict_with_minimal_data(self) -> None:
        """*test_from_dict_with_minimal_data()* tests deserialization with minimal required data."""
        if "\\Pi_{1}" not in self.coefficients:
            pytest.skip("Pi_1 coefficient not found")

        coef = self.coefficients["\\Pi_{1}"]

        # Create minimal dictionary
        minimal_dict = {
            "coefficient": coef.to_dict(),
            "experiments": 10
        }

        # Should create a valid MonteCarlo instance
        mc_sim = MonteCarlo.from_dict(minimal_dict)

        assert mc_sim is not None
        assert mc_sim._coefficient is not None
        assert mc_sim._experiments == 10
