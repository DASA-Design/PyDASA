# -*- coding: utf-8 -*-
"""
Module test_simulation.py
===========================================

Unit tests for the MonteCarloSim class in PyDASA.
"""

# import testing package
import unittest
import pytest
import numpy as np

# import the module to test
from pydasa.analysis.simulation import MonteCarloSim
from pydasa.buckingham.vashchy import Coefficient
from pydasa.core.parameter import Variable

# import the data to test
from tests.pydasa.data.test_data import get_simulation_test_data

# asserting module imports
assert MonteCarloSim
assert Coefficient
assert Variable
assert get_simulation_test_data


class TestMonteCarloSim(unittest.TestCase):
    """**TestMonteCarloSim** implements unit tests for the MonteCarloSim class.

    Args:
        unittest (TestCase): unittest.TestCase class for unit tests in Python.
    """

    @pytest.fixture(autouse=True)
    def inject_fixtures(self) -> None:
        """*inject_fixtures()* injects global test parameters as a fixture."""
        self.test_data = get_simulation_test_data()

        # Create test variables from the test data
        self.test_variables = {}
        for key, var_data in self.test_data["TEST_VARIABLES"].items():
            var = Variable(**var_data)
            self.test_variables[key] = var

        # Create simple test variables
        self.simple_variables = {}
        for key, var_data in self.test_data["SIMPLE_VARIABLES"].items():
            var = Variable(**var_data)
            self.simple_variables[key] = var

        # Create Reynolds-specific variables
        self.reynolds_variables = {}
        for key, var_data in self.test_data["REYNOLDS_VARIABLES"].items():
            var = Variable(**var_data)
            self.reynolds_variables[key] = var

    def test_default_simulation(self) -> None:
        """*test_default_simulation()* tests creating a simulation with default values."""
        sim = MonteCarloSim()

        # Test initialization
        assert sim is not None
        assert isinstance(sim, MonteCarloSim)

        # Test default values
        assert sim._iterations == 1000
        assert sim._cat == "NUM"
        assert sim._pi_expr is None
        assert sim._sym_func is None
        assert sim._exe_func is None
        assert len(sim._variables) == 0
        assert len(sim._distributions) == 0

        # Test statistics defaults
        assert np.isnan(sim._mean)
        assert np.isnan(sim._median)
        assert np.isnan(sim._std_dev)
        assert sim._count == 0

    def test_custom_initialization(self) -> None:
        """*test_custom_initialization()* tests creating a simulation with custom parameters."""
        sim = MonteCarloSim(
            name="Test Simulation",
            description="Test description",
            _idx=0,
            _sym="MC_\\Pi_{0}",
            _fwk="PHYSICAL",
            _iterations=500
        )

        assert sim.name == "Test Simulation"
        assert sim.description == "Test description"
        assert sim._idx == 0
        assert sim._sym == "MC_\\Pi_{0}"
        assert sim._fwk == "PHYSICAL"
        assert sim._iterations == 500
        assert sim._alias == "MC_Pi_0"

    def test_parse_simple_expression(self) -> None:
        """*test_parse_simple_expression()* tests parsing a simple LaTeX expression."""
        expr = self.test_data["SIMPLE_EXPR"]
        sim = MonteCarloSim(_pi_expr=expr)

        # Test expression was parsed
        assert sim._sym_func is not None
        assert len(sim._symbols) > 0
        assert len(sim._latex_to_py) > 0
        assert len(sim._py_to_latex) > 0

        # Test variable symbols extracted
        var_count = len([k for k in sim._symbols.keys()])
        assert var_count >= 2

    def test_parse_physics_expression(self) -> None:
        """*test_parse_physics_expression()* tests parsing a physics LaTeX expression."""
        expr = self.test_data["PHYSICS_EXPR"]
        sim = MonteCarloSim(_pi_expr=expr)

        assert sim._sym_func is not None
        assert len(sim._symbols) >= 3

    def test_parse_reynolds_expression(self) -> None:
        """*test_parse_reynolds_expression()* tests parsing the Reynolds number expression."""
        expr = self.test_data["REYNOLDS_EXPR"]
        sim = MonteCarloSim(_pi_expr=expr)
        
        assert sim._sym_func is not None
        # Should have rho, v, L, mu
        assert len(sim._symbols) >= 4

    def test_parse_complex_expression(self) -> None:
        """*test_parse_complex_expression()* tests parsing a complex expression."""
        expr = self.test_data["COMPLEX_EXPR"]
        sim = MonteCarloSim(_pi_expr=expr)

        assert sim._sym_func is not None
        assert len(sim._symbols) >= 3

    def test_invalid_expression(self) -> None:
        """*test_invalid_expression()* tests handling invalid expressions."""
        with pytest.raises(ValueError) as excinfo:
            MonteCarloSim(_pi_expr="invalid$$expression@@##")
        assert "Failed to parse" in str(excinfo.value) or "expression" in str(excinfo.value).lower()

    def test_set_reynolds_coefficient(self) -> None:
        """*test_set_reynolds_coefficient()* tests setting the Reynolds coefficient for analysis."""
        # Create Reynolds coefficient
        reynolds_data = self.test_data["REYNOLDS_COEFFICIENT"]
        coef = Coefficient(
            _idx=reynolds_data["_idx"],
            _sym=reynolds_data["_sym"],
            _alias=reynolds_data["_alias"],
            _fwk=reynolds_data["_fwk"],
            _cat=reynolds_data["_cat"],
            _pi_expr=self.test_data["REYNOLDS_EXPR"],
            name=reynolds_data["name"],
            description=reynolds_data["description"]
        )

        # Add variables to coefficient
        vars_lt = {}
        for key, val in self.reynolds_variables.items():
            if key == val._sym:
                vars_lt[key] = self.reynolds_variables[key]
            # vars_lt.append(self.reynolds_variables[var_key])
        coef.variables = vars_lt

        sim = MonteCarloSim()
        sim.set_coefficient(coef)
        assert sim._coefficient is coef
        assert sim._pi_expr == coef.pi_expr
        assert sim._sym_func is not None
        assert reynolds_data["name"] in sim.name
        assert reynolds_data["name"] in sim.description

    def test_set_simple_coefficient(self) -> None:
        """*test_set_simple_coefficient()* tests setting a simple coefficient."""
        simple_data = self.test_data["SIMPLE_COEFFICIENT"]
        coef = Coefficient(
            _idx=simple_data["_idx"],
            _sym=simple_data["_sym"],
            _pi_expr="\\frac{v}{L}",
            name=simple_data["name"]
        )

        sim = MonteCarloSim()
        sim.set_coefficient(coef)

        assert sim._coefficient is coef
        assert sim._pi_expr is not None

    def test_set_coefficient_without_expression(self) -> None:
        """*test_set_coefficient_without_expression()* tests error when coefficient has no expression."""
        coef = Coefficient(_idx=0, _sym="\\Pi_{0}")
        sim = MonteCarloSim()

        with pytest.raises(ValueError) as excinfo:
            sim.set_coefficient(coef)
        assert "valid expression" in str(excinfo.value)

    def test_validate_readiness_no_variables(self) -> None:
        """*test_validate_readiness_no_variables()* tests validation fails without variables."""
        sim = MonteCarloSim(_pi_expr=self.test_data["SIMPLE_EXPR"])

        with pytest.raises(ValueError) as excinfo:
            sim._validate_readiness()
        assert "variable" in str(excinfo.value).lower()

    def test_validate_readiness_no_expression(self) -> None:
        """*test_validate_readiness_no_expression()* tests validation fails without expression."""
        sim = MonteCarloSim()
        sim._variables = self.simple_variables

        with pytest.raises(ValueError) as excinfo:
            sim._validate_readiness()
        assert "expression" in str(excinfo.value).lower()

    def test_validate_readiness_missing_distributions(self) -> None:
        """*test_validate_readiness_missing_distributions()* tests validation fails with missing distributions."""
        sim = MonteCarloSim(_pi_expr=self.test_data["SIMPLE_EXPR"])
        sim._variables = self.simple_variables
        sim._symbols = {"alpha": None, "beta": None}

        with pytest.raises(ValueError) as excinfo:
            sim._validate_readiness()
        assert "distribution" in str(excinfo.value).lower()

    def test_iterations_property(self) -> None:
        """*test_iterations_property()* tests the iterations property."""
        sim = MonteCarloSim()

        # Test getter
        assert sim.iterations == 1000

        # Test setter with valid values
        for val in self.test_data["VALID_ITERATIONS"]:
            sim.iterations = val
            assert sim.iterations == val

    def test_invalid_iterations(self) -> None:
        """*test_invalid_iterations()* tests setting invalid iteration counts."""
        sim = MonteCarloSim()

        for val in self.test_data["INVALID_ITERATIONS"]:
            with pytest.raises(ValueError) as excinfo:
                sim.iterations = val
            assert "positive" in str(excinfo.value).lower()

    def test_generate_sample_independent(self) -> None:
        """*test_generate_sample_independent()* tests sample generation for independent variables."""
        var = self.simple_variables["\\alpha"]
        sim = MonteCarloSim()
        memory = {}

        sample = sim._generate_sample(var, memory)

        assert sample is not None
        assert isinstance(sample, (int, float))
        assert not np.isnan(sample)

    def test_reset_memory(self) -> None:
        """*test_reset_memory()* tests resetting simulation memory."""
        sim = MonteCarloSim(_pi_expr=self.test_data["SIMPLE_EXPR"])
        sim._variables = self.simple_variables
        sim._simul_cache = {"alpha": [1.0, 2.0], "beta": [3.0, 4.0]}

        # Set some data
        sim.inputs = np.array([[1, 2], [3, 4]])
        sim._results = np.array([5, 6])
        sim._mean = 5.5

        # Reset
        sim._reset_memory()

        assert len(sim.inputs) == 0
        assert len(sim._results) == 0
        assert sim._mean == -1.0
        # Check cache was cleared
        for key in sim._simul_cache:
            assert len(sim._simul_cache[key]) == 0

    def test_reset_statistics(self) -> None:
        """*test_reset_statistics()* tests resetting statistical attributes."""
        sim = MonteCarloSim()

        # Set some statistics
        sim._mean = 10.0
        sim._median = 9.5
        sim._std_dev = 2.0
        sim._count = 100

        # Reset
        sim._reset_statistics()

        assert sim._mean == -1.0
        assert sim._median == -1.0
        assert sim._std_dev == -1.0
        assert sim._variance == -1.0
        assert sim._min == -1.0
        assert sim._max == -1.0
        assert sim._count == 0

    def test_calculate_statistics(self) -> None:
        """*test_calculate_statistics()* tests statistics calculation."""
        sim = MonteCarloSim()
        test_results = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim._results = test_results

        sim._calculate_statistics()

        assert sim._mean == pytest.approx(3.0)
        assert sim._median == pytest.approx(3.0)
        assert sim._std_dev == pytest.approx(np.std(test_results))
        assert sim._variance == pytest.approx(np.var(test_results))
        assert sim._min == pytest.approx(1.0)
        assert sim._max == pytest.approx(5.0)
        assert sim._count == 5

    def test_statistics_property(self) -> None:
        """*test_statistics_property()* tests the statistics property."""
        sim = MonteCarloSim()
        sim._results = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim._calculate_statistics()

        stats = sim.statistics

        assert isinstance(stats, dict)
        for stat in self.test_data["EXPECTED_STATISTICS"]:
            assert stat in stats
        assert stats["mean"] == pytest.approx(3.0)
        assert stats["count"] == 5

    def test_statistics_property_no_results(self) -> None:
        """*test_statistics_property_no_results()* tests accessing statistics without results."""
        sim = MonteCarloSim()

        with pytest.raises(ValueError) as excinfo:
            _ = sim.statistics
        assert "run" in str(excinfo.value).lower() or "result" in str(excinfo.value).lower()

    def test_get_confidence_interval(self) -> None:
        """*test_get_confidence_interval()* tests confidence interval calculation."""
        sim = MonteCarloSim()
        sim._results = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim._calculate_statistics()

        for conf in self.test_data["CONFIDENCE_LEVELS"]:
            lower, upper = sim.get_confidence_interval(conf)
            assert isinstance(lower, float)
            assert isinstance(upper, float)
            assert lower < upper
            assert lower <= sim._mean <= upper

    def test_get_confidence_interval_invalid(self) -> None:
        """*test_get_confidence_interval_invalid()* tests invalid confidence levels."""
        sim = MonteCarloSim()
        sim._results = np.array([1.0, 2.0, 3.0])
        sim._calculate_statistics()

        for conf in self.test_data["INVALID_CONFIDENCE"]:
            with pytest.raises(ValueError) as excinfo:
                sim.get_confidence_interval(conf)
            assert "0 and 1" in str(excinfo.value) or "between" in str(excinfo.value)

    def test_get_confidence_interval_no_results(self) -> None:
        """*test_get_confidence_interval_no_results()* tests confidence interval without results."""
        sim = MonteCarloSim()

        with pytest.raises(ValueError) as excinfo:
            sim.get_confidence_interval()
        assert "result" in str(excinfo.value).lower()

    def test_results_property(self) -> None:
        """*test_results_property()* tests the results property."""
        sim = MonteCarloSim()
        test_results = np.array([1.0, 2.0, 3.0])
        sim._results = test_results

        results = sim.results

        assert isinstance(results, np.ndarray)
        assert len(results) == 3
        assert np.array_equal(results, test_results)

    def test_results_property_no_results(self) -> None:
        """*test_results_property_no_results()* tests accessing results without running simulation."""
        sim = MonteCarloSim()

        with pytest.raises(ValueError) as excinfo:
            _ = sim.results
        assert "result" in str(excinfo.value).lower()

    def test_individual_statistic_properties(self) -> None:
        """*test_individual_statistic_properties()* tests individual statistic properties."""
        sim = MonteCarloSim()
        sim._results = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim._calculate_statistics()

        assert sim.mean == pytest.approx(3.0)
        assert sim.median == pytest.approx(3.0)
        assert sim.min_value == pytest.approx(1.0)
        assert sim.max_value == pytest.approx(5.0)
        assert sim.count == 5
        assert isinstance(sim.std_dev, float)
        assert isinstance(sim.variance, float)

    def test_clear(self) -> None:
        """*test_clear()* tests clearing simulation data."""
        sim = MonteCarloSim(
            name="Test",
            description="Description",
            _idx=5,
            _sym="MC_\\Pi_{5}",
            _pi_expr=self.test_data["SIMPLE_EXPR"],
            _iterations=500
        )
        sim._results = np.array([1.0, 2.0, 3.0])

        sim.clear()

        assert sim._idx == -1
        assert sim.name == ""
        assert sim.description == ""
        assert sim._pi_expr is None
        assert sim._iterations == 1000
        assert len(sim.inputs) == 0
        assert len(sim._results) == 0
        assert sim._mean == -1.0

    def test_to_dict(self) -> None:
        """*test_to_dict()* tests dictionary serialization."""
        sim = MonteCarloSim(
            name="Test Simulation",
            description="Test description",
            _idx=0,
            _sym="MC_\\Pi_{0}",
            _iterations=500
        )
        sim._results = np.array([1.0, 2.0, 3.0])
        sim._calculate_statistics()

        data = sim.to_dict()

        assert isinstance(data, dict)
        assert data["name"] == "Test Simulation"
        assert data["description"] == "Test description"
        assert data["idx"] == 0
        assert data["sym"] == "MC_\\Pi_{0}"
        assert data["iterations"] == 500
        assert "mean" in data
        assert "median" in data
        assert "count" in data

    def test_extract_results(self) -> None:
        """*test_extract_results()* tests extracting simulation results."""
        # Create simple coefficient
        coef = Coefficient(
            _idx=0,
            _sym="\\Pi_{0}",
            _pi_expr=self.test_data["SIMPLE_EXPR"]
        )

        sim = MonteCarloSim()
        sim.set_coefficient(coef)
        sim._py_to_latex = {"alpha": "\\alpha", "beta": "\\beta"}
        sim.inputs = np.array([[1.0, 2.0], [3.0, 4.0]])
        sim._results = np.array([[5.0], [6.0]])

        results = sim.extract_results()

        assert isinstance(results, dict)
        assert len(results) > 0
        # Should have coefficient results
        assert "\\Pi_{0}" in results or coef._sym in results
        # Should have variable results
        assert len(results["\\Pi_{0}"]) == 2 or len(results[coef._sym]) == 2

    def test_variables_property(self) -> None:
        """*test_variables_property()* tests the variables property."""
        sim = MonteCarloSim()
        sim._variables = self.simple_variables

        variables = sim.variables

        assert isinstance(variables, dict)
        assert len(variables) == len(self.simple_variables)

    def test_coefficient_property(self) -> None:
        """*test_coefficient_property()* tests the coefficient property."""
        coef = Coefficient(_idx=0, _sym="\\Pi_{0}", pi_expr=self.test_data["SIMPLE_EXPR"])
        sim = MonteCarloSim()
        sim.set_coefficient(coef)

        assert sim.coefficient is coef

    def test_distributions_property(self) -> None:
        """*test_distributions_property()* tests the distributions property."""
        sim = MonteCarloSim()
        test_dists = {"var1": {"dtype": "uniform", "params": {}}}
        sim._distributions = test_dists

        dists = sim.distributions

        assert isinstance(dists, dict)
        assert len(dists) == 1
