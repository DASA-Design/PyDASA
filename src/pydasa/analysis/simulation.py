# -*- coding: utf-8 -*-
"""
Module simulation.py
===========================================

Module for Monte Carlo Simulation execution and analysis in *PyDASA*.

This module provides the MonteCarloSim class for performing Monte Carlo simulations on dimensionless coefficients derived from dimensional analysis.

Classes:

    **MonteCarloSim**: Performs Monte Carlo simulations on dimensionless coefficients.

*IMPORTANT:* Based on the theory from:

    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Generic, Callable, Tuple
import numpy as np
from sympy import lambdify
from scipy import stats

# import matplotlib.pyplot as plt
# from sympy import lambdify

# Import validation base classes
from pydasa.core.basic import Validation

# Import related classes
from pydasa.buckingham.vashchy import Coefficient

# Import utils
from pydasa.utils.default import T
from pydasa.utils.latex import parse_latex, create_latex_mapping

# Import configuration
from pydasa.utils.latex import latex_to_python
# from pydasa.utils import config as cfg


@dataclass
class MonteCarloSim(Validation, Generic[T]):
    """**MonteCarloSim** class for stochastic analysis in *PyDASA*.

    Performs Monte Carlo simulations on dimensionless coefficients to analyze the coefficient's distribution and sensitivity to input parameter
    variations.

    Attributes:
        # Identification and Classification
        name (str): User-friendly name of the Monte Carlo simulation.
        description (str): Brief summary of the simulation.
        _idx (int): Index/precedence of the simulation.
        _sym (str): Symbol representation (LaTeX or alphanumeric).
        _alias (str): Python-compatible alias for use in code.
        _fwk (str): Framework context (PHYSICAL, COMPUTATION, SOFTWARE, CUSTOM).
        _cat (str): Category of analysis (SYM, NUM, HYB).

        # Expression Management
        _pi_expr (str): LaTeX expression to analyze.
        _sym_func (Callable): Sympy function of the simulation.
        _exe_func (Callable): Executable function for numerical evaluation.
        _variables (List[str]): Variable symbols in the expression.
        _latex_to_py (Dict[str, str]): Mapping from LaTeX to Python variable names.
        _py_to_latex (Dict[str, str]): Mapping from Python to LaTeX variable names.

        # Simulation Configuration
        _iterations (int): Number of simulation to run. Default is 1000.
        _distributions (Dict[str, Any]): Variable sampling distributions and specifications (specific name, parameters, and function).

        # Results
        inputs (np.ndarray): variable simulated inputs.
        _results (np.ndarray): Raw simulation results.
        summary (Dict[str, float]): Statistical summary of the simulation results.

        # Statistics
        _mean (float): Mean value of simulation results.
        _median (float): Median value of simulation results.
        _std_dev (float): Standard deviation of simulation results.
        _variance (float): Variance of simulation results.
        _min (float): Minimum value in simulation results.
        _max (float): Maximum value in simulation results.
        _count (int): Number of valid simulation results.
    """

    # Category attribute
    # :attr: _cat
    _cat: str = "NUM"
    """Category of sensitivity analysis (SYM, NUM)."""

    # Expression properties
    # :attr: _pi_expr
    _pi_expr: Optional[str] = None
    """LaTeX expression to analyze."""

    # :attr: _sym_func
    _sym_func: Optional[Callable] = None
    """Sympy function of the coefficient."""

    # :attr: _exe_func
    _exe_func: Optional[Callable] = None
    """Executable function for numerical evaluation."""

    # :attr: _variables
    _variables: Dict[str: Any] = field(default_factory=dict)
    """Variable symbols in the expression."""

    # :attr: _symbols
    _symbols: Dict[str: Any] = field(default_factory=dict)
    """Python symbols for the variables."""

    # :attr: _aliases
    _aliases: Dict[str: Any] = field(default_factory=dict)
    """Variable aliases for use in code."""

    # :attr: _latex_to_py
    _latex_to_py: Dict[str, str] = field(default_factory=dict)
    """Mapping from LaTeX symbols to Python-compatible names."""

    # :attr: _py_to_latex
    _py_to_latex: Dict[str, str] = field(default_factory=dict)
    """Mapping from Python-compatible names to LaTeX symbols."""

    # :attr: _coefficient
    _coefficient: Optional[Coefficient] = None
    """Coefficient for the simulation."""

    # Simulation configuration
    # :attr: _iterations
    _iterations: int = 1000
    """Number of simulation to run."""

    # :attr: _distributions
    _distributions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Variable sampling distributions and specifications that includes:
        - 'dtype': Name of the variable.
        - 'params': Distribution parameters (mean, std_dev, etc.).
        - 'func':  function for sampling, ussually in Lambda format.
    """

    # Results
    # :attr: inputs
    inputs: Optional[np.ndarray] = None
    """Sample value range for the simulation."""

    # :attr: _results
    _results: Optional[np.ndarray] = None
    """Raw simulation results."""

    # Individual statistics attributes
    # :attr: _mean
    _mean: float = -1.0
    """Mean value of simulation results."""

    # :attr: _median
    _median: float = -1.0
    """Median value of simulation results."""

    # :attr: _std_dev
    _std_dev: float = -1.0
    """Standard deviation of simulation results."""

    # :attr: _variance
    _variance: float = -1.0
    """Variance of simulation results."""

    # :attr: _min
    _min: float = -1.0
    """Minimum value in simulation results."""

    # :attr: _max
    _max: float = -1.0
    """Maximum value in simulation results."""

    # :attr: _count
    _count: int = -1
    """Number of valid simulation results."""

    # :attr: _statistics
    _statistics: Optional[Dict[str, float]] = None
    """Statistical summary of the Monte Carlo simulation results."""

    def __post_init__(self) -> None:
        """*__post_init__()* Initializes the Monte Carlo simulation."""
        # Initialize from base class
        super().__post_init__()

        # Set default symbol if not specified
        if not self._sym:
            self._sym = f"MC_\\Pi_{{{self._idx}}}" if self._idx >= 0 else "MC_\\Pi_{}"

        # Set default Python alias if not specified
        if not self._alias:
            self._alias = latex_to_python(self._sym)

        # Set name and description if not already set
        if not self.name:
            self.name = f"{self._sym} Monte Carlo"
        if not self.description:
            self.description = f"Monte Carlo simulation for {self._sym}"

        if self._pi_expr:
            # Parse the expression
            self._parse_expression(self._pi_expr)

        # TODO post init exectution to allocate memory is not working, fix it!
        # Preallocate full array space
        n_vars = len(self._variables)
        self.inputs = np.zeros((self._iterations, n_vars))
        self._results = np.zeros((self._iterations, 1))

    def _validate_simulation_ready(self) -> None:
        """*_validate_simulation_ready() * Checks if the simulation can be performed.

        Raises:
            ValueError: If the simulation is not ready due to missing variables, executable function, distributions, or invalid number of iterations.
        """
        if not self._variables:
            raise ValueError("No variables found in the expression.")
        if not self._sym_func:
            raise ValueError("No expression has been defined for analysis.")
        # if not self._exe_func:
        #     raise ValueError("No executable function defined for simulation.")
        if not self._distributions:
            _vars = self._variables
            missing = [v for v in _vars if v not in self._distributions]
            if missing:
                _msg = f"Missing distributions for variables: {missing}"
                raise ValueError(_msg)
        if self._iterations < -1:
            _msg = f"Invalid number of iterations: {self._iterations}"
            raise ValueError(_msg)

    def set_coefficient(self, coef: Coefficient) -> None:
        """*set_coefficient()* Configure analysis from a coefficient.

        Args:
            coef (Coefficient): Dimensionless coefficient to analyze.

        Raises:
            ValueError: If the coefficient doesn't have a valid expression.
        """
        if not coef.pi_expr:
            raise ValueError("Coefficient does not have a valid expression.")

        # save coefficient
        self._coefficient = coef

        # Set expression
        self._pi_expr = coef.pi_expr
        # parse coefficient expresion
        if coef._pi_expr:
            self._parse_expression(self._pi_expr)

        # Set name and description if not already set
        if not self.name:
            self.name = f"{coef.name} Monte Carlo"
        if not self.description:
            self.description = f"Monte Carlo simulation for {coef.name}"

    def _parse_expression(self, expr: str) -> None:
        """*_parse_expression()* Parse the LaTeX expression into a sympy function.

        Args:
            expr (str): LaTeX expression to parse.

        Raises:
            ValueError: If the expression cannot be parsed.
        """
        try:
            # Parse the expression
            self._sym_func = parse_latex(expr)

            # Create symbol mapping
            maps = create_latex_mapping(expr)
            self._symbols = maps[0]
            self._aliases = maps[1]
            self._latex_to_py = maps[2]
            self._py_to_latex = maps[3]

            # Substitute LaTeX symbols with Python symbols
            for latex_sym, py_sym in self._symbols.items():
                self._sym_func = self._sym_func.subs(latex_sym, py_sym)

            # Get Python variable names
            self._variables = [str(s) for s in self._sym_func.free_symbols]
            self._variables = sorted(self._variables)

        except Exception as e:
            _msg = f"Failed to parse expression: {str(e)}"
            raise ValueError(_msg)

    def set_distribution(self,
                         var: str,
                         dist: Callable,
                         specs: Dict = None) -> None:
        # TODO old code, probably dont need it, delete later!
        """*set_distribution()* Set the sampling distribution for a variable in the coefficient.
        Args:
            var (str): Variable name to set the distribution.
            dist (Callable): Callable function that samples from the distribution.

        Raises:
            ValueError: If the variable is not found in the expression or if the distribution is not callable.
            ValueError: If specs are provided but not a dictionary.
        """

        if var not in self._symbols:
            _msg = f"Variable '{var}' not found in expression. "
            _msg += f"Available variables: {[self._py_to_latex.get(v, v) for v in self._symbols]}"
            raise ValueError(_msg)

        if not callable(dist):
            _msg = f"Distribution must be callable. Got: {type(dist)}"
            raise ValueError(_msg)

        self._distributions[var] = dist

        if specs:
            if not isinstance(specs, dict):
                _msg = f"Specs must be a dictionary. Got: {type(specs)}"
                raise ValueError(_msg)
            self.specs[var] = specs

    def run(self, iters: int = None) -> None:
        """*run()* Perform the Monte Carlo simulation.

        Raises:
            ValueError: If the simulation is not ready due to missing variables, executable function, distributions, or invalid number of iterations.
            ValueError: If there are numerical errors during simulation runs (e.g., division by zero).
        """
        # Validate simulation readiness
        self._validate_simulation_ready()

        # Set iterations if necesary
        if iters is not None:
            self._iterations = iters

        # Clear previous results
        self._results = np.zeros((0,))
        self._reset_statistics()

        # Preallocate full array space
        n_vars = len(self._variables)
        self.inputs = np.zeros((self._iterations, n_vars))
        self._results = np.zeros((self._iterations, 1))  # 2D array for results

        # Run simulation
        i = 0
        while i < self._iterations:
            # print(f"Running simulation {i+1}/{self._iterations}")
            try:
                # Sample values from distributions
                samples = {}
                for var in self._latex_to_py.keys():
                    # if the distribution exists
                    if var in self._distributions:
                        # accessing the distribution in the distribution specs
                        samples[var] = self._distributions[var]["func"]()
                    # otherwise, raise error
                    else:
                        _msg = f"Missing distribution for variable: {var}"
                        raise ValueError(_msg)

                # Prepare sorted/ordered values for function evaluation
                sorted_vals = [samples[var] for var in self._latex_to_py.keys()]

                # Create lambdify function using Python symbols
                aliases = [self._aliases[v] for v in self._variables]
                self._exe_func = lambdify(aliases, self._sym_func, "numpy")

                # Evaluate the coefficient
                result = float(self._exe_func(*sorted_vals))
                # save simulation inputs and results
                self.inputs[i, :] = sorted_vals
                self._results[i] = result
                # self._results.append(result)

            except Exception as e:
                # Handle numerical errors (e.g., division by zero)
                _msg = f"Error during simulation run {i}: {str(e)}"
                # TODO add logger later
                # print(_msg)
                raise ValueError(_msg)
                # continue
            i += 1

        # Calculate statistics
        if len(self._results) == self._iterations:
            self._calculate_statistics()
        else:
            raise ValueError("Invalid results, check yout distributions!")

    def _reset_statistics(self) -> None:
        """*_reset_statistics()* Reset all statistical attributes to default values."""
        self._mean = -1.0
        self._median = -1.0
        self._std_dev = -1.0
        self._variance = -1.0
        self._min = -1.0
        self._max = -1.0
        self._count = 0

    def _calculate_statistics(self) -> None:
        """*_calculate_statistics()* Calculate statistical properties of simulation results."""
        # results = np.array(self._results)

        # Calculate and store each statistic separately
        self._mean = float(np.mean(self._results))
        self._median = float(np.median(self._results))
        self._std_dev = float(np.std(self._results))
        self._variance = float(np.var(self._results))
        self._min = float(np.min(self._results))
        self._max = float(np.max(self._results))
        self._count = len(self._results)

    def get_confidence_interval(self,
                                conf: float = 0.95) -> Tuple[float, float]:
        """*get_confidence_interval()* Calculate the confidence interval for the simulation results.

        Args:
            conf (float, optional): Confidence level for the interval. Defaults to 0.95.

        Raises:
            ValueError: If no results are available or if the confidence level is not between 0 and 1.

        Returns:
            Tuple[float, float]: Lower and upper bounds of the confidence interval.
        """
        if self._results is None:
            _msg = "No results available. Run the simulation first."
            raise ValueError(_msg)

        if not 0 < conf < 1:
            _msg = f"Confidence must be between 0 and 1. Got: {conf}"
            raise ValueError(_msg)

        # Calculate the margin of error using the t-distribution
        alpha = stats.t.ppf((1 + conf) / 2, self._count - 1)
        margin = alpha * self._std_dev / np.sqrt(self._count)
        ans = (self._mean - margin, self._mean + margin)
        return ans

    def extract_results(self) -> Dict[str, Any]:
        """*extract_results()* Extract simulation results.

        Returns:
            Dict[str, Any]: Dictionary containing simulation results.
        """
        export = {}

        # Extract all values for each variable (column)
        for i, var in enumerate(self._py_to_latex.values()):
            # Get the entire column for this variable (all simulation runs)
            column = self.inputs[:, i]

            # Use a meaningful key that includes variable name and coefficient
            # key = f"{var}@{self._coefficient.sym}"
            key = f"{var}"
            export[key] = column

        # Add the coefficient results
        export[self._coefficient.sym] = self._results.flatten()
        return export

    # Properties for accessing results and statistics

    @property
    def results(self) -> List[float]:
        """*results* Raw simulation results.

        Returns:
            List[float]: Copy of the simulation results.

        Raises:
            ValueError: If no results are available.
        """
        if self._results is None:
            raise ValueError("No results available. Run the simulation first.")
        return self._results.copy()

    @property
    def statistics(self) -> Dict[str, float]:
        """*statistics* Get the statistical analysis of simulation results.
        Raises:
            ValueError: If no results are available.

        Returns:
            Dict[str, float]: Dictionary containing statistical properties:
                - "mean"": Mean value of results
                - "median": Median value of results
                - "std_dev": Standard deviation of results
                - "variance": Variance of results
                - "min": Minimum value in results
                - "max": Maximum value in results
                - "count": Number of valid results
        """
        if self._results is None:
            _msg = "No statistics available. Run the simulation first."
            raise ValueError(_msg)

        # Build statistics dictionary from individual attributes
        self._statistics = {
            "mean": self._mean,
            "median": self._median,
            "std_dev": self._std_dev,
            "variance": self._variance,
            "min": self._min,
            "max": self._max,
            "count": self._count
        }
        return self._statistics

    @property
    def iterations(self) -> int:
        """*iterations* Number of simulation iterations.

        Returns:
            int: Current number of iterations.
        """
        return self._iterations

    @iterations.setter
    def iterations(self, val: int) -> None:
        """*iterations* Set the number of simulation runs.

        Args:
            val (int): Number of iterations to run the simulation.

        Raises:
            ValueError: If the number of iterations is not positive.
        """
        if val < 0:
            _msg = f"Number of iterations must be positive. Got: {val}"
            raise ValueError(_msg)
        self._iterations = val

    @property
    def distributions(self) -> Dict[str, Any]:
        """*distributions* Get the variable distributions.

        Returns:
            Dict[str, Any]: Current variable distributions.
        """
        return self._distributions.copy()

    @distributions.setter
    def distributions(self, val: Dict[str, Any]) -> None:
        """*distributions* Set the variable distributions.

        Args:
            val (Dict[str, Any]): New variable distributions.

        Raises:
            ValueError: If the distributions are invalid.
        """
        if not all(callable(v["func"]) for v in val.values()):
            _msg = "All distributions must have callable 'func' functions."
            inv = [k for k, v in val.items() if not callable(v["func"])]
            _msg += f" Invalid entries: {inv}"
            raise ValueError(_msg)
        self._distributions = val

    # Additional properties for statistics

    @property
    def mean(self) -> float:
        """*mean* Mean value of simulation results.

        Returns:
            float: Mean value.

        Raises:
            ValueError: If no results are available.
        """
        if self._results is None:
            _msg = "No statistics available. Run the simulation first."
            raise ValueError(_msg)
        return self._mean

    @property
    def median(self) -> float:
        """*median* Median value of simulation results.

        Returns:
            float: Median value.

        Raises:
            ValueError: If no results are available.
        """
        if self._results is None:
            _msg = "No statistics available. Run the simulation first."
            raise ValueError(_msg)
        return self._median

    @property
    def std_dev(self) -> float:
        """*std_dev* Standard deviation of simulation results.

        Returns:
            float: Standard deviation.

        Raises:
            ValueError: If no results are available.
        """
        if self._results is None:
            _msg = "No statistics available. Run the simulation first."
            raise ValueError(_msg)
        return self._std_dev

    @property
    def variance(self) -> float:
        """*variance* Variance of simulation results.

        Returns:
            float: Variance value.

        Raises:
            ValueError: If no results are available.
        """
        if self._results is None:
            _msg = "No statistics available. Run the simulation first."
            raise ValueError(_msg)
        return self._variance

    @property
    def min_value(self) -> float:
        """*min_value* Minimum value in simulation results.

        Returns:
            float: Minimum value.

        Raises:
            ValueError: If no results are available.
        """
        if self._results is None:
            _msg = "No statistics available. Run the simulation first."
            raise ValueError(_msg)
        return self._min

    @property
    def max_value(self) -> float:
        """*max_value* Maximum value in simulation results.

        Returns:
            float: Maximum value.

        Raises:
            ValueError: If no results are available.
        """
        if self._results is None:
            _msg = "No statistics available. Run the simulation first."
            raise ValueError(_msg)
        return self._max

    @property
    def count(self) -> int:
        """*count* Number of valid simulation results.

        Returns:
            int: Result count.

        Raises:
            ValueError: If no results are available.
        """
        if self._results is None:
            _msg = "No statistics available. Run the simulation first."
            raise ValueError(_msg)
        return self._count

    @property
    def summary(self) -> Dict[str, float]:
        """*summary()* Get the statistical analysis of simulation results.

        Raises:
            ValueError: If no results are available.

        Returns:
            Dict[str, float]: Dictionary containing statistical properties:
                - "inputs" array with the simulation inputs.
                - "results" array with the simulation results.
                - "mean": Mean value of results.
                - "median": Median value of results.
                - "std_dev": Standard deviation of results.
                - "variance": Variance of results.
                - "min": Minimum value in results.
                - "max": Maximum value in results.
                - "count": Number of valid results.
        """
        if self._results is None:
            _msg = "No statistics available. Run the simulation first."
            raise ValueError(_msg)

        # Build summary dictionary from individual attributes
        self._summary = {
            # "inputs": self.inputs,
            # "results": self._results,
            "mean": self._mean,
            "median": self._median,
            "std_dev": self._std_dev,
            "variance": self._variance,
            "min": self._min,
            "max": self._max,
            "count": self._count
        }
        return self._summary

    def clear(self) -> None:
        """*clear()* Reset all attributes to default values."""
        # Reset base class attributes
        self._idx = -1
        self._sym = "MC_\\Pi_{}"
        self._alias = ""
        self._fwk = "PHYSICAL"
        self.name = ""
        self.description = ""

        # Reset simulation attributes
        self._pi_expr = None
        self._sym_func = None
        self._exe_func = None
        self._variables = []
        self._latex_to_py = {}
        self._py_to_latex = {}
        self._iterations = 1000
        self._distributions = {}
        self.inputs = np.zeros((0,))
        self._results = np.zeros((0,))

        # Reset statistics
        self._reset_statistics()

    def to_dict(self) -> Dict[str, Any]:
        """*to_dict()* Convert simulation to dictionary representation."""
        return {
            # Base class attributes
            "idx": self._idx,
            "sym": self._sym,
            "alias": self._alias,
            "fwk": self._fwk,
            "name": self.name,
            "description": self.description,
            # Simulation attributes
            "pi_expr": self._pi_expr,
            # "sym_func": str(self._sym_func) if self._sym_func else None,
            # "exe_func": str(self._exe_func) if self._exe_func else None,
            "variables": self._variables,
            "iterations": self._iterations,
            # "distributions": {k: str(v) for k, v in self._distributions.items()},
            # Results
            "mean": self._mean,
            "median": self._median,
            "std_dev": self._std_dev,
            "variance": self._variance,
            "min": self._min,
            "max": self._max,
            "count": self._count,
            "inputs": self.inputs,
            "results": self._results
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MonteCarloSim:
        """*from_dict()* Create simulation from dictionary representation."""
        # Create basic instance
        instance = cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            _idx=data.get("idx", -1),
            _sym=data.get("sym", ""),
            _fwk=data.get("fwk", "PHYSICAL"),
            _alias=data.get("alias", ""),
            _pi_expr=data.get("pi_expr", None),
            _iterations=data.get("iterations", 1000),
        )

        # Optionally set statistics if available
        if "statistics" in data:
            stats = data["statistics"]
            instance._mean = stats.get("mean", -1.0)
            instance._median = stats.get("median", -1.0)
            instance._std_dev = stats.get("std_dev", -1.0)
            instance._variance = stats.get("variance", -1.0)
            instance._min = stats.get("min", -1.0)
            instance._max = stats.get("max", -1.0)
            instance._count = stats.get("count", -1)

        return instance

    # def plot_histogram(self, bins: int = 30, figsize: Tuple[int, int] = (10, 6),
    # FIXME old code, delete when ready!!!
    #                   title: str = None, save_path: str = None) -> None:
    #     """*plot_histogram()* Plot a histogram of simulation results."""
    #     if not self._results:
    #         raise ValueError("No simulation results to plot. Run the simulation first.")

    #     plt.figure(figsize=figsize)
    #     plt.hist(self._results, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')

    #     # Add statistics to the plot
    #     stats_text = (
    #         f"Mean: {self._mean:.4f}\n"
    #         f"Std Dev: {self._std_dev:.4f}\n"
    #         f"Min: {self._min:.4f}\n"
    #         f"Max: {self._max:.4f}\n"
    #         f"Samples: {self._count}"
    #     )
    #     plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction',
    #                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    #                 va='top', fontsize=10)

    #     # Add title and labels
    #     plot_title = title if title else f"Monte Carlo Simulation: {self._pi_expr}"
    #     plt.title(plot_title)
    #     plt.xlabel("Coefficient Value")
    #     plt.ylabel("Frequency")
    #     plt.grid(True, alpha=0.3)

    #     if save_path:
    #         plt.savefig(save_path, dpi=300, bbox_inches='tight')

    #     plt.show()
