# -*- coding: utf-8 -*-
"""
Module simulation.py
===========================================

Module for Monte Carlo Simulation execution and analysis in *PyDASA*.

This module provides the MonteCarlo class for performing Monte Carlo simulations on dimensionless coefficients derived from dimensional analysis.

Classes:

    **MonteCarlo**: Performs Monte Carlo simulations on dimensionless coefficients.

*IMPORTANT:* Based on the theory from:

    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple, Union

# python third-party modules
import numpy as np
from numpy.typing import NDArray
from sympy import lambdify
# from sympy import Expr, Symbol
from scipy import stats
import sympy as sp

# Import validation base classes
from pydasa.core.basic import Foundation

# Import related classes
from pydasa.dimensional.buckingham import Coefficient
from pydasa.elements.parameter import Variable

# Import utils
from pydasa.serialization.parser import parse_latex, create_latex_mapping

# Import configuration
from pydasa.serialization.parser import latex_to_python

# # Type aliases
# SymbolDict = Dict[str, sp.Symbol]
# # FIX: Allow Basic or Expr since subs() returns Basic
# SymExpr = Union[sp.Expr, sp.Basic]


@dataclass
class MonteCarlo(Foundation):
    """**MonteCarlo** class for stochastic analysis in *PyDASA*.

    Performs Monte Carlo simulations on dimensionless coefficients to analyze the coefficient's distribution and sensitivity to input parameter
    variations.

    Args:
        Foundation: Foundation class for validation of symbols and frameworks.

    Attributes:
        # Core Identification
        name (str): User-friendly name of the Monte Carlo simulation.
        description (str): Brief summary of the simulation.
        _idx (int): Index/precedence of the simulation.
        _sym (str): Symbol representation (LaTeX or alphanumeric).
        _alias (str): Python-compatible alias for use in code.
        _fwk (str): Framework context (PHYSICAL, COMPUTATION, SOFTWARE, CUSTOM).
        _cat (str): Category of analysis (SYM, NUM, HYB).

        # Coefficient and Expression Management
        _coefficient (Optional[Coefficient]): Coefficient for the simulation.
        _pi_expr (str): LaTeX expression to analyze.
        _sym_func (Callable): Sympy function of the simulation.
        _exe_func (Callable): Executable function for numerical evaluation.

        # Variable Management
        _variables (Dict[str, Variable]): Variable symbols in the expression.
        _symbols (Dict[str, Any]): Python symbols for the variables.
        _aliases (Dict[str, Any]): Variable aliases for use in code.
        _latex_to_py (Dict[str, str]): Mapping from LaTeX to Python variable names.
        _py_to_latex (Dict[str, str]): Mapping from Python to LaTeX variable names.

        # Simulation Configuration
        _experiments (int): Number of simulation experiments to run. Default is -1.
        _distributions (Dict[str, Dict[str, Any]]): Variable sampling distributions.
        _simul_cache (Dict[str, NDArray[np.float64]]): Working sampled values cache.

        # Results and Inputs
        inputs (Optional[np.ndarray]): Variable simulated inputs.
        _results (Optional[np.ndarray]): Raw simulation results.

        # Statistics
        _mean (float): Mean value of simulation results.
        _median (float): Median value of simulation results.
        _std_dev (float): Standard deviation of simulation results.
        _variance (float): Variance of simulation results.
        _min (float): Minimum value in simulation results.
        _max (float): Maximum value in simulation results.
        _count (int): Number of valid simulation results.
        _statistics (Optional[Dict[str, float]]): Statistical summary.
    """

    # ========================================================================
    # Core Identification
    # ========================================================================

    # :attr: name
    _name: str = ""
    """User-friendly name of the Monte Carlo simulation."""

    # :attr: description
    description: str = ""
    """Brief summary of the simulation."""

    # :attr: _idx
    _idx: int = -1
    """Index/precedence of the simulation."""

    # :attr: _sym
    _sym: str = ""
    """Symbol representation (LaTeX or alphanumeric)."""

    # :attr: _alias
    _alias: str = ""
    """Python-compatible alias for use in code."""

    # :attr: _fwk
    _fwk: str = "PHYSICAL"
    """Framework context (PHYSICAL, COMPUTATION, SOFTWARE, CUSTOM)."""

    # :attr: _cat
    _cat: str = "NUM"
    """Category of analysis (SYM, NUM, HYB)."""

    # ========================================================================
    # Coefficient and Expression Management
    # ========================================================================

    # :attr: _coefficient
    _coefficient: Coefficient = field(default_factory=Coefficient)
    """Coefficient for the simulation."""

    # :attr: _pi_expr
    _pi_expr: Optional[str] = None
    """LaTeX expression to analyze."""

    # :attr: _sym_func
    _sym_func: Optional[Union[sp.Expr, sp.Basic]] = None
    """Sympy expression object for the coefficient (Mul, Add, Pow, Symbol, etc.)."""

    # :attr: _exe_func
    _exe_func: Optional[Callable[..., Union[float, np.ndarray]]] = None
    """Compiled executable function for evaluation of the coefficient."""

    # ========================================================================
    # Variable Management
    # ========================================================================

    # :attr: _variables
    _variables: Dict[str, Variable] = field(default_factory=dict)
    """Dictionary of variables in the expression."""

    # :attr: _symbols
    _symbols: Dict[str, sp.Symbol] = field(default_factory=dict)
    """Map from variable names (strings) to sympy Symbols."""

    # :attr: _aliases
    _aliases: Dict[str, sp.Symbol] = field(default_factory=dict)
    """Map from Variable aliases to sympy Symbols."""

    # :attr: _latex_to_py
    _latex_to_py: Dict[str, str] = field(default_factory=dict)
    """Map from LaTeX symbols to Python-compatible names."""

    # :attr: _py_to_latex
    _py_to_latex: Dict[str, str] = field(default_factory=dict)
    """Map from Python-compatible names to LaTeX symbols."""

    # :attr: _var_symbols
    _var_symbols: List[str] = field(default_factory=list)
    """List of variable names extracted from expression."""
    # ========================================================================
    # Simulation Configuration
    # ========================================================================

    # :attr: _experiments
    _experiments: int = -1
    """Number of simulation iterations to run."""

    # :attr: _distributions
    _distributions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Variable sampling distributions and specifications that includes:
        - 'dtype': Distribution type name.
        - 'params': Distribution parameters (mean, std_dev, etc.).
        - 'func': Function for sampling, usually in Lambda format.
        - 'depends': List of variables this variable depends on.
    """

    # :attr: _dependencies
    _dependencies: Dict[str, List[str]] = field(default_factory=dict, init=False)
    """Variable dependencies for simulations."""

    # :attr: _simul_cache
    _simul_cache: Dict[str, NDArray[np.float64]] = field(default_factory=dict)
    """Working sampled values during each simulation iteration. Memory cache."""

    # ========================================================================
    # Results and Inputs
    # ========================================================================

    # :attr: inputs
    inputs: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64))
    """Sample value range for the simulation."""

    # :attr: _results
    _results: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64))
    """Raw simulation results."""

    # ========================================================================
    # Statistics
    # ========================================================================

    # :attr: _mean
    _mean: float = np.nan
    """Mean value of simulation results."""

    # :attr: _median
    _median: float = np.nan
    """Median value of simulation results."""

    # :attr: _std_dev
    _std_dev: float = np.nan
    """Standard deviation of simulation results."""

    # :attr: _variance
    _variance: float = np.nan
    """Variance of simulation results."""

    # :attr: _min
    _min: float = np.nan
    """Minimum value in simulation results."""

    # :attr: _max
    _max: float = np.nan
    """Maximum value in simulation results."""

    # :attr: _count
    _count: int = 0
    """Number of valid simulation results."""

    # :attr: _statistics
    _statistics: Optional[Dict[str, float]] = None
    """Statistical summary of the Monte Carlo simulation results."""

    # ========================================================================
    # Initialization
    # ========================================================================

    def __post_init__(self) -> None:
        """*__post_init__()* Initializes the Monte Carlo simulation."""
        # Initialize from base class
        super().__post_init__()

        # Validate coefficient
        if not self._coefficient.pi_expr:
            raise ValueError("Coefficient must have a valid expression")

        # Derive expression from coefficient
        self._pi_expr = self._coefficient.pi_expr

        # Set default symbol if not specified
        if not self._sym:
            self._sym = f"MC_\\Pi_{{{self._idx}}}" if self._idx >= 0 else "MC_\\Pi_{}"

        # Set default Python alias if not specified
        if not self._alias:
            self._alias = latex_to_python(self._sym)

        # Set name and description if not already set
        if not self._name:
            self._name = f"{self._sym} Monte Carlo"

        if not self.description:
            self.description = f"Monte Carlo simulation for {self._sym}"

        if self._pi_expr:
            # Parse the expression
            self._parse_expression(self._pi_expr)

        # Preallocate full array space with NaN
        n_sym = len(self._symbols)
        if n_sym > 0:
            # Only allocate if we have variables
            if self.inputs.size == 0:  # Check size, not None
                self.inputs = np.full((self._experiments, n_sym),
                                      np.nan,
                                      dtype=np.float64)
            if self._results.size == 0:  # Check size, not None
                self._results = np.full((self._experiments, 1),
                                        np.nan,
                                        dtype=np.float64)

        # Only initialize cache if not already provided
        if not self._simul_cache:
            # Create local cache only if no external cache provided
            for var in self._variables.keys():
                self._simul_cache[var] = np.full((self._experiments, 1),
                                                 np.nan,
                                                 dtype=np.float64)

        # Statistics initialized to NaN (not calculated yet)
        self._mean = np.nan
        self._median = np.nan
        self._std_dev = np.nan
        self._variance = np.nan
        self._min = np.nan
        self._max = np.nan

        # Zero makes sense here
        self._count = 0

    # ========================================================================
    # Foundation and Configuration
    # ========================================================================

    def _validate_readiness(self) -> None:
        """*_validate_readiness()* Checks if the simulation can be performed.

        Raises:
            ValueError: If the simulation is not ready due to missing variables, executable function, distributions, or invalid number of iterations.
        """
        if not self._variables:
            raise ValueError("No variables found in the expression.")
        if not self._sym_func:
            raise ValueError("No expression has been defined for analysis.")
        if not self._distributions:
            _vars = self._variables
            missing = [v for v in _vars if v not in self._distributions]
            if missing:
                _msg = f"Missing distributions for variables: {missing}"
                raise ValueError(_msg)
        if self._experiments < 1:
            _msg = f"Invalid number of iterations: {self._experiments}"
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

        # Save coefficient
        self._coefficient = coef

        # Set expression
        self._pi_expr = coef.pi_expr

        # Parse coefficient expression
        if coef._pi_expr:
            self._parse_expression(self._pi_expr)

        # Set name and description if not already set
        if not self._name:
            self._name = f"{coef.name} Monte Carlo Experiments"
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

            if self._sym_func is None:
                raise ValueError("Parsing returned None")

            # Store the sympy expression
            self._sym_func = self._sym_func

            # Create symbol mapping
            maps = create_latex_mapping(expr)

            symbols_raw: Dict[Any, sp.Symbol] = maps[0]
            aliases_raw: Dict[str, sp.Symbol] = maps[1]
            latex_to_py: Dict[str, str] = maps[2]
            py_to_latex: Dict[str, str] = maps[3]

            # Convert Symbol keys to strings
            self._symbols = {
                str(k): v for k, v in symbols_raw.items()
            }
            self._aliases = aliases_raw
            self._latex_to_py = latex_to_py
            self._py_to_latex = py_to_latex

            # Substitute LaTeX symbols with Python symbols
            for latex_sym_key, py_sym in symbols_raw.items():
                if self._sym_func is None:
                    break

                # Handle both string and Symbol keys
                if isinstance(latex_sym_key, sp.Symbol):
                    # subs() returns Basic, which is fine
                    self._sym_func = self._sym_func.subs(latex_sym_key, py_sym)
                else:
                    # Try to find the symbol by name
                    latex_symbol = sp.Symbol(str(latex_sym_key))
                    self._sym_func = self._sym_func.subs(latex_symbol, py_sym)

            # Get Python variable names as strings
            if self._sym_func is not None and hasattr(self._sym_func, 'free_symbols'):
                free_symbols = self._sym_func.free_symbols
                self._var_symbols = sorted([str(s) for s in free_symbols])
            else:
                raise ValueError("Expression has no free symbols")

        except Exception as e:
            _msg = f"Failed to parse expression: {str(e)}"
            raise ValueError(_msg)

    # ========================================================================
    # Simulation Execution
    # ========================================================================

    def _generate_sample(self,
                         var: Variable,
                         memory: Dict[str, float]) -> float:
        """*_generate_sample()* Generate a sample for a given variable.

        Args:
            var (Variable): The variable to generate a sample for.
            memory (Dict[str, float]): The current iteration values.

        Returns:
            float: The generated sample.
        """
        # Initialize sample
        data: float = -1.0

        # relevant data type, HOTFIX
        _type = (list, tuple, np.ndarray)

        # Get dependency values from memory
        chace_deps = []
        for dep in var.depends:
            if dep in memory:
                dep_val = memory[dep]
                # If dependency is a list/tuple/array, take the last value
                if isinstance(dep_val, (list, tuple, np.ndarray)):
                    dep_val = dep_val[-1]
                chace_deps.append(dep_val)

        # print(f"chace_deps: {chace_deps}")

        # if the distribution function is defined
        if var._dist_func is not None:
            # If the variable is independent
            if not var.depends:
                data = var.sample()

            # If the variable has dependencies
            elif len(var.depends) == len(chace_deps):
                raw_data = var.sample(*chace_deps)
                # print(f"raw_data: {raw_data}")

                # Handle array-like results
                if isinstance(raw_data, _type):
                    # get the last number
                    data = raw_data[-1]

                    # adjust the memory accordingly to the rest of the list
                    for dep in var.depends:
                        if dep in memory:
                            memory[dep] = raw_data[var.depends.index(dep)]
                # otherwise, its a number
                else:
                    data = raw_data

        # print(f"dependencies keys {var.depends}")
        # print(f"memory: {memory}")

        # Store sample in memory
        memory[var.sym] = float(data)

        # return sampled data
        return data

    def run(self, iters: Optional[int] = None) -> None:
        """*run()* Execute the Monte Carlo simulation.

        Args:
            iters (int, optional): Number of iterations to run. If None, uses _experiments.

        Raises:
            ValueError: If simulation is not ready or encounters errors during execution.
        """
        # Validate simulation readiness
        self._validate_readiness()

        # Set iterations if necessary
        if iters is not None:
            self._experiments = iters

        # Clear previous results, inputs, and intermediate values
        self._reset_memory()

        # Create lambdify function using Python symbols
        aliases = [self._aliases[v] for v in self._var_symbols]
        self._exe_func = lambdify(aliases, self._sym_func, "numpy")

        if self._exe_func is None:
            raise ValueError("Failed to create executable function")

        # Run experiment loop
        for _iter in range(self._experiments):
            try:
                # Dict to store sample memory for the iteration
                memory: Dict[str, float] = {}

                # run through all variables
                for var in self._variables.values():
                    # Check for cached value
                    cached_val = self._get_cached_value(var.sym, _iter)

                    # if no cached value, generate new sample
                    if cached_val is None or np.isnan(cached_val):
                        # Generate sample for the variable
                        val = self._generate_sample(var, memory)
                        # Store the sample in the iteration values
                        memory[var.sym] = val
                        self._set_cached_value(var.sym, _iter, val)

                    # otherwise use cached value
                    else:
                        # Use cached value
                        memory[var.sym] = cached_val

                # Prepare sorted/ordered values from memory for evaluation
                sorted_vals = [memory[var] for var in self._latex_to_py]

                # FIXME hotfix for queue functions
                _type = (list, tuple, np.ndarray)
                # Handle adjusted values
                if any(isinstance(v, _type) for v in sorted_vals):
                    sorted_vals = [
                        v[-1] if isinstance(v, _type) else v for v in sorted_vals]

                # Evaluate the coefficient
                result = float(self._exe_func(*sorted_vals))

                # Handle array results
                if isinstance(result, _type):
                    result = result[-1]
                    sorted_vals = [v[-1] for v in result]

                # Save simulation inputs and results
                self.inputs[_iter, :] = sorted_vals
                self._results[_iter] = result

            except Exception as e:
                _msg = f"Error during simulation run {_iter}: {str(e)}"
                raise ValueError(_msg)

        # Calculate statistics
        self._calculate_statistics()

    # ========================================================================
    # Memory and Statistics Management
    # ========================================================================

    def _reset_memory(self) -> None:
        """*_reset_memory()* Reset results and inputs arrays."""
        # reseting full array space with NaN
        n_sym = len(self._symbols)
        self.inputs = np.full((self._experiments, n_sym), np.nan)
        self._results = np.full((self._experiments, 1), np.nan)

        # # reset intermediate values
        # for var in self._variables.keys():
        #     self._simul_cache[var] = np.full((self._experiments, 1),
        #                                      np.nan,
        #                                      dtype=np.float64)

    def _reset_statistics(self) -> None:
        """*_reset_statistics()* Reset all statistical attributes to default values."""
        # reset statistics to NaN or zero
        self._mean = np.nan
        self._median = np.nan
        self._std_dev = np.nan
        self._variance = np.nan
        self._min = np.nan
        self._max = np.nan
        self._count = 0

    def _calculate_statistics(self) -> None:
        """*_calculate_statistics()* Calculate statistical properties of simulation results."""
        # Check for empty array (size == 0), not None
        if self._results.size == 0:
            raise ValueError("No results available. Run simulation first.")

        else:
            self._mean = float(np.mean(self._results))
            self._median = float(np.median(self._results))
            self._std_dev = float(np.std(self._results))
            self._variance = float(np.var(self._results))
            self._min = float(np.min(self._results))
            self._max = float(np.max(self._results))
            self._count = len(self._results)

    def get_confidence_interval(self,
                                conf: float = 0.95) -> Tuple[float, float]:
        """*get_confidence_interval()* Calculate the confidence interval.

        Args:
            conf (float, optional): Confidence level for the interval. Defaults to 0.95.

        Raises:
            ValueError: If no results are available or if the confidence level is invalid.

        Returns:
            Tuple[float, float]: Lower and upper bounds of the confidence interval.
        """
        if self._results.size == 0:
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

    # ========================================================================
    # simulation cache management
    # ========================================================================

    def _validate_cache_locations(self,
                                  var_syms: Union[str, List[str]],
                                  idx: int) -> bool:
        """*_validate_cache_locations()* Check if cache locations are valid for variable(s) at the iteration.

        Args:
            var_syms (Union[str, List[str]]): Variable symbol(s) to check.
            idx (int): Iteration index to check.

        Returns:
            bool: True if all cache locations are valid (including NaN placeholders), False otherwise.
        """
        # Convert single string to list for uniform handling
        syms = [var_syms] if isinstance(var_syms, str) else var_syms

        # Start with assumption that cache is invalid
        valid = False

        # Check each symbol
        for var_sym in syms:
            # Reset validity check for each variable
            var_valid = False

            # Get cache array for the variable
            cache_array = self._simul_cache.get(var_sym, None)

            # Check if cache exists and location is valid
            if cache_array is not None:
                # Check if index is within bounds
                if idx < cache_array.shape[0] and idx >= 0:
                    # Location exists - valid regardless of whether value is NaN
                    # (NaN is a valid placeholder for uncomputed values)
                    var_valid = True

            # If any variable is invalid, entire check fails
            if not var_valid:
                return False

        # All variables passed validation
        valid = True
        return valid

    def _get_cached_value(self, var_sym: str, idx: int) -> Optional[float]:
        """*_get_cached_value()* Retrieve cached value for variable at the iteration.

        Args:
            var_sym (str): Variable symbol.
            idx (int): Iteration index.

        Returns:
            Optional[float]: Cached value if valid, None otherwise.
        """
        # Initialize return value
        cache_data = None

        # Check if cache location is valid
        if self._validate_cache_locations(var_sym, idx):
            # Retrieve cached data
            cache_data = self._simul_cache[var_sym][idx]
            # if value is not NaN (valid location, but no data yet)
            if not np.isnan(cache_data):
                # cast to float the computed value
                cache_data = float(cache_data)
        # return valid cache location
        return cache_data

    def _set_cached_value(self,
                          var_sym: str,
                          idx: int,
                          val: Union[float, Dict]) -> None:
        """*_set_cached_value()* Store value in cache for variable at the iteration.

        Args:
            var_sym (str): Variable symbol.
            idx (int): Iteration index.
            val (Union[float, Dict]): Value to cache. It can be a normal number (float) or a memory cache correction (dict).

        Raises:
            ValueError: If cache location is invalid.
        """
        # Normalize input to dictionary format
        cache_updates = val if isinstance(val, dict) else {var_sym: val}

        # Validate all cache locations
        if not self._validate_cache_locations(list(cache_updates.keys()), idx):
            invalid_vars = list(cache_updates.keys())
            _msg = f"Invalid cache location at index {idx}. "
            _msg += f"For variables: {invalid_vars}"
            raise ValueError(_msg)

        # Store all values
        for k, v in cache_updates.items():
            self._simul_cache[k][idx, 0] = v

        # # if value is a float, store directly
        # if isinstance(val, float):

        #     # if cache location is invalid, raise error
        #     if self._validate_cache_locations(var_sym, idx) is False:
        #         _msg = f"Invalid cache location for: {var_sym}[{idx}]"
        #         raise ValueError(_msg)

        #     self._simul_cache[var_sym][idx, 0] = val
        # # if value is a dict, update multiple cache entries
        # elif isinstance(val, Dict):
        #     if any([not self._validate_cache_locations(k, idx) for k in val.keys()]):
        #         _msg = f"Invalid cache location at index {idx}. "
        #         invalid_vars = [
        #             k for k in val.keys() if not self._validate_cache_locations(k, idx)
        #         ]
        #         _msg += f"Invalid Cache Variables: {invalid_vars}"
        #         raise ValueError(_msg)
        #     for k, v in val.items():
        #         self._simul_cache[k][idx, 0] = v

    # ========================================================================
    # Results Extraction
    # ========================================================================

    def extract_results(self) -> Dict[str, NDArray[np.float64]]:
        """*extract_results()* Extract simulation results.

        Returns:
            Dict[str, NDArray[np.float64]]: Dictionary containing simulation results.
        """
        export: Dict[str, NDArray[np.float64]] = {}

        # Extract all values for each variable (column)
        for i, var in enumerate(self._py_to_latex.values()):
            # Get the entire column for this variable (all simulation runs)
            column = self.inputs[:, i]

            # Use a meaningful key that includes variable name and coefficient
            key = f"{var}@{self._coefficient.sym}"
            export[key] = column

        # Add the coefficient results
        export[self._coefficient.sym] = self._results.flatten()
        return export

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def variables(self) -> Dict[str, Variable]:
        """*variables* Get the variables involved in the simulation.

        Returns:
            Dict[str, Variable]: Dictionary of variable symbols and Variable objects.
        """
        return self._variables.copy()

    @property
    def coefficient(self) -> Optional[Coefficient]:
        """*coefficient* Get the coefficient associated with the simulation.

        Returns:
            Optional[Coefficient]: The associated Coefficient object, or None.
        """
        return self._coefficient

    @property
    def results(self) -> NDArray[np.float64]:
        """*results* Raw simulation results.

        Returns:
            NDArray[np.float64]: Copy of the simulation results.

        Raises:
            ValueError: If no results are available.
        """
        if self._results.size == 0:
            raise ValueError("No results available. Run the simulation first.")
        return self._results.copy()

    @property
    def statistics(self) -> Dict[str, float]:
        """*statistics* Get the statistical analysis of simulation results.

        Raises:
            ValueError: If no results are available.

        Returns:
            Dict[str, float]: Dictionary containing statistical properties.
        """
        if self._results.size == 0:
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
    def experiments(self) -> int:
        """*experiments* Number of simulation experiments.

        Returns:
            int: Current number of experiments.
        """
        return self._experiments

    @experiments.setter
    def experiments(self, val: int) -> None:
        """*experiments* Set the number of simulation runs.

        Args:
            val (int): Number of experiments to run the simulation.

        Raises:
            ValueError: If the number of experiments is not positive.
        """
        if val < 1:
            _msg = f"Number of experiments must be positive. Got: {val}"
            raise ValueError(_msg)
        self._experiments = val

    @property
    def distributions(self) -> Dict[str, Dict[str, Any]]:
        """*distributions* Get the variable distributions.

        Returns:
            Dict[str, Dict[str, Any]]: Current variable distributions.
        """
        return self._distributions.copy()

    @distributions.setter
    def distributions(self, val: Dict[str, Dict[str, Any]]) -> None:
        """*distributions* Set the variable distributions.

        Args:
            val (Dict[str, Dict[str, Any]]): New variable distributions.

        Raises:
            ValueError: If the distributions are invalid.
        """
        if not all(callable(v["func"]) for v in val.values()):
            _msg = "All distributions must have callable 'func' functions."
            inv = [k for k, v in val.items() if not callable(v["func"])]
            _msg += f" Invalid entries: {inv}"
            raise ValueError(_msg)
        self._distributions = val

    @property
    def dependencies(self) -> Dict[str, List[str]]:
        """*dependencies* Get variable dependencies.

        Returns:
            Dict[str, List[str]]: Dictionary of variable dependencies.
        """
        return self._dependencies

    @dependencies.setter
    def dependencies(self, val: Dict[str, List[str]]) -> None:
        """*dependencies* Set variable dependencies.

        Args:
            val (Dict[str, List[str]]): New variable dependencies.
        """
        if not isinstance(val, dict):
            _msg = f"Dependencies must be dict, got {type(val)}"
            raise TypeError(_msg)
        self._dependencies = val

    # Individual statistics properties

    @property
    def mean(self) -> float:
        """*mean* Mean value of simulation results.

        Returns:
            float: Mean value.

        Raises:
            ValueError: If no results are available.
        """
        if self._results.size == 0:
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
        if self._results.size == 0:
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
        if self._results.size == 0:
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
        if self._results.size == 0:
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
        if self._results.size == 0:
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
        if self._results.size == 0:
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
        if self._results.size == 0:
            _msg = "No statistics available. Run the simulation first."
            raise ValueError(_msg)
        return self._count

    @property
    def summary(self) -> Dict[str, float]:
        """*summary* Get the statistical analysis of simulation results.

        Raises:
            ValueError: If no results are available.

        Returns:
            Dict[str, float]: Dictionary containing statistical properties.
        """
        if self._results.size == 0:
            _msg = "No statistics available. Run the simulation first."
            raise ValueError(_msg)

        # Build summary dictionary from individual attributes
        self._summary = {
            "mean": self._mean,
            "median": self._median,
            "std_dev": self._std_dev,
            "variance": self._variance,
            "min": self._min,
            "max": self._max,
            "count": self._count
        }
        return self._summary

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def clear(self) -> None:
        """*clear()* Reset all attributes to default values."""
        # Reset base class attributes
        self._idx = -1
        self._sym = "MC_\\Pi_{}"
        self._alias = ""
        self._fwk = "PHYSICAL"
        self._name = ""
        self.description = ""

        # Reset simulation attributes
        self._pi_expr = None
        self._sym_func = None
        self._exe_func = None
        self._variables = {}
        self._latex_to_py = {}
        self._py_to_latex = {}
        self._experiments = -1
        self._distributions = {}

        # reset results, inputs and intermediate values
        self._reset_memory()

        # Reset statistics
        self._reset_statistics()

    # ========================================================================
    # Serialization
    # ========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """*to_dict()* Convert simulation to dictionary representation."""
        return {
            # Foundation class attributes
            "idx": self._idx,
            "sym": self._sym,
            "alias": self._alias,
            "fwk": self._fwk,
            "name": self._name,
            "description": self.description,
            # Simulation attributes
            "pi_expr": self._pi_expr,
            "variables": self._variables,
            "iterations": self._experiments,
            # Results
            "mean": self._mean,
            "median": self._median,
            "std_dev": self._std_dev,
            "variance": self._variance,
            "min": self._min,
            "max": self._max,
            "count": self._count,
            "inputs": self.inputs.tolist() if self.inputs.size > 0 else None,
            "results": self._results.tolist() if self._results.size > 0 else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MonteCarlo":
        """*from_dict()* Create simulation from dictionary representation.

        Args:
            data (Dict[str, Any]): Dictionary representation.

        Returns:
            MonteCarlo: New simulation instance.
        """
        # Create basic instance
        instance = cls(
            _name=data.get("name", ""),
            description=data.get("description", ""),
            _idx=data.get("idx", -1),
            _sym=data.get("sym", "MC_\\Pi_{}"),
            _fwk=data.get("fwk", "PHYSICAL"),
            _alias=data.get("alias", ""),
            _cat=data.get("cat", "NUM"),
            _pi_expr=data.get("pi_expr", None),
            _experiments=data.get("iterations", -1),
        )

        # The to_dict() method stores them at the top level
        instance._mean = data.get("mean", np.nan)
        instance._median = data.get("median", np.nan)
        instance._std_dev = data.get("std_dev", np.nan)
        instance._variance = data.get("variance", np.nan)
        instance._min = data.get("min", np.nan)
        instance._max = data.get("max", np.nan)
        instance._count = data.get("count", 0)

        # Optionally set inputs and results if available
        if "inputs" in data and data["inputs"] is not None:
            instance.inputs = np.array(data["inputs"], dtype=np.float64)

        if "results" in data and data["results"] is not None:
            instance._results = np.array(data["results"], dtype=np.float64)

        # Optionally set variables if available
        if "variables" in data and data["variables"]:
            for sym, specs in data["variables"].items():
                instance._variables[sym] = Variable(**specs)

        return instance
