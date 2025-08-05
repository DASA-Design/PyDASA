# -*- coding: utf-8 -*-
"""
Module for **MonteCarloSim** analysis in *PyDASA*.

This module provides the MonteCarloSim class for performing Monte Carlo simulations on dimensionless coefficients derived from dimensional analysis.

*IMPORTANT:* Based on the theory from:

    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Generic, Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sympy import lambdify

# Import validation base classes
from src.pydasa.core.basic import Validation

# Import related classes
from src.pydasa.buckingham.vashchy import Coefficient

# Import utils
from src.pydasa.utils.default import T
from src.pydasa.utils.latex import parse_latex, create_latex_mapping

# Import configuration
from src.pydasa.utils import config as cfg


@dataclass
class MonteCarloSim(Validation, Generic[T]):
    """**MonteCarloSim** class for stochastic analysis in *PyDASA*.

    Performs Monte Carlo simulations on dimensionless coefficients to analyze the coefficient's distribution and sensitivity to input variable variations.

    Attributes:
        # Identification and Classification
        name (str): User-friendly name of the Monte Carlo simulation.
        description (str): Brief summary of the simulation.
        _idx (int): Index/precedence of the simulation.
        _sym (str): Symbol representation (LaTeX or alphanumeric).
        _pyalias (str): Python-compatible alias for use in code.
        _fwk (str): Framework context (PHYSICAL, COMPUTATION, SOFTWARE, CUSTOM).
        
        # Expression Management
        _pi_expr (str): LaTeX expression to analyze.
        _sym_func (Callable): Sympy function of the simulation.
        _exe_func (Callable): Executable function for numerical evaluation.
        _variables (List[str]): Variable symbols in the expression.
        _latex_to_py (Dict[str, str]): Mapping from LaTeX to Python variable names.
        _py_to_latex (Dict[str, str]): Mapping from Python to LaTeX variable names.
        
        # Simulation Configuration
        _iterations (int): Number of simulation runs.
        _distributions (Dict[str, Callable]): Variable sampling distributions.
        _bounds (Dict[str, Tuple[float, float]]): Min/max bounds for each variable.
        
        # Results
        _results (List[float]): Raw simulation results.
        
        # Statistics
        _mean (float): Mean value of simulation results.
        _median (float): Median value of simulation results.
        _std_dev (float): Standard deviation of simulation results.
        _variance (float): Variance of simulation results.
        _min (float): Minimum value in simulation results.
        _max (float): Maximum value in simulation results.
        _count (int): Number of valid simulation results.
    """

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
    _variables: List[str] = field(default_factory=list)
    """Variable symbols in the expression."""

    # :attr: _latex_to_py
    _latex_to_py: Dict[str, str] = field(default_factory=dict)
    """Mapping from LaTeX symbols to Python-compatible names."""

    # :attr: _py_to_latex
    _py_to_latex: Dict[str, str] = field(default_factory=dict)
    """Mapping from Python-compatible names to LaTeX symbols."""
    
    # Simulation configuration
    # :attr: _iterations
    _iterations: int = 1000
    """Number of simulation runs."""
    
    # :attr: _distributions
    _distributions: Dict[str, Callable] = field(default_factory=dict)
    """Variable sampling distributions."""
    
    # :attr: _bounds
    _bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    """Min/max bounds for each variable."""
    
    # Results
    # :attr: _results
    _results: List[float] = field(default_factory=list)
    """Raw simulation results."""
    
    # Individual statistics attributes
    # :attr: _mean
    _mean: float = 0.0
    """Mean value of simulation results."""
    
    # :attr: _median
    _median: float = 0.0
    """Median value of simulation results."""
    
    # :attr: _std_dev
    _std_dev: float = 0.0
    """Standard deviation of simulation results."""
    
    # :attr: _variance
    _variance: float = 0.0
    """Variance of simulation results."""
    
    # :attr: _min
    _min: float = 0.0
    """Minimum value in simulation results."""
    
    # :attr: _max
    _max: float = 0.0
    """Maximum value in simulation results."""
    
    # :attr: _count
    _count: int = 0
    """Number of valid simulation results."""

    def __post_init__(self) -> None:
        """*__post_init__()* Initializes the Monte Carlo simulation."""
        # Initialize from base class
        super().__post_init__()

        # Set default symbol if not specified
        if not self._sym:
            self._sym = f"MC_\\Pi_{{{self._idx}}}" if self._idx >= 0 else "MC_\\Pi_{}"
        
        # Set default Python alias if not specified
        if not self._pyalias:
            from src.pydasa.utils.latex import latex_to_python
            self._pyalias = latex_to_python(self._sym)

        # Set name and description if not already set
        if not self.name:
            self.name = f"{self._sym} Monte Carlo"
        if not self.description:
            self.description = f"Monte Carlo simulation for {self._sym}"

        if self._pi_expr:
            # Parse the expression
            self._parse_expression(self._pi_expr)

    def _validate_simulation_ready(self) -> None:
        """*_validate_simulation_ready()* Checks if the simulation can be performed."""
        if not self._variables:
            raise ValueError("No variables found in the expression.")
        if not self._exe_func:
            raise ValueError("No executable function has been defined for simulation.")
        if not self._distributions:
            missing = [v for v in self._variables if v not in self._distributions]
            if missing:
                raise ValueError(f"Missing distributions for variables: {missing}")
        if self._iterations <= 0:
            raise ValueError(f"Invalid number of iterations: {self._iterations}")

    def set_coefficient(self, coef: Coefficient) -> None:
        """*set_coefficient()* Configure simulation from a coefficient."""
        if not coef.pi_expr:
            raise ValueError("Coefficient does not have a valid expression.")

        # Set expression
        self._pi_expr = coef.pi_expr
        
        # Parse coefficient expression
        self._parse_expression(self._pi_expr)
        
        # Set name and description if not already set
        if not self.name:
            self.name = f"{coef.name} Monte Carlo"
        if not self.description:
            self.description = f"Monte Carlo simulation for {coef.name}"

    def _parse_expression(self, expr: str) -> None:
        """*_parse_expression()* Parse the LaTeX expression into a sympy function."""
        try:
            # Parse the expression
            self._sym_func = parse_latex(expr)

            # Create symbol mapping
            maps = create_latex_mapping(expr)
            symbols_map, aliases_map, self._latex_to_py, self._py_to_latex = maps

            # Substitute LaTeX symbols with Python symbols
            for latex_sym, py_sym in symbols_map.items():
                self._sym_func = self._sym_func.subs(latex_sym, py_sym)

            # Get Python variable names
            self._variables = sorted([str(s) for s in self._sym_func.free_symbols])

            # Create executable function
            self._exe_func = lambdify([aliases_map[v] for v in self._variables], 
                                      self._sym_func, "numpy")
            
        except Exception as e:
            _msg = f"Failed to parse expression: {str(e)}"
            raise ValueError(_msg)

    def set_distribution(self, variable: str, distribution: Callable, 
                         bounds: Tuple[float, float] = None) -> None:
        """*set_distribution()* Set the sampling distribution for a variable."""
        # Handle both LaTeX and Python variable names
        var_py = self._latex_to_py.get(variable, variable)
        
        if var_py not in self._variables:
            _msg = f"Variable '{variable}' not found in expression. "
            _msg += f"Available variables: {[self._py_to_latex.get(v, v) for v in self._variables]}"
            raise ValueError(_msg)
            
        if not callable(distribution):
            raise ValueError(f"Distribution must be callable. Got: {type(distribution)}")
            
        self._distributions[var_py] = distribution
        
        if bounds:
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise ValueError(f"Bounds must be a tuple of (min, max). Got: {bounds}")
            self._bounds[var_py] = bounds

    def set_uniform_distribution(self, variable: str, min_val: float, max_val: float) -> None:
        """*set_uniform_distribution()* Set a uniform distribution for a variable."""
        if min_val >= max_val:
            raise ValueError(f"Minimum value {min_val} must be less than maximum value {max_val}")
            
        def uniform_sampler():
            return np.random.uniform(min_val, max_val)
            
        self.set_distribution(variable, uniform_sampler, (min_val, max_val))
        
    def set_normal_distribution(self, variable: str, mean: float, std_dev: float, 
                               bounds: Tuple[float, float] = None) -> None:
        """*set_normal_distribution()* Set a normal distribution for a variable."""
        if std_dev <= 0:
            raise ValueError(f"Standard deviation must be positive. Got: {std_dev}")
            
        if bounds:
            min_val, max_val = bounds
            def normal_sampler():
                # Truncated normal distribution
                while True:
                    val = np.random.normal(mean, std_dev)
                    if min_val <= val <= max_val:
                        return val
        else:
            def normal_sampler():
                return np.random.normal(mean, std_dev)
                
        self.set_distribution(variable, normal_sampler, bounds)
        
    def set_iterations(self, iterations: int) -> None:
        """*set_iterations()* Set the number of simulation runs."""
        if iterations <= 0:
            raise ValueError(f"Number of iterations must be positive. Got: {iterations}")
        self._iterations = iterations
        
    def run(self) -> None:
        """*run()* Execute the Monte Carlo simulation."""
        # Validate simulation readiness
        self._validate_simulation_ready()
        
        # Clear previous results
        self._results.clear()
        self._reset_statistics()
        
        # Run simulation
        for _ in range(self._iterations):
            try:
                # Sample values from distributions
                samples = {}
                for var in self._variables:
                    if var in self._distributions:
                        samples[var] = self._distributions[var]()
                    else:
                        raise ValueError(f"Missing distribution for variable: {var}")
                
                # Prepare ordered values for function evaluation
                ordered_values = [samples[var] for var in self._variables]
                
                # Evaluate the coefficient
                result = float(self._exe_func(*ordered_values))
                self._results.append(result)
                
            except Exception as e:
                # Handle numerical errors (e.g., division by zero)
                continue
        
        # Calculate statistics
        if self._results:
            self._calculate_statistics()
        else:
            raise ValueError("No valid results were generated. Check your distributions.")
    
    def _reset_statistics(self) -> None:
        """*_reset_statistics()* Reset all statistical attributes to default values."""
        self._mean = 0.0
        self._median = 0.0
        self._std_dev = 0.0
        self._variance = 0.0
        self._min = 0.0
        self._max = 0.0
        self._count = 0
            
    def _calculate_statistics(self) -> None:
        """*_calculate_statistics()* Calculate statistical properties of simulation results."""
        results = np.array(self._results)
        
        # Calculate and store each statistic separately
        self._mean = float(np.mean(results))
        self._median = float(np.median(results))
        self._std_dev = float(np.std(results))
        self._variance = float(np.var(results))
        self._min = float(np.min(results))
        self._max = float(np.max(results))
        self._count = len(results)
        
    def plot_histogram(self, bins: int = 30, figsize: Tuple[int, int] = (10, 6),
                      title: str = None, save_path: str = None) -> None:
        """*plot_histogram()* Plot a histogram of simulation results."""
        if not self._results:
            raise ValueError("No simulation results to plot. Run the simulation first.")
            
        plt.figure(figsize=figsize)
        plt.hist(self._results, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add statistics to the plot
        stats_text = (
            f"Mean: {self._mean:.4f}\n"
            f"Std Dev: {self._std_dev:.4f}\n"
            f"Min: {self._min:.4f}\n"
            f"Max: {self._max:.4f}\n"
            f"Samples: {self._count}"
        )
        plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    va='top', fontsize=10)
        
        # Add title and labels
        plot_title = title if title else f"Monte Carlo Simulation: {self._pi_expr}"
        plt.title(plot_title)
        plt.xlabel("Coefficient Value")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def get_statistics(self) -> Dict[str, float]:
        """*get_statistics()* Get the statistical analysis of simulation results."""
        if not self._results:
            raise ValueError("No statistics available. Run the simulation first.")
            
        # Build statistics dictionary from individual attributes
        return {
            "mean": self._mean,
            "median": self._median,
            "std_dev": self._std_dev,
            "variance": self._variance,
            "min": self._min,
            "max": self._max,
            "count": self._count
        }
    
    def get_results(self) -> List[float]:
        """*get_results()* Get the raw simulation results."""
        if not self._results:
            raise ValueError("No results available. Run the simulation first.")
        return self._results.copy()
    
    def get_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """*get_confidence_interval()* Calculate confidence interval for the mean."""
        if not self._results:
            raise ValueError("No results available. Run the simulation first.")
            
        if not 0 < confidence < 1:
            raise ValueError(f"Confidence level must be between 0 and 1. Got: {confidence}")
        
        # Calculate the margin of error using the t-distribution
        from scipy import stats
        margin = stats.t.ppf((1 + confidence) / 2, self._count - 1) * self._std_dev / np.sqrt(self._count)
        
        return (self._mean - margin, self._mean + margin)
        
    def clear(self) -> None:
        """*clear()* Reset all attributes to default values."""
        # Reset base class attributes
        self._idx = -1
        self._sym = "MC_\\Pi_{}"
        self._pyalias = ""
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
        self._bounds = {}
        self._results = []
        
        # Reset statistics
        self._reset_statistics()
        
    def to_dict(self) -> Dict[str, Any]:
        """*to_dict()* Convert simulation to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "idx": self._idx,
            "sym": self._sym,
            "pyalias": self._pyalias,
            "fwk": self._fwk,
            "pi_expr": self._pi_expr,
            "variables": self._variables,
            "iterations": self._iterations,
            "bounds": self._bounds,
            "statistics": {
                "mean": self._mean,
                "median": self._median,
                "std_dev": self._std_dev,
                "variance": self._variance,
                "min": self._min,
                "max": self._max,
                "count": self._count
            }
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
            _pyalias=data.get("pyalias", ""),
            _pi_expr=data.get("pi_expr", None),
            _iterations=data.get("iterations", 1000)
        )
        
        # Optionally set statistics if available
        if "statistics" in data:
            stats = data["statistics"]
            instance._mean = stats.get("mean", 0.0)
            instance._median = stats.get("median", 0.0) 
            instance._std_dev = stats.get("std_dev", 0.0)
            instance._variance = stats.get("variance", 0.0)
            instance._min = stats.get("min", 0.0)
            instance._max = stats.get("max", 0.0)
            instance._count = stats.get("count", 0)
            
        return instance