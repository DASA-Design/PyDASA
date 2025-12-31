# -*- coding: utf-8 -*-
"""
Module scenario.py
===========================================

Module for sensitivity analysis in *PyDASA*.

This module provides the DimSensitivity class for performing sensitivity analysis on dimensional coefficients derived from dimensional analysis.

Classes:

    **DimSensitivity**: Performs sensitivity analysis on dimensional coefficients in *PyDASA*.

*IMPORTANT:* Based on the theory from:
    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
import re

# Third-party modules
import numpy as np
# import sympy as sp
from sympy import diff, lambdify    # , symbols
from SALib.sample.fast_sampler import sample
from SALib.analyze.fast import analyze

# Import validation base classes
from pydasa.core.basic import Validation

# Import related classes
from pydasa.buckingham.vashchy import Coefficient
# from pydasa.core.parameter import Variable

# Import utils
from pydasa.utils.latex import parse_latex
from pydasa.utils.latex import create_latex_mapping
from pydasa.utils.latex import latex_to_python

# Import configuration
# Import the 'cfg' module to allow global variable editing
from pydasa.core import config as cfg


@dataclass
class DimSensitivity(Validation):
    # FIXME clean code, some vars and types are inconsistent
    """**DimSensitivity** class for analyzing variable impacts in *PyDASA*.

    Performs sensitivity analysis on dimensionless coefficients to determine which variables have the most significant impact on the system behavior.

    Args:
        Validation: Base class for validation of symbols and frameworks.

    Attributes:
        # Identification and Classification
        name (str): User-friendly name of the sensitivity analysis.
        description (str): Brief summary of the sensitivity analysis.
        _idx (int): Index/precedence of the sensitivity analysis.
        _sym (str): Symbol representation (LaTeX or alphanumeric).
        _alias (str): Python-compatible alias for use in code.
        _fwk (str): Framework context (PHYSICAL, COMPUTATION, SOFTWARE, CUSTOM).
        _cat (str): Category of analysis (SYM, NUM).

        # Expression Management
        _pi_expr (str): LaTeX expression to analyze.
        _sym_func (Callable): Sympy function of the sensitivity.
        _exe_func (Callable): Executable function for numerical evaluation.
        _variables (Dict[str, Variable]): Variable symbols in the expression.
        _symbols (Dict[str, Any]): Python symbols for the variables.
        _aliases (Dict[str, Any]): Variable aliases for use in code.

        # Analysis Configuration
        var_bounds (List[List[float]]): Min/max bounds for each variable.
        var_values (Dict[str, float]): Values for symbolic analysis.
        var_ranges (np.ndarray): Sample value range for numerical analysis.
        n_samples (int): Number of samples for analysis.

        # Results
        results (Dict[str, Any]): Analysis results.
    """

    # Category attribute
    # :attr: _cat
    _cat: str = "SYM"
    """Category of sensitivity analysis (SYM, NUM)."""

    # Expression properties
    # :attr: _pi_expr
    _pi_expr: Optional[str] = None
    """LaTeX expression to analyze."""

    # :attr: _sym_func
    _sym_func: Optional[Callable] = None
    """Sympy function of the sensitivity."""

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

    # Analysis configuration
    # :attr: var_bounds
    var_bounds: List[List[float]] = field(default_factory=list)
    """Min/max bounds for each variable."""

    # :attr: var_values
    var_values: Dict[str, float] = field(default_factory=dict)
    """Values for symbolic analysis."""

    # :attr: var_domains
    var_domains: Optional[np.ndarray] = None
    """Sample domain (inputs) for numerical analysis."""

    # :attr: var_ranges
    var_ranges: Optional[np.ndarray] = None
    """Sample value range (results) for numerical analysis."""

    # :attr: n_samples
    n_samples: int = 1000
    """Number of samples for analysis."""

    # Results
    # :attr: results
    results: Dict[str, Any] = field(default_factory=dict)
    """Analysis results."""

    def __post_init__(self) -> None:
        """*__post_init__()* Initializes the sensitivity analysis. Validates basic properties, sets default values, and processes the expression if provided.
        """
        # Initialize from base class
        super().__post_init__()

        # Set default symbol if not specified
        if not self._sym:
            self._sym = f"SANSYS_\\Pi_{{{self._idx}}}" if self._idx >= 0 else "SANSYS_\\Pi_{}"
        # Set default Python alias if not specified
        if not self._alias:
            self._alias = latex_to_python(self._sym)

        # Set name and description if not already set
        if not self.name:
            self.name = f"{self._sym} Sensitivity"
        if not self.description:
            self.description = f"Sensitivity analysis for {self._sym}"

        if self._pi_expr:
            # Parse the expression
            self._parse_expression(self._pi_expr)

    def _validate_analysis_ready(self) -> None:
        """*_validate_analysis_ready()* Checks if the analysis can be performed.

        Raises:
            ValueError: If the variables are missing.
            ValueError: If the python-compatible variables are missing.
            ValueError: If the symbolic expression is missing.
        """
        if not self._variables:
            raise ValueError("No variables found in the expression.")
        if not self._aliases:
            raise ValueError("No Python aliases found for variables.")
        if not self._sym_func:
            raise ValueError("No expression has been defined for analysis.")

    def set_coefficient(self, coef: Coefficient) -> None:
        """*set_coefficient()* Configure analysis from a coefficient.

        Args:
            coef (Coefficient): Dimensionless coefficient to analyze.

        Raises:
            ValueError: If the coefficient doesn't have a valid expression.
        """
        if not coef.pi_expr:
            raise ValueError("Coefficient does not have a valid expression.")

        # Set expression
        self._pi_expr = coef.pi_expr
        # parse coefficient expresion
        if coef._pi_expr:
            self._parse_expression(self._pi_expr)

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
            # self._variables = sorted(self._aliases.keys())
            fsyms = self._sym_func.free_symbols
            self._variables = sorted([str(s) for s in fsyms])

            # """
            # # OLD code, first version, keep for reference!!!
            # self.results = {
            #     var: lambdify(self._variables, diff(self._sym_fun, var), "numpy")(
            #         *[vals[v] for v in self.variables]
            #     )
            #     for var in self._variables
            # }
            # """
        except Exception as e:
            _msg = f"Failed to parse expression: {str(e)}"
            raise ValueError(_msg)

    def analyze_symbolically(self,
                             vals: Dict[str, float]) -> Dict[str, float]:
        """*analyze_symbolically()* Perform symbolic sensitivity analysis.

        Args:
            vals (Dict[str, float]): Dictionary mapping variable names to values.

        Returns:
            Dict[str, float]: Sensitivity results for each variable.
        """
        # # parse the coefficient expression
        # self._parse_expression(self._pi_expr)

        # save variable values for the analysis
        self.var_values = vals

        # Check that all required variables are provided
        var_lt = [str(v) for v in self._latex_to_py]
        missing_vars = set(var_lt) - set(list(vals.keys()))
        if missing_vars:
            _msg = f"Missing values for variables: {missing_vars}"
            raise ValueError(_msg)

        # Validate analysis readiness
        self._validate_analysis_ready()

        # trying symbolic coefficient sensitivity analysis
        try:
            py_to_latex = self._py_to_latex
            results = dict()
            functions = dict()
            if self._variables:
                for var in self._variables:
                    # Create lambdify function using Python symbols
                    expr = diff(self._sym_func, var)
                    aliases = [self._aliases[v] for v in self._variables]
                    # self._exe_func = lambdify(aliases, expr, "numpy")
                    func = lambdify(aliases, expr, "numpy")
                    functions[py_to_latex[var]] = func

                    # Convert back to LaTeX variables for result keys
                    val_args = [vals[py_to_latex[v]] for v in self._variables]
                    res = functions[py_to_latex[var]](*val_args)
                    results[py_to_latex[var]] = res

            self._exe_func = functions
            self.results = results
            return self.results

        except Exception as e:
            _msg = f"Error calculating sensitivity for {var}: {str(e)}"
            raise ValueError(_msg)

    def analyze_numerically(self,
                            vals: List[str],
                            bounds: List[List[float]],
                            n_samples: int = 1000) -> Dict[str, Any]:
        """*analyze_numerically()* Perform numerical sensitivity analysis.

        Args:
            vals (List[str]): List of variable names to analyze.
            bounds (List[List[float]]): Bounds for each variable [min, max].
            n_samples (int, optional): Number of samples to use. Defaults to 1000.

        Returns:
            Dict[str, Any]: Detailed sensitivity analysis results.
        """
        # # parse the coefficient expression
        # self._parse_expression(self._pi_expr)

        # Validate analysis readiness
        self._validate_analysis_ready()

        # trying numeric coefficient sensitivity analysis
        try:
            # Validate bounds length matches number of variables
            if len(bounds) != len(self._variables):
                _msg = f"Expected {len(self._variables)} "
                _msg += f"bounds (one per variable), got {len(bounds)}"
                raise ValueError(_msg)

            # Set number of samples
            self.n_samples = n_samples
            # Store bounds
            self.var_bounds = bounds

            results = dict()
            if self._variables:

                # Set up problem definition for SALib
                problem = {
                    "num_vars": len(vals),
                    "names": self.variables,
                    "bounds": bounds,
                }

                # Generate samples (domain)
                self.var_domains = sample(problem, n_samples)
                _len = len(self._variables)
                self.var_domains = self.var_domains.reshape(-1, _len)

                # Create lambdify function using Python symbols
                aliases = [self._aliases[v] for v in self._variables]
                self._exe_func = lambdify(aliases, self._sym_func, "numpy")

                # Evaluate function at sample points
                Y = np.apply_along_axis(lambda v: self._exe_func(*v),
                                        1, self.var_domains)
                self.var_ranges = Y.reshape(-1, 1)

                # Perform FAST (Fourier Amplitude Sensitivity Test) analysis
                results = analyze(problem, Y)

                # Convert back to LaTeX variables for result keys
                if "names" in results:
                    py_to_latex = self._py_to_latex
                    results["names"] = [py_to_latex.get(v, v) for v in results["names"]]

            self.results = results
            return self.results

        except Exception as e:
            _msg = f"Error calculating sensitivity for {problem}: {str(e)}"
            raise ValueError(_msg)

    # Property getters and setters

    @property
    def cat(self) -> str:
        """*cat* Get the analysis category.

        Returns:
            str: Category (SYM, NUM).
        """
        return self._cat

    @cat.setter
    def cat(self, val: str) -> None:
        """*cat* Set the analysis category.

        Args:
            val (str): Category value.

        Raises:
            ValueError: If category is invalid.
        """
        if val.upper() not in cfg.SENS_ANSYS_DT:
            raise ValueError(
                f"Invalid category: {val}. "
                f"Must be one of: {', '.join(cfg.SENS_ANSYS_DT.keys())}"
            )
        self._cat = val.upper()

    @property
    def pi_expr(self) -> Optional[str]:
        """*pi_expr* Get the expression to analyze.

        Returns:
            Optional[str]: LaTeX expression.
        """
        return self._pi_expr

    @pi_expr.setter
    def pi_expr(self, val: str) -> None:
        """*pi_expr* Set the expression to analyze.

        Args:
            val (str): LaTeX expression.

        Raises:
            ValueError: If expression is invalid.
        """
        # Regular expression to match valid LaTeX strings or alphanumeric strings
        if not (val.isalnum() or re.match(cfg.LATEX_RE, val)):
            _msg = "LaTeX expression must be a valid string. "
            _msg += f"Provided: '{val}' "
            _msg += "Examples: 'V', 'd', '\\Pi_{0}', '\\rho'."
            raise ValueError(_msg)
        # Update expression
        self._pi_expr = val

        # Parse expression
        self._parse_expression(self._pi_expr)

    @property
    def sym_func(self) -> Optional[Callable]:
        """*sym_func* Get the symbolic function.

        Returns:
            Optional[Callable]: Symbolic expression.
        """
        return self._sym_func

    @sym_func.setter
    def sym_func(self, val: Callable) -> None:
        """*sym_func* Set the symbolic function.

        Args:
            val (Callable): Symbolic function.

        Raises:
            ValueError: If function is not callable.
        """
        if not callable(val):
            _msg = "Sympy function must be callable. "
            _msg += f"Provided: {type(val)}"
            raise ValueError(_msg)
        self._sym_func = val

    @property
    def exe_func(self) -> Optional[Callable]:
        """*exe_func* Get the executable function.

        Returns:
            Optional[Callable]: Executable function for numerical evaluation.
        """
        return self._exe_func

    @property
    def variables(self) -> Dict[str, Any]:
        """*variables* Get the variables in the expression.

        Returns:
            Dict[str, Any]:: Variable symbols.
        """
        return self._variables

    @property
    def symbols(self) -> Dict[str, Any]:
        """*symbols* Get the Python symbols for the variables.

        Returns:
            Dict[str, Any]: Dictionary mapping variable names to sympy symbols.
        """
        return self._symbols

    @property
    def aliases(self) -> Dict[str, Any]:
        """*aliases* Get the Python aliases for the variables.

        Returns:
            Dict[str, Any]:: Python-compatible variable names.
        """
        return self._aliases

    def clear(self) -> None:
        """*clear()* Reset all attributes to default values. Resets all sensitivity analysis properties to their initial state.
        """
        # Reset base class attributes
        self._idx = -1
        self._sym = "SANSYS_\\Pi_{}"
        self._alias = ""
        self._fwk = "PHYSICAL"
        self.name = ""
        self.description = ""

        # Reset sensitivity-specific attributes
        self._cat = "SYM"
        self._pi_expr = None
        self._sym_func = None
        self._exe_func = None
        self._variables = []
        self._symbols = {}
        self._aliases = {}
        self.var_bounds = []
        self.var_values = {}
        self.var_domains = None
        self.var_ranges = None
        self.n_samples = 1000
        self.results = {}

    def to_dict(self) -> Dict[str, Any]:
        """*to_dict()* Convert sensitivity analysis to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation of sensitivity analysis.
        """
        return {
            "name": self.name,
            "description": self.description,
            "idx": self._idx,
            "sym": self._sym,
            "alias": self._alias,
            "fwk": self._fwk,
            "cat": self._cat,
            "pi_expr": self._pi_expr,
            "variables": self._variables,
            "symbols": self._symbols,
            "aliases": self._aliases,
            "var_bounds": self.var_bounds,
            "var_values": self.var_values,
            "var_domains": self.var_domains,
            "var_ranges": self.var_ranges,
            "n_samples": self.n_samples,
            "results": self.results
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DimSensitivity:
        """*from_dict()* Create sensitivity analysis from dictionary representation.

        Args:
            data (Dict[str, Any]): Dictionary representation of sensitivity analysis.

        Returns:
            DimSensitivity: New sensitivity analysis instance.
        """
        # Create basic instance
        instance = cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            _idx=data.get("idx", -1),
            _sym=data.get("sym", ""),
            _cat=data.get("cat", "SYM"),
            _fwk=data.get("fwk", "PHYSICAL"),
            _alias=data.get("alias", ""),
            _pi_expr=data.get("pi_expr", None),
            _variables=data.get("variables", {}),
            _symbols=data.get("symbols", {}),
            _aliases=data.get("aliases", {}),
            var_bounds=data.get("var_bounds", []),
            var_values=data.get("var_values", {}),
            var_domains=data.get("var_domains", None),
            var_ranges=data.get("var_ranges", None),
            n_samples=data.get("n_samples", 1000),
            # results=data.get("results", {})
        )

        # Set additional properties if available
        if "results" in data:
            instance.results = data["results"]
        return instance
