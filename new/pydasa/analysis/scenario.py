# -*- coding: utf-8 -*-
"""
Module for **DimSensitivity** analysis in *PyDASA*.

This module provides the DimSensitivity class for performing sensitivity analysis on dimensional coefficients derived from dimensional analysis.

*IMPORTANT:* Based on the theory from:
    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Generic, Callable
import re

# Third-party modules
import numpy as np
# import sympy as sp
from sympy import diff, lambdify    # , symbols
from SALib.sample.fast_sampler import sample
from SALib.analyze.fast import analyze

# Import validation base classes
from new.pydasa.core.basic import Validation

# Import related classes
from new.pydasa.buckingham.vashchy import Coefficient

# Import utils
from new.pydasa.utils.default import T
from new.pydasa.utils.latex import parse_latex
from new.pydasa.utils.latex import create_latex_mapping
from new.pydasa.utils.latex import latex_to_python

# Import configuration
# Import the 'cfg' module to allow global variable editing
from new.pydasa.utils import config as cfg


@dataclass
class DimSensitivity(Validation, Generic[T]):
    """**DimSensitivity** class for analyzing variable impacts in *PyDASA*. Performs sensitivity analysis on dimensionless coefficients to determine which variables have the most significant impact on the system behavior.

    Attributes:
        # Identification and Classification
        name (str): User-friendly name of the sensitivity analysis.
        description (str): Brief summary of the sensitivity analysis.
        _idx (int): Index/precedence of the sensitivity analysis.
        _sym (str): Symbol representation (LaTeX or alphanumeric).
        _pyalias (str): Python-compatible alias for use in code.
        _fwk (str): Framework context (PHYSICAL, COMPUTATION, SOFTWARE, CUSTOM).
        _cat (str): Category of analysis (SYM, NUM).

        # Expression Management
        _pi_expr (str): LaTeX expression to analyze.
        _sym_func (Callable): Sympy function of the sensitivity.
        _exe_func (Callable): Executable function for numerical evaluation.
        _variables (Dict[str]): Variable symbols in the expression.
        _symbols (Dict[str, Any]): Python symbols for the variables.
        _pyaliases (Dict[str, Any]): Variable aliases for use in code.

        # Analysis Configuration
        var_bounds (List[List[float]]): Min/max bounds for each variable.
        var_values (Dict[str, float]): Values for symbolic analysis.
        var_range (np.ndarray): Sample value range for numerical analysis.
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

    # :attr: _pyaliases
    _pyaliases: Dict[str: Any] = field(default_factory=dict)
    """Variable aliases for use in code."""

    # Analysis configuration
    # :attr: var_bounds
    var_bounds: List[List[float]] = field(default_factory=list)
    """Min/max bounds for each variable."""

    # :attr: var_values
    var_values: Dict[str, float] = field(default_factory=dict)
    """Values for symbolic analysis."""

    # :attr: var_range
    var_range: Optional[np.ndarray] = None
    """Sample values for numerical analysis."""

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
        if not self._pyalias:
            self._pyalias = latex_to_python(self._sym)
        # Process expression if provided
        # if self._pi_expr:
        #     self._parse_expression(self._pi_expr)

    # def _parse_expression(self, expr: str) -> None:
    # FIXME old code, not used anymore, remove after tests
    #     """*_parse_expression()* Parse the LaTeX expression into a sympy function.

    #     Args:
    #         expr (str): LaTeX expression to parse.

    #     Raises:
    #         ValueError: If the expression cannot be parsed.
    #     """
    #     try:
    #         # Parse the expression
    #         answer = parse_latex_symbols(expr)
    #         self._sym_func, self._exe_func, self._variables = answer
    #         print(self._variables)
    #         print(self._sym_func)
    #         print(self._exe_func)
    #         # self._exe_func = lambdify(self._variables, self._sym_func, "numpy")
    #         # Use list of symbols in defined order
    #         sym_list = [symbols(v) for v in self._variables]
    #         # Ensure we create a numerical function that returns floats, not symbolic expressions
    #         self._exe_func = lambdify(sym_list, self._sym_func, "numpy")

    #     except Exception as e:
    #         _msg = f"Failed to parse expression: {str(e)}"
    #         raise ValueError(_msg)

    def _parse_expression(self, expr: str) -> None:
        """*_parse_expression()* Parse the LaTeX expression into a sympy function.

        Args:
            expr (str): LaTeX expression to parse.

        Raises:
            ValueError: If the expression cannot be parsed.
        """
        try:
            print(expr)
            self._sym_func = parse_latex(expr)
            print(f"Original parsed: {self._sym_func}")
            print(f"LaTeX symbols: {self._sym_func.free_symbols}")

            self._symbols, self._pyaliases = create_latex_mapping(expr)
            print(f"Symbol mapping: {self._symbols}")
            print(f"Python symbols: {self._pyaliases}")

            # Substitute LaTeX symbols with Python symbols
            for latex_sym, py_sym in self._symbols.items():
                self._sym_func = self._sym_func.subs(latex_sym, py_sym)
            print(f"After sustitution: {self._sym_func}")
            # OLD
            # # Use the parse_latex_symbols function
            # answer = parse_latex_symbols(expr)
            # self._sym_func, self._exe_func, self._variables = answer

            # Get Python variable names
            self._variables = sorted([str(s)
                                     for s in self._sym_func.free_symbols])
            print(f"Python variables: {self._variables}")

            # Verify if we have sympy stuff
            # TODO improve check later
            if self._variables is None:
                raise ValueError("Parsed variables is not callable")

            # # Verify that we have a usable executable function
            # if not callable(self._exe_func):
            #     raise ValueError("Failed to create executable function")

        except Exception as e:
            _msg = f"Failed to parse expression: {str(e)}"
            raise ValueError(_msg)

    def _validate_analysis_ready(self) -> None:
        """*_validate_analysis_ready()* Checks if the analysis can be performed.

        Raises:
            ValueError: If required components are missing.
        """
        if not self._variables:
            raise ValueError("No variables found in the expression.")
        if not self._pyaliases:
            raise ValueError("No Python aliases found for variables.")
        if not self._sym_func:
            raise ValueError("No expression has been defined for analysis.")
        # if not self._exe_func:
        #     raise ValueError("Executable function has not been created.")

    # def set_coefficient(self, coef: Coefficient) -> None:
    # TODO possible i dont need it, delete later
    #     """*set_coefficient()* Configure analysis from a coefficient.

    #     Args:
    #         coef (Coefficient): Dimensionless coefficient to analyze.

    #     Raises:
    #         ValueError: If the coefficient doesn't have a valid expression.
    #     """
    #     if not coef.pi_expr:
    #         raise ValueError("Coefficient does not have a valid expression.")

    #     # Set expression
    #     self._pi_expr = coef.pi_expr

    #     # Copy over the Python alias if available
    #     if coef.pyalias:
    #         self._pyalias = coef.pyalias

    #     # Parse expression
    #     self._parse_expression(self._pi_expr)

    #     # Set name and description if not already set
    #     if not self.name:
    #         self.name = f"Sensitivity of {coef.name}"

    #     if not self.description:
    #         self.description = f"Sensitivity analysis for {coef.name}"

    def analyze_symbolically(self,
                             vals: Dict[str, float]) -> Dict[str, float]:
        """*analyze_symbolically()* Perform symbolic sensitivity analysis.

        Args:
            vals (Dict[str, float]): Dictionary mapping variable names to values.

        Returns:
            Dict[str, float]: Sensitivity results for each variable.
        """
        # OLD CODE, TO DESTROY LATER!!!
        # # Validate analysis readiness
        # self._validate_analysis_ready()

        # # Check that all required variables are provided
        # print("Variables to analyze:", self._variables)
        # print("Symbols:", self._symbols)
        # print("Python Aliases:", self._pyaliases)
        # var_lt = [str(v) for v in self._variables]
        # missing_vars = set(var_lt) - set(list(vals.keys()))
        # if missing_vars:
        #     raise ValueError(f"Missing values for variables: {missing_vars}")

        self._sym_func = parse_latex(self._pi_expr)
        print(f"Original parsed: {self._sym_func}")
        print(f"LaTeX symbols: {self._sym_func.free_symbols}")

        # Create symbol mapping
        self._symbols, self._pyaliases = create_latex_mapping(self._pi_expr)
        print(f"Symbol mapping: {self._symbols}")
        print(f"Python symbols: {self._pyaliases}")

        # Substitute LaTeX symbols with Python symbols
        for latex_sym, py_sym in self._symbols.items():
            self._sym_func = self._sym_func.subs(latex_sym, py_sym)

        """
        # OLD code, first version, keep for reference!!!
        self.results = {
            var: lambdify(self._variables, diff(self._sym_fun, var), "numpy")(
                *[vals[v] for v in self.variables]
            )
            for var in self._variables
        }
        """

        # Get Python variable names
        self._variables = sorted([str(s) for s in self._sym_func.free_symbols])
        print(f"Python variables: {self._variables}")
        try:
            results = dict()
            if self._variables:
                for var in self._variables:
                    # Create lambdify function using Python symbols
                    expr = diff(self._sym_func, var)
                    aliases = [self._pyaliases[v] for v in self._variables]
                    self._exe_func = lambdify(aliases, expr, "numpy")
                    val_args = [vals[v] for v in self._variables]
                    res = self._exe_func(*val_args)
                    print(f"With {dict(zip(self._variables, val_args))} => {res}")
                    results[var] = res
            self.results = results
            return self.results

        except Exception as e:
            raise ValueError(
                f"Error calculating sensitivity for {var}: {str(e)}")

    def analyze_numerically(self,
                            bounds: List[List[float]],
                            n_samples: int = 1000) -> Dict[str, Any]:
        """*analyze_numerically()* Perform numerical sensitivity analysis.

        Args:
            bounds (List[List[float]]): Bounds for each variable [min, max].
            n_samples (int, optional): Number of samples to use. Defaults to 1000.

        Returns:
            Dict[str, Any]: Detailed sensitivity analysis results.
        """
        # # Validate analysis readiness
        # self._validate_analysis_ready()

        self._sym_func = parse_latex(self._pi_expr)
        print(f"Original parsed: {self._sym_func}")
        print(f"LaTeX symbols: {self._sym_func.free_symbols}")

        # Create symbol mapping
        self._symbols, self._pyaliases = create_latex_mapping(self._pi_expr)
        print(f"Symbol mapping: {self._symbols}")
        print(f"Python symbols: {self._pyaliases}")

        # Substitute LaTeX symbols with Python symbols
        for latex_sym, py_sym in self._symbols.items():
            self._sym_func = self._sym_func.subs(latex_sym, py_sym)

        # Get Python variable names
        self._variables = sorted([str(s) for s in self._sym_func.free_symbols])
        print(f"Python variables: {self._variables}")

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
                "num_vars": len(self.variables),
                "names": self.variables,
                "bounds": bounds,
            }

            # Generate samples
            self.var_range = sample(problem, n_samples)
            self.var_range = self.var_range.reshape(-1, len(self._variables))

            # Create lambdify function using Python symbols
            aliases = [self._pyaliases[v] for v in self._variables]
            self._exe_func = lambdify(aliases, self._sym_func, "numpy")

            # Evaluate function at sample points
            Y = np.apply_along_axis(lambda v: self._exe_func(*v), 1, self.var_range)

            # Perform FAST analysis
            results = analyze(problem, Y)

        self.results = results
        return self.results

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
            # FIXME REGEX not working!!!!
            # raise ValueError(_msg)

        # Update expression
        self._pi_expr = val

        # Parse expression
        self._parse_expression(self._pi_expr)

    @property
    def variables(self) -> List[dict[str, Any]]:
        """*variables* Get the variables in the expression.

        Returns:
            List[str]: Variable symbols.
        """
        return self._variables

    @property
    def pyaliases(self) -> List[str]:
        """*pyaliases* Get the Python aliases for the variables.

        Returns:
            List[str]: Python-compatible variable names.
        """
        return self._pyaliases

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

    def clear(self) -> None:
        """*clear()* Reset all attributes to default values. Resets all sensitivity analysis properties to their initial state.
        """
        # Reset base class attributes
        self._idx = -1
        self._sym = "SANSYS_\\Pi_{}"
        self._pyalias = ""
        self._fwk = "PHYSICAL"
        self.name = ""
        self.description = ""

        # Reset sensitivity-specific attributes
        self._cat = "SYM"
        self._pi_expr = None
        self._sym_func = None
        self._exe_func = None
        self._variables = []
        self.var_bounds = []
        self.var_values = {}
        self.var_range = None
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
            "pyalias": self._pyalias,
            "fwk": self._fwk,
            "cat": self._cat,
            "pi_expr": self._pi_expr,
            "variables": self._variables,
            "var_bounds": self.var_bounds,
            "var_values": self.var_values,
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
            _pyalias=data.get("pyalias", ""),
            _fwk=data.get("fwk", "PHYSICAL"),
            _cat=data.get("cat", "SYM"),
            _pi_expr=data.get("pi_expr", None),
            n_samples=data.get("n_samples", 1000)
        )

        # Set additional properties if available
        if "results" in data:
            instance.results = data["results"]
        return instance
