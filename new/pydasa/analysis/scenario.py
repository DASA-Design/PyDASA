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
from sympy import diff, lambdify
from SALib.sample.fast_sampler import sample
from SALib.analyze.fast import analyze

# Import validation base classes
from new.pydasa.core.basic import Validation

# Import related classes
from new.pydasa.buckingham.vashchy import Coefficient

# Import utils
from new.pydasa.utils.default import T
from new.pydasa.utils.latex import parse_latex_symbols
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
        _fwk (str): Framework context (PHYSICAL, COMPUTATION, DIGITAL, CUSTOM).
        _cat (str): Category of analysis (SYM, Num, HYB).

        # Expression Management
        _pi_expr (str): LaTeX expression to analyze.
        _sym_fun (Callable): Sympy function of the sensitivity.
        _exe_fun (Callable): Executable function for numerical evaluation.
        _variables (List[str]): Variable symbols in the expression.

        # Analysis Configuration
        _var_bounds (List[List[float]]): Min/max bounds for each variable.
        _var_values (Dict[str, float]): Values for symbolic analysis.
        var_val (np.ndarray): Sample values for numerical analysis.
        _num_samples (int): Number of samples for analysis.

        # Results
        result (Dict[str, Any]): Analysis results.
    """

    # Category attribute
    # :attr: _cat
    _cat: str = "SYM"
    """Category of sensitivity analysis (SYM, Num, HYB)."""

    # Expression properties
    # :attr: _pi_expr
    _pi_expr: Optional[str] = None
    """LaTeX expression to analyze."""

    # :attr: _sym_fun
    _sym_fun: Optional[Callable] = None
    """Sympy function of the sensitivity."""

    # :attr: _exe_fun
    _exe_fun: Optional[Callable] = None
    """Executable function for numerical evaluation."""

    # :attr: _variables
    _variables: List[str] = field(default_factory=list)
    """Variable symbols in the expression."""

    # Analysis configuration
    # :attr: _var_bounds
    _var_bounds: List[List[float]] = field(default_factory=list)
    """Min/max bounds for each variable."""

    # :attr: _var_values
    _var_values: Dict[str, float] = field(default_factory=dict)
    """Values for symbolic analysis."""

    # :attr: var_val
    var_val: Optional[np.ndarray] = None
    """Sample values for numerical analysis."""

    # :attr: _num_samples
    _num_samples: int = 1000
    """Number of samples for analysis."""

    # Results
    # :attr: result
    result: Dict[str, Any] = field(default_factory=dict)
    """Analysis results."""

    def __post_init__(self) -> None:
        """*__post_init__()* Initializes the sensitivity analysis. Validates basic properties, sets default values, and processes the expression if provided.
        """
        # Initialize from base class
        super().__post_init__()

        # Set default symbol if not specified
        if not self._sym:
            self._sym = f"SEN ANSYS \\Pi_{{{self._idx}}}" if self._idx >= 0 else "SEN ANSYS \\Pi_{}"

        # Process expression if provided
        if self._pi_expr:
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
            self._sym_fun, self._variables = parse_latex_symbols(expr)

            # Extract variable names and sort them
            self._variables = sorted([str(v) for v in self._sym_fun.free_symbols])

            # Create executable function
            self._exe_fun = lambdify(self._variables, self._sym_fun, "numpy")
        except Exception as e:
            _msg = f"Failed to parse expression: {str(e)}"
            raise ValueError(_msg)

    def _validate_analysis_ready(self) -> None:
        """*_validate_analysis_ready()* Checks if the analysis can be performed.

        Raises:
            ValueError: If required components are missing.
        """
        if not self._sym_fun:
            raise ValueError("No expression has been defined for analysis.")
        if not self._variables:
            raise ValueError("No variables found in the expression.")
        if not self._exe_fun:
            raise ValueError("Executable function has not been created.")

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

        # Parse expression
        self._parse_expression(self._pi_expr)

        # Set name and description if not already set
        if not self.name:
            self.name = f"Sensitivity of {coef.name}"

        if not self.description:
            self.description = f"Sensitivity analysis for {coef.name}"

    def analyze_symbolically(self, values: Dict[str, float]) -> Dict[str, float]:
        """*analyze_symbolically()* Perform symbolic sensitivity analysis.

        Args:
            values (Dict[str, float]): Dictionary mapping variable names to values.

        Returns:
            Dict[str, float]: Sensitivity results for each variable.

        Raises:
            ValueError: If required values are missing.
        """
        # Validate analysis readiness
        self._validate_analysis_ready()

        # Store variable values
        self._var_values = values

        # Compute partial derivatives
        self.result = {
            var: lambdify(self.variables, diff(self._sym_fun, var), "numpy")(
                *[values[v] for v in self.variables]
            )
            for var in self.variables
        }

        return self.result

    def analyze_numerically(self,
                            bounds: List[List[float]],
                            num_samples: int = 1000) -> Dict[str, Any]:
        """*analyze_numerically()* Perform numerical sensitivity analysis.

        Args:
            bounds (List[List[float]]): Bounds for each variable [min, max].
            num_samples (int, optional): Number of samples to use. Defaults to 1000.

        Returns:
            Dict[str, Any]: Detailed sensitivity analysis results.

        Raises:
            ValueError: If bounds are invalid.
        """
        # Validate analysis readiness
        self._validate_analysis_ready()

        # Set number of samples
        self._num_samples = num_samples

        # Store bounds
        self._var_bounds = bounds

        # Set up problem definition for SALib
        problem = {
            "num_vars": len(self.variables),
            "names": self.variables,
            "bounds": bounds,
        }

        # Generate samples
        self.var_val = sample(problem,
                              num_samples).reshape(-1, len(self.variables))

        # Evaluate function at sample points
        Y = np.apply_along_axis(lambda v: self._exe_fun(*v), 1, self.var_val)

        # Perform FAST analysis
        self.result = analyze(problem, Y)

        return self.result

    # Property getters and setters

    @property
    def cat(self) -> str:
        """*cat* Get the analysis category.

        Returns:
            str: Category (SYM, Num, HYB).
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
    def variables(self) -> List[str]:
        """*variables* Get the variables in the expression.

        Returns:
            List[str]: Variable symbols.
        """
        return self._variables

    @property
    def sym_fun(self) -> Optional[Callable]:
        """*sym_fun* Get the symbolic function.

        Returns:
            Optional[Callable]: Symbolic expression.
        """
        return self._sym_fun

    @sym_fun.setter
    def sym_fun(self, val: Callable) -> None:
        """*sym_fun* Set the symbolic function.

        Args:
            val (Callable): Symbolic function.

        Raises:
            ValueError: If function is not callable.
        """
        if not callable(val):
            _msg = "Sympy function must be callable. "
            _msg += f"Provided: {type(val)}"
            raise ValueError(_msg)
        self._sym_fun = val

    @property
    def num_samples(self) -> int:
        """*num_samples* Get the number of samples.

        Returns:
            int: Number of samples.
        """
        return self._num_samples

    @num_samples.setter
    def num_samples(self, val: int) -> None:
        """*num_samples* Set the number of samples.

        Args:
            val (int): Number of samples.

        Raises:
            ValueError: If value is invalid.
        """
        if not isinstance(val, int) or val < -1:
            raise ValueError("Number of samples must be a positive integer.")
        self._num_samples = val

    def clear(self) -> None:
        """*clear()* Reset all attributes to default values. Resets all sensitivity analysis properties to their initial state.
        """
        # Reset base class attributes
        self._idx = -1
        self._sym = "SEN ANSYS \\Pi_{}"
        self._fwk = "PHYSICAL"
        self.name = ""
        self.description = ""

        # Reset sensitivity-specific attributes
        self._cat = "SYM"
        self._pi_expr = None
        self._sym_fun = None
        self._exe_fun = None
        self._variables = []
        self._var_bounds = []
        self._var_values = {}
        self.var_val = None
        self._num_samples = 1000
        self.result = {}

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
            "fwk": self._fwk,
            "cat": self._cat,
            "pi_expr": self._pi_expr,
            "variables": self._variables,
            "var_bounds": self._var_bounds,
            "var_values": self._var_values,
            "num_samples": self._num_samples,
            "result": self.result
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
            _fwk=data.get("fwk", "PHYSICAL"),
            _cat=data.get("cat", "SYM"),
            _pi_expr=data.get("pi_expr", None),
            _num_samples=data.get("num_samples", 1000)
        )

        # Set additional properties if available
        if "result" in data:
            instance.result = data["result"]

        return instance
