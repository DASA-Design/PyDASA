# -*- coding: utf-8 -*-
"""
Module for representing Dimensionless Coefficients in Dimensional Analysis for *PyDASA*.

This module provides the Coefficient class which models dimensionless numbers
used in Vaschy-Buckingham's π-theorem for dimensional analysis.

*IMPORTANT:* Based on the theory from:
    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""

# native python modules
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Generic
# import re

# Third-party modules
import numpy as np

# Import validation base classes
from new.pydasa.core.basic import Validation

# Import utils
from new.pydasa.utils.default import T
from new.pydasa.utils.error import inspect_name as _insp_var
# import the 'cfg' module to allow global variable edition
from new.pydasa.utils import config as cfg


@dataclass
class Coefficient(Validation, Generic[T]):
    """**Coefficient** class for Dimensional Analysis in *PyDASA*.

    A comprehensive implementation that represents dimensionless coefficients
    (π numbers) used in the Vaschy-Buckingham π-theorem method.

    Attributes:
        # Identification and Classification
        name (str): User-friendly name of the coefficient.
        description (str): Brief summary of the coefficient.
        _idx (int): Index/precedence in the dimensional matrix.
        _sym (str): Symbol representation (LaTeX or alphanumeric).
        _fwk (str): Framework context (PHYSICAL, COMPUTATION, DIGITAL, CUSTOM).
        _cat (str): Category (COMPUTED, DERIVED).
        relevance (bool): Flag indicating if coefficient is relevant for analysis.

        # Coefficient Construction
        _param_lt (List[str]): Parameter symbols used in coefficient.
        _dim_col (List[int]): Dimensional column for matrix operations.
        _pivot_lt (List[int]): Pivot indices in dimensional matrix.
        _pi_expr (str): Symbolic expression of coefficient.
        par_dims (Dict[str, int]): Dimensional parameter exponents.

        # Value Ranges
        _min (float): Minimum value of the coefficient.
        _max (float): Maximum value of the coefficient.
        _avg (float): Average value of the coefficient.
        _step (float): Step size for simulations.
        _data (np.ndarray): Array of coefficient values for analysis.
    """

    # Category attribute (COMPUTED, DERIVED)
    # :attr: _cat
    _cat: str = "COMPUTED"
    """Category of the coefficient (COMPUTED, DERIVED)."""

    # Coefficient construction properties
    # :attr: _param_lt
    _param_lt: List[str] = field(default_factory=list)
    """Parameter symbols used in the coefficient."""

    # :attr: _dim_col
    _dim_col: List[int] = field(default_factory=list)
    """Dimensional column for matrix operations."""

    # :attr: _pivot_lt
    _pivot_lt: Optional[List[int]] = None
    """Pivot indices in dimensional matrix."""

    # :attr: _pi_expr
    _pi_expr: Optional[str] = None
    """Symbolic expression of coefficient."""

    # :attr: par_dims
    par_dims: Optional[Dict[str, int]] = None
    """Dimensional parameter exponents in coefficient."""

    # Value ranges
    # :attr: _min
    _min: Optional[float] = None
    """Minimum value of coefficient."""

    # :attr: _max
    _max: Optional[float] = None
    """Maximum value of coefficient."""

    # :attr: _avg
    _avg: Optional[float] = None
    """Average value of coefficient."""

    # :attr: _step
    _step: float = 1e-3
    """Step size for simulations."""

    # :attr: _data
    _data: np.ndarray = field(default_factory=lambda: np.array([]))
    """Array of coefficient values for analysis."""

    # Flags
    # :attr: relevance
    relevance: bool = True
    """Flag indicating if coefficient is relevant for analysis."""

    def __post_init__(self) -> None:
        """*__post_init__()* Initializes the coefficient and validates its properties.

        Performs validation of core properties and builds the coefficient expression
        based on parameter symbols and their respective dimensional exponents.

        Raises:
            ValueError: If parameter list and dimensional column have different lengths.
        """
        # Initialize from base class
        super().__post_init__()

        # Set the Pi symbol if not specified
        if not self._sym:
            if self._idx >= 0:
                self._sym = f"\\Pi_{{{self._idx}}}"
            else:
                self._sym = "\\Pi_{}"

        self.cat = self._cat
        self.param_lt = self._param_lt
        self.dim_col = self._dim_col
        self.pivot_lt = self._pivot_lt
        self.pi_expr, self.par_dims = self._build_expression(self._param_lt,
                                                             self._dim_col)

        # # Build expression if parameters and dimensions are provided
        # if self._param_lt and self._dim_col:
        #     self.pi_expr, self.par_dims = self._build_expression(self._param_lt, self._dim_col)

        # Set up data array if all required values are provided
        if all([self._min, self._max, self._step]):
            self._data = np.arange(self._min, self._max, self._step)

    def _validate_list(self, lt: List, exp_type: List[type]) -> bool:
        """*_validate_list()* Validates a list with expected element types.

        Args:
            lt (List): List to validate.
            exp_type (tuple): Expected types for list elements.

        Raises:
            ValueError: If list is empty or contains invalid types.

        Returns:
            bool: True if the list is valid.
        """
        if not isinstance(lt, list):
            _msg = f"{_insp_var(lt)} must be a list. "
            _msg += f"Provided: {type(lt)}"
            raise ValueError(_msg)
        if not all(isinstance(x, exp_type) for x in lt):
            _msg = f"{_insp_var(lt)} must contain {exp_type} elements."
            _msg += f" Provided: {[type(x).__name__ for x in lt]}"
            raise ValueError(_msg)
        if len(lt) == 0:
            _msg = f"{_insp_var(lt)} cannot be empty. "
            _msg += f"Provided: {lt}"
            raise ValueError(_msg)
        return True

    def _build_expression(self,
                          param_lt: List[str],
                          dim_col: List[int]) -> tuple[str, dict]:
        """*_build_expression()* Builds LaTeX expression for coefficient.

        Args:
            param_lt (List[str]): List of parameter symbols.
            dim_col (List[int]): List of dimensional exponents.

        Raises:
            ValueError: If parameter list and dimensional column have different lengths.

        Returns:
            tuple[str, Dict[str, int]]: LaTeX expression and parameter exponents.
        """
        # Validate parameter list and dimensional column
        if len(param_lt) != len(dim_col):
            _msg = f"Parameter list len ({len(param_lt)}) and "
            _msg += f"dimensional column len ({len(dim_col)}) must be equal."
            raise ValueError(_msg)

        # Initialize working variables
        numerator = []
        denominator = []
        parameters = {}

        # Process parameters and their exponents
        for sym, exp in zip(param_lt, dim_col):
            # Add to numerator if exponent is positive
            if exp > 0:
                part = sym if exp == 1 else f"{sym}^{{{exp}}}"
                numerator.append(part)
            # Add to denominator if exponent is negative
            elif exp < 0:
                part = sym if exp == -1 else f"{sym}^{{{-exp}}}"
                denominator.append(part)
            # Skip zero exponents
            else:
                continue
            # Store parameter exponent
            parameters[sym] = exp

        # Build expression
        num_str = "1" if not numerator else "*".join(numerator)

        # Return expression based on whether denominator exists
        if not denominator:
            return num_str, parameters
        else:
            denom_str = "*".join(denominator)
            return f"\\frac{{{num_str}}}{{{denom_str}}}", parameters

    # Property getters and setters

    @property
    def cat(self) -> str:
        """*cat* Get the coefficient category.

        Returns:
            str: Category (COMPUTED, DERIVED).
        """
        return self._cat

    @cat.setter
    def cat(self, val: str) -> None:
        """*cat* Set the coefficient category.

        Args:
            val (str): Category value.

        Raises:
            ValueError: If category is invalid.
        """
        if val.upper() not in cfg.DC_CAT_DT:
            raise ValueError(
                f"Invalid category: {val}. "
                f"Must be one of: {', '.join(cfg.DC_CAT_DT.keys())}"
            )
        self._cat = val.upper()

    @property
    def param_lt(self) -> List[str]:
        """*param_lt* Get the parameter symbols list.

        Returns:
            List[str]: Parameter symbols list.
        """
        return self._param_lt

    @param_lt.setter
    def param_lt(self, val: List[str]) -> None:
        """*param_lt* Set the parameter symbols list.

        Args:
            val (List[str]): Parameter symbols list.

        Raises:
            ValueError: If list is invalid.
        """
        if self._validate_list(val, (str,)):
            self._param_lt = val
            # # Update expression if dimensional column is available
            # if self._dim_col:
            #     self._pi_expr, self.par_dims = self._build_expression(val, self._dim_col)

    @property
    def dim_col(self) -> List[int]:
        """*dim_col* Get the dimensional column.

        Returns:
            List[int]: Dimensional column.
        """
        return self._dim_col

    @dim_col.setter
    def dim_col(self, val: List[int]) -> None:
        """*dim_col* Set the dimensional column.

        Args:
            val (List[int]): Dimensional column.

        Raises:
            ValueError: If list is invalid.
        """
        if self._validate_list(val, (int, float)):
            self._dim_col = [int(x) for x in val]  # Convert all to int
            # # Update expression if parameter list is available
            # if self._param_lt:
            #     self._pi_expr, self.par_dims = self._build_expression(self._param_lt, self._dim_col)

    @property
    def pivot_lt(self) -> Optional[List[int]]:
        """*pivot_lt* Get the pivot indices list.

        Returns:
            Optional[List[int]]: Pivot indices list.
        """
        return self._pivot_lt

    @pivot_lt.setter
    def pivot_lt(self, val: List[int]) -> None:
        """*pivot_lt* Set the pivot indices list.

        Args:
            val (List[int]): Pivot indices list.

        Raises:
            ValueError: If list is invalid.
        """
        if val is None:
            self._pivot_lt = None
        elif self._validate_list(val, (int,)):
            self._pivot_lt = val

    @property
    def pi_expr(self) -> Optional[str]:
        """*pi_expr* Get the coefficient expression.

        Returns:
            Optional[str]: Coefficient expression.
        """
        return self._pi_expr

    @pi_expr.setter
    def pi_expr(self, val: str) -> None:
        """*pi_expr* Set the coefficient expression.

        Args:
            val (str): Coefficient expression.

        Raises:
            ValueError: If expression is invalid.
        """
        if not isinstance(val, str):
            raise ValueError(f"Expression must be a string, got {type(val).__name__}")
        self._pi_expr = val

    # Value range properties

    @property
    def min(self) -> Optional[float]:
        """*min* Get minimum range value.

        Returns:
            Optional[float]: Minimum range value.
        """
        return self._min

    @min.setter
    def min(self, val: Optional[float]) -> None:
        """*min* Sets minimum range value.

        Args:
            val (Optional[float]): Minimum range value.

        Raises:
            ValueError: If value is invalid or greater than max.
        """
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError("Minimum range must be a number.")
        if val > self._max:
            _msg = f"Minimum {val} cannot be greater"
            _msg = f" than maximum {self._max}."
            raise ValueError(_msg)
        self._min = val

    @property
    def max(self) -> Optional[float]:
        """*max* Get the maximum range value.

        Returns:
            Optional[float]: Maximum range value.
        """
        return self._max

    @max.setter
    def max(self, val: Optional[float]) -> None:
        """*max* Sets the maximum range value.

        Args:
            val (Optional[float]): Maximum range value.

        Raises:
            ValueError: If value is invalid or less than min.
        """
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError("Maximum val must be a number.")
        if val < self._min:
            _msg = f"Maximum {val} cannot be less"
            _msg = f" than minimum {self._min}."
            raise ValueError(_msg)
        self._max = val

    @property
    def avg(self) -> Optional[float]:
        """*avg* Get the average value.

        Returns:
            Optional[float]: average value.
        """
        return self._avg

    @avg.setter
    def avg(self, val: Optional[float]) -> None:
        """*avg* Sets the average value.

        Args:
            val (Optional[float]): average value.

        Raises:
            ValueError: If value is invalid or outside min-max range.
        """
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError("Average value must be a number.")

        low = (self._min is not None and val < self._min)
        high = (self._max is not None and val > self._max)
        if low or high:
            _msg = f"Average {val}. "
            _msg += f"must be between {self._min} and {self._max}."
            raise ValueError(_msg)
        self._avg = val

    @property
    def step(self) -> Optional[float]:
        """*step* Get standardized step size.

        Returns:
            Optional[float]: Step size (always standardized).
        """
        return self._step

    @step.setter
    def step(self, val: Optional[float]) -> None:
        """*step* Set standardized step size.

        Args:
            val (Optional[float]): Step size (always standardized).

        Raises:
            ValueError: If step is invalid, zero, or too large.

        """
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError("Step must be a number.")

        if val == 0:
            raise ValueError("Step cannot be zero.")

        if val >= self._std_max - self._std_min:
            _msg = f"Step {val} must be less than range"
            _msg += f" {self._std_max - self._std_min}."
            raise ValueError(_msg)

        # Update range if all values are available
        if all([self._std_min is not None,
                self._std_max is not None,
                self._step is not None]):
            self._std_rng = np.arange(self._std_min,
                                      self._std_max,
                                      self._step)

        self._step = val

    @property
    def data(self) -> np.ndarray:
        """*data* Get the data array.

        Returns:
            np.ndarray: Data array.
        """
        return self._data

    @data.setter
    def data(self, val: Optional[np.ndarray]) -> None:
        """*data* Set the data array.

        Args:
            val (Optional[np.ndarray]): Data array.

        Raises:
            ValueError: If value is not a numpy array.
        """
        if val is None:
            # Generate array from min, max, step
            if all([self._min is not None,
                    self._max is not None,
                    self._step is not None]):
                self._data = np.arange(self._min,
                                       self._max,
                                       self._step)
        elif not isinstance(val, np.ndarray):
            raise ValueError(f"Data must be a numpy array, got {type(val).__name__}")
        else:
            self._data = val

    def clear(self) -> None:
        """*clear()* Reset all attributes to default values.

        Resets all coefficient properties to their initial state.
        """
        # Reset base class attributes
        self._idx = -1
        self._sym = ""
        self._fwk = "PHYSICAL"
        self.name = ""
        self.description = ""

        # Reset coefficient-specific attributes
        self._cat = "COMPUTED"
        self._param_lt = []
        self._dim_col = []
        self._pivot_lt = None
        self._pi_expr = None
        self.par_dims = None
        self._min = None
        self._max = None
        self._avg = None
        self._step = 1e-3
        self._data = np.array([])
        self.relevance = True

    def to_dict(self) -> Dict[str, Any]:
        """*to_dict()* Convert coefficient to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation of coefficient.
        """
        return {
            "name": self.name,
            "description": self.description,
            "idx": self._idx,
            "sym": self._sym,
            "fwk": self._fwk,
            "cat": self._cat,
            "param_lt": self._param_lt,
            "dim_col": self._dim_col,
            "pivot_lt": self._pivot_lt,
            "pi_expr": self._pi_expr,
            "par_dims": self.par_dims,
            "min": self._min,
            "max": self._max,
            "avg": self._avg,
            "step": self._step,
            "relevance": self.relevance
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Coefficient:
        """*from_dict()* Create coefficient from dictionary representation.

        Args:
            data (Dict[str, Any]): Dictionary representation of coefficient.

        Returns:
            Coefficient: New coefficient instance.
        """
        return cls(**data)
