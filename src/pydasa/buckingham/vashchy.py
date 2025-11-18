# -*- coding: utf-8 -*-
"""
Module vashchy.py
===========================================

Module for representing Dimensionless Coefficients in Dimensional Analysis for *PyDASA*.

This module provides the Coefficient class which models dimensionless numbers used in Vaschy-Buckingham's π-theorem for dimensional analysis.

Classes:

    **Coefficient**: Represents a dimensionless coefficient with properties, validation, and symbolic expression.

*IMPORTANT:* Based on the theory from:

    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""

# native python modules
from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Optional, List, Dict, Any, Generic, Tuple, Union, Sequence
# import re

# Third-party modules
import numpy as np

# Import validation base classes
from pydasa.core.basic import Validation
from pydasa.core.parameter import Variable

# Import utils
from pydasa.utils.default import T
from pydasa.utils.error import inspect_var
from pydasa.utils.latex import latex_to_python
# Import global configuration
# import the 'cfg' module to allow global variable edition
from pydasa.utils import config as cfg


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
        _alias (str): Python-compatible alias for use in code.
        _fwk (str): Framework context (PHYSICAL, COMPUTATION, SOFTWARE, CUSTOM).
        _cat (str): Category (COMPUTED, DERIVED).
        relevance (bool): Flag indicating if coefficient is relevant for analysis.

        # Coefficient Construction
        _variables (Dict[str, Variable]): Variables symbols used in coefficient.
        _dim_col (List[int]): Dimensional column for matrix operations.
        _pivot_lt (List[int]): Pivot indices in dimensional matrix.
        _pi_expr (str): Symbolic expression of coefficient.
        var_dims (Dict[str, int]): Dimensional variable exponents.

        # Value Ranges
        _min (float): Minimum value of the coefficient.
        _max (float): Maximum value of the coefficient.
        _mean (float): Average value of the coefficient.
        _step (float): Step size for simulations.
        _data (np.ndarray): Array of coefficient values for analysis.
    """

    # Category attribute (COMPUTED, DERIVED)
    # :attr: _cat
    _cat: str = "COMPUTED"
    """Category of the coefficient (COMPUTED, DERIVED)."""

    # Coefficient construction properties
    # :attr: _variables
    _variables: Dict[str, Variable] = field(default_factory=dict)
    """Variables symbols used in the coefficient."""

    # Coefficient calculation related variables
    # :attr: _dim_col
    _dim_col: List[int] = field(default_factory=list)
    """Dimensional column for matrix operations."""

    # :attr: _pivot_lt
    _pivot_lt: Optional[List[int]] = field(default_factory=list)
    """Pivot indices in dimensional matrix."""

    # :attr: _pi_expr
    _pi_expr: Optional[str] = None
    """Symbolic expression of coefficient."""

    # :attr: var_dims
    var_dims: Optional[Dict[str, int]] = field(default_factory=dict)
    """Dimensional variable exponents in coefficient."""

    # Value ranges Variable
    # :attr: _min
    _min: Optional[float] = None
    """Minimum value of the coefficient, always in standardized units."""

    # :attr: _max
    _max: Optional[float] = None
    """Maximum value of the coefficient, always in standardized units."""

    # :attr: _mean
    _mean: Optional[float] = None
    """Average value of the coefficient, always in standardized units."""

    # :attr: _dev
    _dev: Optional[float] = None
    """Standard deviation of the coefficient, always in standardized units."""

    # :attr: _step
    _step: Optional[float] = 1e-3
    """Step size for simulations, always in standardized units."""

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
        based on variable symbols and their respective dimensional exponents.

        Raises:
            ValueError: If variable list and dimensional column have different lengths.
        """
        # Initialize from base class
        super().__post_init__()

        # Set the Pi symbol if not specified
        if not self._sym:
            if self._idx >= 0:
                self._sym = f"\\Pi_{{{self._idx}}}"
            else:
                self._sym = "\\Pi_{}"
        # Set the Python alias if not specified
        if not self._alias:
            self._alias = latex_to_python(self._sym)

        self.cat = self._cat
        self.variables = self._variables

        # Defaults to empty list
        self.dim_col = self._dim_col or []
        self.pivot_lt = self._pivot_lt or []

        # Only build expression if we have variables
        if len(self._variables) > 0 and len(self._dim_col) > 0:
            var_keys = list(self._variables.keys())
            self.pi_expr, self.var_dims = self._build_expression(var_keys,
                                                                 self._dim_col)

        else:
            self.pi_expr = ""
            self.var_dims = {}

        # Set data
        self.data = self._data

        # # Build expression if parameters and dimensions are provided
        # FIXME this is not working, fix later!
        # if self._variables and self._dim_col:
        #     self.pi_expr, self.var_dims = self._build_expression(self._variables, self._dim_col)

        # Set up data array if all required values are provided
        if all([self._min, self._max, self._step]):
            self._data = np.arange(self._min, self._max, self._step)

    def _validate_sequence(self, seq: Sequence,
                           exp_type: Union[type, Tuple[type, ...]]) -> bool:
        """*_validate_sequence()* Validates a list with expected element types.

        Args:
            seq (Sequence): Sequence to validate.
            exp_type (Union[type, Tuple[type, ...]]): Expected type(s) for sequence elements. Can be a single type or a tuple of types.

        Raises:
            ValueError: If the object is not a sequence.
            ValueError: If the sequence is empty.
            ValueError: If the sequence contains elements of unexpected types.

        Returns:
            bool: True if the list is valid.
        """
        # Explicitly reject strings (they're sequences but not what we want)
        if isinstance(seq, str):
            _msg = f"{inspect_var(seq)} must be a list or tuple, not a string. "
            _msg += f"Provided: {type(seq).__name__}"
            raise ValueError(_msg)

        if not isinstance(seq, Sequence):
            _msg = f"{inspect_var(seq)} must be from type: '{exp_type}', "
            _msg += f"Provided: {type(seq).__name__}"
            raise ValueError(_msg)

        if len(seq) == 0:
            _msg = f"{inspect_var(seq)} cannot be empty. Actual sequence: {seq}"
            raise ValueError(_msg)

        # Convert list to tuple if needed for isinstance(), just in case
        type_check = exp_type if isinstance(exp_type, tuple) else (exp_type,)

        if not all(isinstance(x, type_check) for x in seq):

            # Format expected types for error message
            if isinstance(exp_type, tuple):
                type_names = " or ".join(t.__name__ for t in exp_type)
            else:
                type_names = exp_type.__name__

            _msg = f"{inspect_var(seq)} must contain {type_names} elements."
            _msg += f" Provided: {[type(x).__name__ for x in seq]}"
            raise ValueError(_msg)

        return True

    def _build_expression(self,
                          var_lt: List[str],
                          dim_col: List[int]) -> tuple[str, dict]:
        """*_build_expression()* Builds LaTeX expression for coefficient.

        Args:
            var_lt (List[str]): List of variable symbols.
            dim_col (List[int]): List of dimensional exponents.

        Raises:
            ValueError: If variable list and dimensional column have different lengths.

        Returns:
            tuple[str, Dict[str, int]]: LaTeX expression and variable exponents.
        """
        # Validate variable list and dimensional column
        if len(var_lt) != len(dim_col):
            _msg = f"Variables list len ({len(var_lt)}) and "
            _msg += f"dimensional column len ({len(dim_col)}) must be equal."
            raise ValueError(_msg)

        # Initialize working variables
        numerator = []
        denominator = []
        parameters = {}

        # Process parameters and their exponents
        for sym, exp in zip(var_lt, dim_col):
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
            # Store variable exponent
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
            ValueError: If category is not supported.
        """
        if val.upper() not in cfg.DC_CAT_DT:
            _msg = f"Category {val} is invalid. "
            _msg += f"Must be one of: {', '.join(cfg.DC_CAT_DT.keys())}"
            raise ValueError(_msg)
        self._cat = val.upper()

    @property
    def variables(self) -> Dict[str, Variable]:
        """*variables* Get the variable symbols list.

        Returns:
            Dict[str, Variable]: Variables symbols list.
        """
        return self._variables

    @variables.setter
    def variables(self, val: Dict[str, Variable]) -> None:
        """*variables* Set the variable symbols list.

        Args:
            val (Dict[str, Variable]): Variables symbols list.

        Raises:
            ValueError: If variables list is invalid.
        """
        # Validate type
        if not isinstance(val, dict):
            _msg = f"Variables must be a dict, got {type(val).__name__}"
            raise ValueError(_msg)

        # check non-empty dictt
        if len(val) > 0:
            self._validate_sequence(list(val.keys()), (str,))
            self._validate_sequence(list(val.values()), (Variable,))

        # If validation passes, assign
        self._variables = val

    @property
    def dim_col(self) -> List[int]:
        """*dim_col* Get the dimensional column.

        Returns:
            List[int]: Dimensional column.
        """
        return self._dim_col

    @dim_col.setter
    def dim_col(self, val: Union[List[int], List[float]]) -> None:
        """*dim_col* Set the dimensional column.

        Args:
            val (Union[List[int], List[float]]): Dimensional column.
        Raises:
            ValueError: If dimensional column is invalid.
        """
        # Validate type first
        if not isinstance(val, list):
            _msg = "Dimensions must be a list with int or float values. "
            _msg += f"Provided: {type(val).__name__}"
            raise ValueError(_msg)

        # Validate sequence if not empty
        if len(val) > 0:
            self._validate_sequence(val, (int, float))

        # If validation passes, assign (convert to integers)
        self._dim_col = [int(x) for x in val]

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
            ValueError: If pivot list is invalid.
        """
        # Handle None and empty list (both allowed)
        if val is None or len(val) == 0:
            self._pivot_lt = val
            return

        # Validate non-empty list
        self._validate_sequence(val, (int,))
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
            ValueError: If the coefficient expression is not a string.
        """
        if not isinstance(val, str):
            _msg = f"Expression must be a string, got {type(val).__name__}."
            raise ValueError(_msg)
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
            ValueError: If value is not a number.
            ValueError: If value is greater than max.
        """
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError("Minimum range must be a number.")

        if val is not None and self._max is not None and val > self._max:
            _msg = f"Minimum {val} cannot be greater than maximum {self._max}."
            raise ValueError(_msg)

        self._min = val

        # Update range if all values are available
        if all([self._min is not None,
                self._max is not None,
                self._step is not None]):
            self._range = np.arange(self._min,
                                    self._max,
                                    self._step)

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
            ValueError: If value is not a number.
            ValueError: If value is less than min.
        """
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError("Maximum val must be a number.")

        # Check if both values exist before comparing
        if val is not None and self._min is not None and val < self._min:
            _msg = f"Maximum {val} cannot be less than minimum {self._min}."
            raise ValueError(_msg)

        self._max = val

        # Update range if all values are available
        if all([self._min is not None,
                self._max is not None,
                self._step is not None]):
            self._range = np.arange(self._min,
                                    self._max,
                                    self._step)

    @property
    def mean(self) -> Optional[float]:
        """*mean* Get the average value.

        Returns:
            Optional[float]: average value.
        """
        return self._mean

    @mean.setter
    def mean(self, val: Optional[float]) -> None:
        """*mean* Sets the average value.

        Args:
            val (Optional[float]): average value.

        Raises:
            ValueError: If value is not a number.
            ValueError: If value is outside min-max range.
        """
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError("Mean value must be a number.")

        # Only validate range if val is not None
        if val is not None:
            low = (self._min is not None and val < self._min)
            high = (self._max is not None and val > self._max)
            if low or high:
                _msg = f"Mean {val} "
                _msg += f"must be between {self._min} and {self._max}."
                raise ValueError(_msg)

        self._mean = val

    @property
    def dev(self) -> Optional[float]:
        """*dev* Get the Variable standard deviation.

        Returns:
            Optional[float]: Variable standard deviation.
        """
        return self._dev

    @dev.setter
    def dev(self, val: Optional[float]) -> None:
        """*dev* Sets the Variable standard deviation.

        Args:
            val (Optional[float]): Variable standard deviation.
        Raises:
            ValueError: If value not a valid number.
        """
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError("Standard deviation must be a number.")

        self._dev = val

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
            ValueError: If step is not a valid number
            ValueError: If step is zero.
            ValueError: If step is greater than range.
        """
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError("Step must be a number.")

        if val == 0:
            raise ValueError("Step cannot be zero.")

        # Validate step against range (only if min/max are set)
        if val is not None and self._min is not None and self._max is not None:
            range_size = self._max - self._min
            if val >= range_size:
                _msg = f"Step {val} must be less than range: {range_size}."
                raise ValueError(_msg)

        self._step = val

        # Update range if all values are available
        if all([self._min is not None,
                self._max is not None,
                self._step is not None]):
            self._range = np.arange(self._min,
                                    self._max,
                                    self._step)

    @property
    def data(self) -> np.ndarray:
        """*data* Get the data array.

        Returns:
            np.ndarray: Data array. If not explicitly set, generates from min, max, step.
        """
        # If data was explicitly set, return it
        if len(self._data) > 0:
            return self._data

        # Otherwise, generate from range parameters
        if self._min is not None and self._max is not None and self._step != 0:
            return np.arange(self._min, self._max, self._step)

        # Default to empty array
        return np.array([], dtype=float)

    @data.setter
    def data(self, val: Union[np.ndarray, list]) -> None:
        """*data* Set the data array.

        Args:
            val (Union[np.ndarray, list]): Data array or list.

        Raises:
            ValueError: If data cannot be converted to numpy array.
        """
        if not isinstance(val, (np.ndarray, list,)):
            _msg = "Data must be a numpy array. "
            _msg += f"Provided: {type(val).__name__}"
            raise ValueError(_msg)

        # Convert list to numpy array if needed
        if isinstance(val, list):
            self._data = np.array(val, dtype=float)

        # otherwise, let it be
        else:
            self._data = val

    def clear(self) -> None:
        """*clear()* Reset all attributes to default values.

        Resets all coefficient properties to their initial state.
        """
        # Reset base class attributes
        self._idx = -1
        self._sym = ""
        self._alias = ""
        self._fwk = "PHYSICAL"
        self.name = ""
        self.description = ""

        # Reset coefficient-specific attributes
        self._cat = "COMPUTED"
        self._variables = {}
        self._dim_col = []
        self._pivot_lt = None
        self._pi_expr = None
        self.var_dims = None
        self._min = None
        self._max = None
        self._mean = None
        self._step = 1e-3
        self._data = np.array([])
        self.relevance = True

    def to_dict(self) -> Dict[str, Any]:
        """*to_dict()* Convert variable to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation of variable.
        """
        result = {}

        # Get all dataclass fields
        for f in fields(self):
            attr_name = f.name
            attr_value = getattr(self, attr_name)

            # Skip numpy arrays (not JSON serializable without special handling)
            if isinstance(attr_value, np.ndarray):
                # Convert to list for JSON compatibility
                attr_value = attr_value.tolist()

            # Skip callables (can't be serialized)
            if callable(attr_value) and attr_name == "_dist_func":
                continue

            # Remove leading underscore from private attributes
            if attr_name.startswith("_"):
                clean_name = attr_name[1:]  # Remove first character
            else:
                clean_name = attr_name

            result[clean_name] = attr_value

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Coefficient:
        """*from_dict()* Create variable from dictionary representation.

        Args:
            data (Dict[str, Any]): Dictionary representation of variable.

        Returns:
            Variable: New variable instance.
        """
        # Get all valid field names from the dataclass
        field_names = {f.name for f in fields(cls)}

        # Map keys without underscores to keys with underscores
        mapped_data = {}

        for key, value in data.items():
            # Try the key as-is first (handles both _idx and name)
            if key in field_names:
                mapped_data[key] = value
            # Try adding underscore prefix (handles idx -> _idx)
            elif f"_{key}" in field_names:
                mapped_data[f"_{key}"] = value
            # Try removing underscore prefix (handles _name -> name if needed)
            elif key.startswith("_") and key[1:] in field_names:
                mapped_data[key[1:]] = value
            else:
                # Use as-is for unknown keys (will be validated by dataclass)
                mapped_data[key] = value

        # Convert lists back to numpy arrays for range attributes
        for range_key in ["std_range", "_std_range"]:
            if range_key in mapped_data and isinstance(mapped_data[range_key], list):
                mapped_data[range_key] = np.array(mapped_data[range_key])

        return cls(**mapped_data)
