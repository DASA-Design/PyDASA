# -*- coding: utf-8 -*-
"""
Module parameter.py
===========================================

Module for representing **Variable** entities in Dimensional Analysis for *PyDASA*.

Classes:
    **Variable**: Represents a Variable with dimensional properties, value ranges, and validation.

*IMPORTANT:* Based on the theory from:

    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""

# native python modules
# dataclass imports
from __future__ import annotations
from dataclasses import dataclass, field, fields
# data type inports
from typing import Optional, List, Dict, Any, Callable
import re
# numerical imports
import numpy as np

# custom modules
# basic-core class imports
from pydasa.core.basic import Foundation

# pattern interpreter import
from pydasa.utils.latex import latex_to_python

# Import configuration
# import the 'cfg' module to allow global variable edition
from pydasa.core import setup as cfg


@dataclass
class Variable(Foundation):
    """**Variable** class for Dimensional Analysis in *PyDASA*.

    A comprehensive implementation that combines Parameter and Variable functionality
    for use in dimensional analysis calculations, simulations, and sensitivity analysis.

    Attributes:
        # Identification and Classification
        name (str): User-friendly name of the variable.
        description (str): Brief summary of the variable.
        _idx (int): Index/precedence in the dimensional matrix.
        _sym (str): Symbol representation (LaTeX or alphanumeric).
        _alias (str): Python-compatible alias for use in code.
        _fwk (str): Framework context (PHYSICAL, COMPUTATION, SOFTWARE, CUSTOM).
        _cat (str): Category (INPUT, OUT, CTRL).
        relevant (bool): Flag indicating if variable is relevant for analysis.

        # Dimensional Properties
        _dims (str): Dimensional expression (e.g., "L*T^-1").
        _units (str): Units of measure (e.g., "m/s").
        _sym_exp (str): Sympy-compatible expression.
        _dim_col (List[int]): Dimensional column for matrix operations.

        # Standarized Dimensional Properties
        _std_dims (str): Standardized dimensional expression.

        # Value Ranges (Original Units)
        _min (float): Minimum value in original units.
        _max (float): Maximum value in original units.
        _mean (float): Mean value in original units.
        _dev (float): Standard deviation in original units.

        # Value Ranges (Standardized Units)
        _std_units (str): Standardized units of measure.
        _std_min (float): Minimum value in standard units.
        _std_max (float): Maximum value in standard units.
        _std_mean (float): Mean value in standard units.
        _std_dev (float): Standard deviation in standard units.
        _step (float): Step size for simulations.
        _std_range (np.ndarray): Range array for analysis.
    """

    # Private attributes
    # Category attribute (INPUT, OUT, CTRL)
    # :attr: _cat
    _cat: str = "IN"
    """Category of the variable (INPUT, OUT, CTRL)."""

    # Dimensional properties
    # :attr: _dims
    _dims: str = ""
    """Dimensional expression (e.g., "L*T^-1")."""

    # :attr: _units
    _units: str = ""
    """Units of measure (e.g., "m/s")."""

    # Processed dimensional attributes
    # :attr: _std_dims
    _std_dims: Optional[str] = None
    """Standardized dimensional expression. e.g.: from [T^2*L^-1] to [L^(-1)*T^(2)]."""

    # :attr: _sym_exp
    _sym_exp: Optional[str] = None
    """Sympy-compatible dimensional expression. e.g.: from [T^2*L^-1] to [T**2*L**(-1)]."""

    # :attr: _std_col
    _dim_col: List[int] = field(default_factory=list)
    """Dimensional column for matrix operations. e.g.: from [T^2*L^-1] to [2, -1]."""

    # Value ranges (original units)
    # :attr: _min
    _min: Optional[float] = None
    """Minimum value in original units."""

    # :attr: _max
    _max: Optional[float] = None
    """Maximum value in original units."""

    # :attr: _mean
    _mean: Optional[float] = None
    """Mean value in original units."""

    # :attr: _dev
    _dev: Optional[float] = None
    """Standard deviation in original units."""

    # Value ranges (standardized units)
    # :attr: _std_units
    _std_units: str = ""
    """Standardized units of measure. e.g `km/h` -> `m/s`, `kByte/s` -> `bit/s`."""

    # :attr: _std_min
    _std_min: Optional[float] = None
    """Minimum value in standard units."""

    # :attr: _std_max
    _std_max: Optional[float] = None
    """Maximum value in standard units."""

    # :attr: _std_mean
    _std_mean: Optional[float] = None
    """Mean value in standard units."""

    # :attr: _std_dev
    _std_dev: Optional[float] = None
    """Standard deviation in standard units."""

    # :attr: _step
    _step: Optional[float] = None
    """Step size for simulations."""

    # :attr: _std_range
    _std_range: np.ndarray = field(default_factory=lambda: np.array([]))
    """Range array for analysis."""

    # distribution specifications
    # :attr: _dist_type
    _dist_type: str = "uniform"
    """Type of distribution (e.g., 'uniform', 'normal'). By default is 'uniform'."""

    # :attr: _dist_params
    _dist_params: Optional[Dict[str, Any]] = field(default_factory=dict)
    """Parameters for the distribution (e.g., {'min': 0, 'max': 1} for uniform)."""

    # :attr: _depends
    _depends: List[str] = field(default_factory=list)
    """List of variable names that this variable depends on. (e.g., for calculated variables like F = m*a)."""

    # :attr: _dist_func
    _dist_func: Optional[Callable[..., float]] = None
    """Callable representing the distribution function defined externally by the user."""

    # Flags
    # :attr: relevant
    relevant: bool = False
    """Flag indicating if variable is relevant for analysis."""

    def __post_init__(self) -> None:
        """*__post_init__()* Initializes the variable and validates its properties.

        Performs validation of core properties and processes dimensional expressions.
        Sets up range arrays if all required values are provided.

        Raises:
            ValueError: If dimensional expression is invalid.
        """
        # Initialize from base class
        super().__post_init__()

        if not self._sym:
            self._sym = f"V_{self._idx}" if self._idx >= 0 else "V_{}"

        # Set the Python alias if not specified
        if not self._alias:
            self._alias = latex_to_python(self._sym)

        # Process dimensions if provided
        if len(self._dims) > 0 and self._dims != "n.a.":
            if not self._validate_exp(self._dims, cfg.WKNG_FDU_RE):
                _msg = f"Invalid dimensions '{self.name}': {self._dims}. "
                _msg += f"Check FDUs according to precedence: {cfg.WKNG_FDU_PREC_LT}"
                raise ValueError(_msg)
            self._prepare_dims()

        # Set up range array if all required values are provided
        if all([self._std_min, self._std_max, self._step]):
            self._std_range = np.arange(self._std_min,
                                        self._std_max,
                                        self._step)

    def _validate_exp(self, exp: str, regex: str) -> bool:
        """*_validate_exp()* Validates an expression using a regex pattern (inclde dimensions and units,).

        Args:
            dims (str): Dimensions of the parameter. It is a string with the FDU formula of the parameter. e.g.: [T^2*L^-1]

        Returns:
            bool: True if the dimensions are valid, False otherwise, ignoring null or empty strings.
        """
        # TODO improve this ignoring null or empty strings for constants
        if exp in [None, ""]:
            return True
        return bool(re.match(regex, exp))

    def _validate_in_list(self, value: str, prec_lt: List[str]) -> bool:
        """*_validate_in_list()* Validates if a value exists in a list of allowed values.

        Args:
            value (str): Value to validate.
            prec_lt (List[str]): List of allowed values.

        Returns:
            bool: True if the value is in the list, False otherwise.
        """
        if value in [None, ""]:
            return False
        return value in prec_lt

    def _prepare_dims(self) -> None:
        """*_prepare_dims()* Processes dimensional expressions for analysis.

        Standardizes and sorts dimensions, creates sympy expression and dimensional column.
        """
        self._std_dims = self._standardize_dims(self._dims)
        self._std_dims = self._sort_dims(self._std_dims)
        self._sym_exp = self._setup_sympy(self._std_dims)
        self._dim_col = self._setup_column(self._sym_exp)

    def _standardize_dims(self, dims: str) -> str:
        """*_standardize_dims()* Standardizes dimensional expression format.

        Args:
            dims (str): Dimensional expression (e.g., "L*T^-1").

        Returns:
            str: Standardized expression with parentheses (e.g., "L^(1)*T^(-1)").
        """
        # Add parentheses to powers
        _pattern = re.compile(cfg.WKNG_POW_RE)
        dims = _pattern.sub(lambda m: f"({m.group(0)})", dims)

        # Add ^1 to dimensions without explicit powers
        _pattern = re.compile(cfg.WKNG_NO_POW_RE)
        dims = _pattern.sub(lambda m: f"{m.group(0)}^(1)", dims)
        return dims

    def _sort_dims(self, dims: str) -> str:
        """*_sort_dims()* Sorts dimensions based on FDU precedence.

        Args:
            dims (str): Standardized dimensional expression. e.g.: [T^2*L^-1].

        Returns:
            str: Sorted dimensional expression. e.g.: [L^(-1)*T^(2)].
        """
        # TODO move '*' as global operator to cfg module?
        # Split by multiplication operator
        _dims_lt = dims.split("*")

        # Sort based on FDU precedence
        _dims_lt.sort(key=lambda x: cfg.WKNG_FDU_PREC_LT.index(x[0]))

        # Rejoin with multiplication operator
        _dims = "*".join(_dims_lt)
        return _dims

    def _setup_sympy(self, dims: str) -> str:
        """*_setup_sympy()* Creates sympy-compatible expression.

        Args:
            dims (str): Standardized dimensional expression. e.g.: [T^2*L^-1].

        Returns:
            str: Sympy-compatible expression. e.g.: [T**2* L**(-1)].
        """
        # TODO move '*' and '* ' as global operator to cfg module?
        # TODO do I use also regex for this?
        # replace '*' with '* ' for sympy processing
        # # replace '^' with '**' for sympy processing
        return dims.replace("*", "* ").replace("^", "**")

    def _setup_column(self, dims: str) -> List[int]:
        """*_setup_column()* Generates dimensional column (list of exponents) in the Dimensional Matrix.

        Args:
            dims (str): Standardized dimensional expression. e.g.: [T^(2)*L^(-1)]

        Returns:
            List[int]: Exponents with the dimensional expression. e.g.: [2, -1]

        Raises:
            ValueError: If dimensional expression cannot be parsed.
        """
        # split the sympy expression into a list of dimensions
        dims_list = dims.split("* ")
        # set the default list of zeros with the FDU length
        col = [0] * len(cfg.WKNG_FDU_PREC_LT)

        for dim in dims_list:
            # match the exponent of the dimension
            exp_match = re.search(cfg.WKNG_POW_RE, dim)
            if exp_match is None:
                _msg = f"Could not extract exponent from dimension: {dim}"
                raise ValueError(_msg)
            _exp = int(exp_match.group(0))

            # match the symbol of the dimension
            sym_match = re.search(cfg.WKNG_FDU_SYM_RE, dim)
            if sym_match is None:
                _msg = f"Could not extract symbol from dimension: {dim}"
                raise ValueError(_msg)
            _sym = sym_match.group(0)

            # Check if symbol exists in the precedence list
            if _sym not in cfg.WKNG_FDU_PREC_LT:
                _msg = f"Unknown dimensional symbol: {_sym}"
                raise ValueError(_msg)

            # update the column with the exponent of the dimension
            col[cfg.WKNG_FDU_PREC_LT.index(_sym)] = _exp

        return col

    def sample(self, *args) -> float:
        """*sample()* Generate a random sample.

        Args:
            *args: Additional arguments for the distribution function.

        Returns:
            float: Random sample from distribution.

        Raises:
            ValueError: If no distribution has been set.
        """
        if self._dist_func is None:
            _msg = f"No distribution set for variable '{self.sym}'. "
            _msg += "Call set_function() first."
            raise ValueError(_msg)

        # if kwargs are provided, pass them to the function parameters
        elif len(args) > 0:
            return self._dist_func(*args)

        # otherwise, execute without them
        return self._dist_func()

    def has_function(self) -> bool:
        """*has_function()* Check if distribution is set.

        Returns:
            bool: True if distribution is configured.
        """
        return self._dist_func is not None

    # Property getters and setters
    # Identification and Classification

    @property
    def cat(self) -> str:
        """*cat* Get the category of the variable.

        Returns:
            str: Category (INPUT, OUT, CTRL).
        """
        return self._cat

    @cat.setter
    def cat(self, val: str) -> None:
        """*cat* Set the category of the variable.

        Args:
            val (str): Category (INPUT, OUT, CTRL).

        Raises:
            ValueError: If category is invalid.
        """
        _param_keys = cfg.PARAMS_CAT_DT.keys()
        if val.upper() not in _param_keys:
            _msg = f"Invalid category: {val}. "
            _msg += "Category must be one of the following: "
            _msg += f"{', '.join(_param_keys)}."
            raise ValueError(_msg)
        self._cat = val.upper()

    # Dimensional Properties

    @property
    def dims(self) -> str:
        """*dims* Get the dimensional expression.

        Returns:
            str: Dimensions. e.g.: [T^2*L^-1]
        """
        return self._dims

    @dims.setter
    def dims(self, val: str) -> None:
        """*dims* Sets the dimensional expression.

        Args:
            val (str): Dimensions. e.g.: [T^2*L^-1]

        Raises:
            ValueError: If expression is empty
            ValueError: If dimensions are invalid according to the precedence.
        """
        # _precedence_lt = cfg.WKNG_FDU_PREC_LT
        if val is not None and not val.strip():
            raise ValueError("Dimensions cannot be empty.")

        # Process dimensions
        if val and not self._validate_exp(val, cfg.WKNG_FDU_RE):
            _msg = f"Invalid dimensional expression: {val}. "
            _msg += f"FDUS precedence is: {cfg.WKNG_FDU_RE}"
            raise ValueError(_msg)

        self._dims = val

        # automatically prepare the dimensions for analysis
        self._prepare_dims()

    @property
    def units(self) -> str:
        """*units* Get the units of measure.

        Returns:
            str: Units of measure. e.g.: `m/s`, `kg/m3`, etc.
        """
        return self._units

    @units.setter
    def units(self, val: str) -> None:
        """*units* Sets the units of measure.

        Args:
            val (str): Units of measure. i.e `m/s`, `kg/m3`, etc.

        Raises:
            ValueError: If units are empty.
        """
        if val is not None and not val.strip():
            raise ValueError("Units of Measure cannot be empty.")
        self._units = val

    @property
    def sym_exp(self) -> Optional[str]:
        """*sym_exp* Get Sympy-compatible expression.

        Returns:
            Optional[str]: Sympy expression. e.g.: [T**2*L**(-1)]
        """
        return self._sym_exp

    @sym_exp.setter
    def sym_exp(self, val: str) -> None:
        """*sym_exp* Sets Sympy-compatible expression.

        Args:
            val (str): Sympy expression. e.g.: [T**2*L**(-1)]

        Raises:
            ValueError: If the string is empty.
        """
        if val is not None and not val.strip():
            raise ValueError("Dimensional expression cannot be empty.")
        self._sym_exp = val

    @property
    def dim_col(self) -> Optional[List[int]]:
        """*dim_col* Get dimensional column.

        Returns:
            Optional[List[int]]: Dimensional exponents. e.g.: [2, -1]
        """
        return self._dim_col

    @dim_col.setter
    def dim_col(self, val: List[int]) -> None:
        """*dim_col* Sets the dimensional column

        Args:
            val (List[int]): Dimensional exponents. i.e..: [2, -1]

        Raises:
            ValueError: if the dimensional column is not a list of integers.
        """
        if val is not None and not isinstance(val, list):
            raise ValueError("Dimensional column must be a list of integers.")
        self._dim_col = val

    # Standardized Dimensional Properties

    @property
    def std_dims(self) -> Optional[str]:
        """*std_dims* Get the standardized dimensional expression.

        Returns:
            Optional[str]: Standardized dimensional expression. e.g.: [L^(-1)*T^(2)]
        """
        return self._std_dims

    @std_dims.setter
    def std_dims(self, val: str) -> None:
        """*std_dims* Sets the standardized dimensional expression.

        Args:
            val (str): Standardized dimensional expression. e.g.: [L^(-1)*T^(2)]

        Raises:
            ValueError: If the standardized dimensional expression is empty.
        """
        if val is not None and not val.strip():
            raise ValueError("Standardized dimensions cannot be empty.")
        self._std_dims = val

    # Value Ranges (Original Units)

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
            ValueError: If value not a valid number.
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
            ValueError: If value is not a valid number.
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
        """*mean* Get the Variable average value.

        Returns:
            Optional[float]: Variable average value.
        """
        return self._mean

    @mean.setter
    def mean(self, val: Optional[float]) -> None:
        """*mean* Sets the Variable mean value.

        Args:
            val (Optional[float]): Variable mean value.

        Raises:
            ValueError: If value not a valid number.
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

    # Value Ranges (Standardized Units)

    @property
    def std_units(self) -> Optional[str]:
        """*std_units* Get the standardized Unit of Measure.

        Returns:
            Optional[str]: standardized Unit of Measure.
        """
        return self._std_units

    @std_units.setter
    def std_units(self, val: str) -> None:
        """*std_units* Sets the standardized Unit of Measure.

        Args:
            val (Optional[str]): standardized Unit of Measure.

        Raises:
            ValueError: If standardized units are empty.
        """
        if val is not None and not val.strip():
            raise ValueError("Standardized Units of Measure cannot be empty.")
        self._std_units = val

    @property
    def std_min(self) -> Optional[float]:
        """*std_min* Get the standardized minimum range value.

        Returns:
            Optional[float]: standardized minimum range value.
        """
        return self._std_min

    @std_min.setter
    def std_min(self, val: Optional[float]) -> None:
        """*std_min* Sets the standardized minimum range value.

        Args:
            val (Optional[float]): standardized minimum range value.

        Raises:
            ValueError: If value not a valid number.
            ValueError: If value is greater than std_max.
        """
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError("Standardized minimum must be a number")

        # Check if both values exist before comparing
        if val is not None and self._std_max is not None and val > self._std_max:
            _msg = f"Standard minimum {val} cannot be greater"
            _msg += f" than standard maximum {self._std_max}."
            raise ValueError(_msg)

        self._std_min = val

        # Update range if all values are available
        if all([self._std_min is not None,
                self._std_max is not None,
                self._step is not None]):
            self._std_range = np.arange(self._std_min,
                                        self._std_max,
                                        self._step)

    @property
    def std_max(self) -> Optional[float]:
        """*std_max* Get the standardized maximum range value.

        Returns:
            Optional[float]: standardized maximum range value.
        """
        return self._std_max

    @std_max.setter
    def std_max(self, val: Optional[float]) -> None:
        """*std_max* Sets the standardized maximum range value.

        Raises:
            ValueError: If value is not a valid number.
            ValueError: If value is less than std_min.
        """
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError("Standardized maximum must be a number")

        # Check if both values exist before comparing
        if val is not None and self._std_min is not None and val < self._std_min:
            _msg = f"Standard maximum {val} cannot be less"
            _msg += f" than standard minimum {self._std_min}."
            raise ValueError(_msg)

        self._std_max = val

        # Update range if all values are available
        if all([self._std_min is not None,
                self._std_max is not None,
                self._step is not None]):
            self._std_range = np.arange(self._std_min,
                                        self._std_max,
                                        self._step)

    @property
    def std_mean(self) -> Optional[float]:
        """*std_mean* Get standardized mean value.

        Returns:
            Optional[float]: standardized mean.
        """
        return self._std_mean

    @std_mean.setter
    def std_mean(self, val: Optional[float]) -> None:
        """*std_mean* Sets the standardized mean value.

        Args:
            val (Optional[float]): standardized mean value.

        Raises:
            ValueError: If value is not a valid number.
            ValueError: If value is outside std_min-std_max range.
        """
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError("Standardized mean must be a number")

        # Only validate range if val is not None
        if val is not None:
            low = (self._std_min is not None and val < self._std_min)
            high = (self._std_max is not None and val > self._std_max)

            if low or high:
                _msg = f"Standard mean {val} "
                _msg += f"must be between {self._std_min} and {self._std_max}."
                raise ValueError(_msg)

        self._std_mean = val

    @property
    def std_dev(self) -> Optional[float]:
        """*std_dev* Get standardized standard deviation.

        Returns:
            Optional[float]: Standardized standard deviation.
        """
        return self._std_dev

    @std_dev.setter
    def std_dev(self, val: Optional[float]) -> None:
        """*std_dev* Sets the standardized standard deviation.

        Args:
            val (Optional[float]): Standardized standard deviation.

        Raises:
            ValueError: If value is not a valid number.
            ValueError: If value is outside std_min-std_max range.
        """
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError(
                "Standardized standard deviation must be a number")

        # Standard deviation should be non-negative
        if val is not None and val < 0:
            raise ValueError(f"Standard deviation {val} cannot be negative.")

        self._std_dev = val

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
        if val is not None and self._std_min is not None and self._std_max is not None:
            range_size = self._std_max - self._std_min
            if val >= range_size:
                _msg = f"Step {val} must be less than range: {range_size}."
                raise ValueError(_msg)

        self._step = val

        # Update range if all values are available
        if all([self._std_min is not None,
                self._std_max is not None,
                self._step is not None]):
            self._std_range = np.arange(self._std_min,
                                        self._std_max,
                                        self._step)

    @property
    def std_range(self) -> np.ndarray:
        """*std_range* Get standardized range array.

        Returns:
            np.ndarray: Range array for range (always standardized).
        """
        return self._std_range

    @std_range.setter
    def std_range(self, val: Optional[np.ndarray]) -> None:
        """*std_range* Set standardized range array.

        Args:
            val (Optional[np.ndarray]): Data array for range (always standardized).

        Raises:
            ValueError: If value is not a numpy array.
        """
        if val is None:
            # Generate range from min, max, step
            if all([self._std_min is not None,
                    self._std_max is not None,
                    self._step is not None]):
                self._std_range = np.arange(self._std_min,
                                            self._std_max,
                                            self._step)

        # TODO check this latter, might be a hindrance
        elif not isinstance(val, np.ndarray):
            _msg = f"Range must be a numpy array, got {type(val)}"
            raise ValueError(_msg)

        else:
            self._std_range = val

    @property
    def dist_type(self) -> str:
        """*dist_type* Get the distribution type.

        Returns:
            str: Distribution type (e.g., 'uniform', 'normal').
        """
        return self._dist_type

    @dist_type.setter
    def dist_type(self, val: str) -> None:
        """*dist_type* Set the distribution type.

        Args:
            val (str): Distribution type.

        Raises:
            ValueError: If distribution type is not supported.
        """
        # TODO improve this for later
        supported_types = [
            "uniform",
            "normal",
            "triangular",
            "exponential",
            "lognormal",
            "custom",
        ]
        if val not in supported_types:
            _msg = f"Unsupported distribution type: {val}. "
            _msg += f"Supported types: {', '.join(supported_types)}"
            raise ValueError(_msg)
        self._dist_type = val

    @property
    def dist_params(self) -> Optional[Dict[str, Any]]:
        """*dist_params* Get the distribution parameters.

        Returns:
            Optional[Dict[str, Any]: Distribution parameters.
        """
        return self._dist_params

    @dist_params.setter
    def dist_params(self, val: Optional[Dict[str, Any]]) -> None:
        """*dist_params* Set the distribution parameters.

        Args:
            val (Optional[Dict[str, Any]): Distribution parameters.

        Raises:
            ValueError: If parameters are invalid for the distribution type.
        """
        if val is None:
            self._dist_params = None
            return None
        # Validate parameters based on distribution type
        if self._dist_type == "uniform":
            if "min" not in val or "max" not in val:
                _msg = f"Invalid keys for: {self._dist_type}: {val}"
                _msg += f" {self._dist_type} needs 'min' and 'max' parameters."
                _msg += f" Provided keys are: {list(val.keys())}."
                raise ValueError(_msg)
            if val["min"] >= val["max"]:
                _msg = f"Invalid range for {self._dist_type}: {val}"
                _msg += f" {self._dist_type} needs 'min' to be less than 'max'."
                _msg += f" Provided: min={val['min']}, max={val['max']}."
                raise ValueError(_msg)
        elif self._dist_type == "normal":
            if "mean" not in val or "std" not in val:
                _msg = f"Invalid keys for: {self._dist_type}: {val}"
                _msg += f" {self._dist_type} needs 'mean' and 'std' parameters."
                _msg += f" Provided keys are: {list(val.keys())}."
                raise ValueError(_msg)
            if val["std"] < 0:
                _msg = f"Invalid value for: {self._dist_type}: {val}"
                _msg += f" {self._dist_type} requires 'std' to be positive."
                _msg += f" Provided: std={val['std']}."
                raise ValueError(_msg)
        self._dist_params = val

    @property
    def dist_func(self) -> Optional[Callable[..., float]]:
        """*dist_func* Get the distribution function.

        Returns:
            Optional[Callable]: Distribution function.
        """
        return self._dist_func

    @dist_func.setter
    def dist_func(self, val: Optional[Callable[..., float]]) -> None:
        """*dist_func* Set the distribution function.

        Args:
            val (Optional[Callable]): Distribution function.

        Raises:
            TypeError: If value is not callable when provided.
        """
        if val is not None and not callable(val):
            _msg = f"Distribution function must be callable, got {type(val)}"
            raise TypeError(_msg)
        self._dist_func = val

    @property
    def depends(self) -> List[str]:
        """*depends* Get the list of variable dependencies.

        Returns:
            List[str]: List of variable names that this variable depends on.
        """
        return self._depends

    @depends.setter
    def depends(self, val: List[str]) -> None:
        """*depends* Set the list of variable dependencies.

        Args:
            val (List[str]): List of variable names that this variable depends on.
        Raises:
            ValueError: If value is not a list of strings.
        """
        if not isinstance(val, list):
            _msg = f"{val} must be a list of strings."
            _msg += f" type {type(val)} found instead."
            raise ValueError(_msg)
        if not all(isinstance(v, str) for v in val):
            _msg = f"{val} must be a list of strings."
            _msg += f" Found types: {[type(v) for v in val]}."
            raise ValueError(_msg)
        self._depends = val

    def clear(self) -> None:
        """*clear()* Reset all attributes to default values.

        Resets all variable properties to their initial state.
        """
        # Reset base class attributes
        self._idx = -1
        self._sym = ""
        self._alias = ""
        self._fwk = "PHYSICAL"
        self.name = ""
        self.description = ""

        # Reset variable-specific attributes
        self._cat = "IN"
        self._dims = ""
        self._units = ""
        self._std_dims = None
        self._sym_exp = None
        self._dim_col = []
        self._min = None
        self._max = None
        self._mean = None
        self._std_units = ""
        self._std_min = None
        self._std_max = None
        self._std_mean = None
        self._step = None
        self._std_range = np.array([])
        self._dist_type = "uniform"
        self._dist_params = {}
        self._dist_func = None
        self.relevant = False

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
    def from_dict(cls, data: Dict[str, Any]) -> Variable:
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
