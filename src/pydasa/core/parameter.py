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

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import re
import numpy as np

# Import validation base classes
from src.pydasa.core.basic import Validation

# Import utility functions
from src.pydasa.utils.latex import latex_to_python

# Import configuration
# import the 'cfg' module to allow global variable edition
from src.pydasa.utils import config as cfg


@dataclass
class Variable(Validation):
    """**Variable** class for Dimensional Analysis in *PyDASA*.

    A comprehensive implementation that combines Parameter and Variable functionality
    for use in dimensional analysis calculations, simulations, and sensitivity analysis.

    Attributes:
        # Identification and Classification
        name (str): User-friendly name of the variable.
        description (str): Brief summary of the variable.
        _idx (int): Index/precedence in the dimensional matrix.
        _sym (str): Symbol representation (LaTeX or alphanumeric).
        _pyalias (str): Python-compatible alias for use in code.
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
    _step: float = 1e-3
    """Step size for simulations."""

    # :attr: _std_range
    _std_range: np.ndarray = field(default_factory=lambda: np.array([]))
    """Range array for analysis."""

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
        if not self._pyalias:
            self._pyalias = latex_to_python(self._sym)

        # Process dimensions if provided
        if self._dims:
            if not self._validate_exp(self._dims, cfg.WKNG_FDU_RE):
                _msg = f"Invalid Variable dimensions '{self.name}': {self._dims}. "
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
            bool: True if the dimensions are valid, False otherwise.
        """
        return bool(re.match(regex, exp))

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
        """
        # split the sympy expression into a list of dimensions
        dims_list = dims.split("* ")
        # set the default list of zeros with the FDU length
        col = [0] * len(cfg.WKNG_FDU_PREC_LT)
        for dim in dims_list:
            # match the exponent of the dimension
            _exp = int(re.search(cfg.WKNG_POW_RE, dim).group(0))
            # match the symbol of the dimension
            _sym = re.search(cfg.WKNG_FDU_SYM_RE, dim).group(0)
            # update the column with the exponent of the dimension
            col[cfg.WKNG_FDU_PREC_LT.index(_sym)] = _exp
        return col

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
        _working_lt = cfg.WKNG_FDU_PREC_LT
        if val is not None and not val.strip():
            raise ValueError("Dimensions cannot be empty.")

        # Process dimensions
        if val and not self._validate_exp(val, _working_lt):
            _msg = f"Invalid dimensions: {val}. "
            _msg += f"Check FDUs according to precedence: {_working_lt}"
            raise ValueError(_msg)
        # automatically prepare the dimensions for analysis
        self._prepare_dims()
        self._dims = val

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
            ValueError: If value is not a valid number.
            ValueError: If value is less than min.
        """
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError("Maximum val must be a number.")
        if val < self._min:
            _msg = f"Maximum {val} cannot be less"
            _msg = f" than minimum {self._min}."
            raise ValueError(_msg)
        self._max = val

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

        low = (self._min is not None and val < self._min)
        high = (self._max is not None and val > self._max)
        if low or high:
            _msg = f"Mean {val}. "
            _msg += f"must be between {self._min} and {self._max}."
            raise ValueError(_msg)
        self._mean = val

    # Value Ranges (Standardized Units)

    @property
    def std_units(self) -> Optional[str]:
        """*std_units* Get the standardized Unit of Measure.

        Returns:
            Optional[str]: standardized Unit of Measure.
        """
        return self._std_units

    @std_units.setter
    def std_units(self, val: Optional[str]) -> None:
        """*std_units* Sets the standardized Unit of Measure.

        Args:
            val (Optional[str]): standardized Unit of Measure.

        Raises:
            ValueError: If standardized units are empty.
        """
        if val is not None and not val.strip():
            _msg = "Standardized Unit of Measure cannot be empty."
            raise ValueError(_msg)
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

        if val > self._std_max:
            _msg = f"Standard minimum val {val} cannot be greater"
            _msg = f" than standard maximum val {self._std_max}."
            raise ValueError(_msg)

        # Update range if all values are available
        if all([self._std_min is not None,
                self._std_max is not None,
                self._step is not None]):
            self._std_range = np.arange(self._std_min,
                                        self._std_max,
                                        self._step)
        self._std_min = val

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
        if val < self._std_min:
            _msg = f"Standard maximum *Variable* {val} cannot be less"
            _msg = f" than standard minimum *Variable* {self._std_min}."
            raise ValueError(_msg)

        # Update range if all values are available
        if all([self._std_min is not None,
                self._std_max is not None,
                self._step is not None]):
            self._std_range = np.arange(self._std_min,
                                        self._std_max,
                                        self._step)
        self._std_max = val

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

        low = (self._std_min is not None and val < self._std_min)
        high = (self._std_max is not None and val > self._std_max)

        if low or high:
            _msg = f"Invalid standard mean value {val}. "
            _msg += f"Must be between {self._std_min} and {self._std_max}."
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
            raise ValueError("Standardized standard deviation must be a number")

        low = (self._std_min is not None and val < self._std_min)
        high = (self._std_max is not None and val > self._std_max)

        if low or high:
            _msg = f"Invalid standard deviation value {val}. "
            _msg += f"Must be between {self._std_min} and {self._std_max}."
            raise ValueError(_msg)
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

        if val >= self._std_max - self._std_min:
            _msg = f"Step {val} must be less than range"
            _msg += f" {self._std_max - self._std_min}."
            raise ValueError(_msg)

        # Update range if all values are available
        if all([self._std_min is not None,
                self._std_max is not None,
                self._step is not None]):
            self._std_range = np.arange(self._std_min,
                                        self._std_max,
                                        self._step)

        self._step = val

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

    def clear(self) -> None:
        """*clear()* Reset all attributes to default values.

        Resets all variable properties to their initial state.
        """
        # Reset base class attributes
        self._idx = -1
        self._sym = ""
        self._pyalias = ""
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
        self._step = 1e-3
        self._std_range = np.array([])
        self.relevant = False

    def to_dict(self) -> Dict[str, Any]:
        """*to_dict()* Convert variable to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation of variable.
        """
        return {
            "name": self.name,
            "description": self.description,
            "idx": self._idx,
            "sym": self._sym,
            "pyalias": self._pyalias,
            "fwk": self._fwk,
            "cat": self._cat,
            "dims": self._dims,
            "units": self._units,
            "min": self._min,
            "max": self._max,
            "mean": self._mean,
            "std_units": self._std_units,
            "std_min": self._std_min,
            "std_max": self._std_max,
            "std_mean": self._std_mean,
            "step": self._step,
            "relevant": self.relevant
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Variable:
        """*from_dict()* Create variable from dictionary representation.

        Args:
            data (Dict[str, Any]): Dictionary representation of variable.

        Returns:
            Variable: New variable instance.
        """
        return cls(**data)
