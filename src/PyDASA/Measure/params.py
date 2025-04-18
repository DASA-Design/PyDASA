# -*- coding: utf-8 -*-
"""
Module for representing *Parameters* and Variables in Dimensional Analysis for *PyDASA*.

*IMPORTANT:* Based on the theory from:

    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""

# native python modules
import re
from typing import Optional, List, Generic
from dataclasses import dataclass, field

# Third-party modules
import numpy as np

# custom modules
# generic error handling and type checking
from Src.PyDASA.Util.dflt import T
# import the 'cfg' module to allow global variable edition
from Src.PyDASA.Util import cfg

# checking custom modules
assert cfg
assert T


@dataclass
class Parameter(Generic[T]):
    """*Parameter* class represents a parameter in Dimensional Analysis for *PyDASA*.

    Args:
        Generic (T): Generic type for a Python data structure.

    Returns:
        Parameter: An object with the following attributes:
            - _idx (int): The Index of the Parameter.
            - _sym (str): The symbol of the Parameter.
            - _fwk (str): The framework of the Parameter. It can be one supported frameworks.
            - _cat (str): The category of the Parameter. It can be one of the following: `INPUT`, `OUTPUT`, or `CONTROL`.
            - _dims (str): The dimensions of the Parameter. It is a string with the FDU formula of the parameter. e.g.: [T^2*L^-1]
            - _std_dims (str): Standarized dimensional expression of the Parameter. It is a string with propper parenthesis and exponents. e.g.: [L^(-1)*T^(2)]
            - _sym_exp (str): The symbolic processed dimensional expression of the Parameter. It is a string suitable for Sympy processing. e.g.: [T**2*L**(-1)]
            - dim_col (List[int]): The dimensional column (list) of the Parameter. It is a list with the exponents of the dimensions in the parameter. e.g.: [2, -1]
            - _units (str): The Units of Measure of the Parameter. There are the dimensional Units the parameter was defined in. e.g.: `m/s`, `kg/m3`, etc.
            - name (str): User-friendly name of the Parameter.
            - description (str): Small summary of the Parameter.
            - relevant (bool): Flag indicating if the Parameter is relevant or not. It decides if the parameter is inside the Dimensional Matrix or not.
    """

    # Private attributes with validation logic
    # :attr: _idx
    _idx: int = -1
    """
    Unique identifier/index of the *Parameter*. It is a non-negative integer for the column order in the Dimensional Matrix.
    """

    # :attr: _sym
    _sym: str = ""
    """
    Symbol of the *Parameter*. It is a LaTeX or an alphanumeric string (preferably a single Latin or Greek letter). It is used for user-friendly representation of the instance.
    """

    # :attr: _fwk
    _fwk: str = "PHYSICAL"
    """
    Framework of the *Parameter* in the Dimensional Matrix. It must be the same as the FDU framework. Can be: `PHYSICAL`, `COMPUTATION`, `DIGITAL` or `CUSTOM`.
    """

    # :attr: _cat`
    _cat: str = "INPUT"
    """
    Category of the *Parameter* in the Dimensional Matrix. It can be one of the following: `INPUT`, `OUTPUT`, or `CONTROL` going in the main diagonal matrix, output vector, and residual matrix respectively.
    """

    # :attr: _dims
    _dims: str = ""
    """
    Dimensions of the *Parameter*. It is a regex with the FDU formula of the parameter. e.g.: [T^2*L^-1].
    """

    # :attr: _std_dims
    _std_dims: Optional[str] = None
    """
    Standarized Dimensional Expression of the *Parameter* for analysis. It is a string with propper parenthesis and exponents. e.g.: from [T^2*L^-1] to [L^(-1)*T^(2)].
    """

    # dimensional expression for the sympy regex procesor
    # :attr: _sym_exp
    _sym_exp: Optional[str] = None
    """
    Standardized Dimensional Expression of the *Parameter* for sympy processing. e.g.: from [T^2*L^-1] to [T**2*L**(-1)].
    """

    # list with the dimensions exponent as integers
    # :attr: _dim_col
    _dim_col: Optional[List[int]] = field(default_factory=list)
    """
    Dimensional Column (list) of the *Parameter* for analysis. It is a list with the exponents of the dimensions in the parameter. e.g.: from [T^2*L^-1] to [2, -1].
    """

    # :attr: _units
    _units: str = ""
    """
    Units of Measure of the *Parameter*. It is a string with the dimensional Units of Measure the parameter was defined in. e.g.: `m/s`, `kg/m3`, bit/s, etc.
    """

    # Public attributes
    # :attr: name
    name: str = ""
    """
    User-friendly name of the *Parameter*.
    """

    # :attr: description
    description: str = ""
    """
    Brief summary of the *Parameter*.
    """

    # :attr: relevant
    relevant: bool = False
    """
    Flag indicating if the *Parameter* is relevant or not. It is used to identify whether the parameter is inside the dimensional matrix or not.
    """

    def __post_init__(self) -> None:
        """*__post_init__()* Initializes the Parameter and validates its dimensions.

        Raises:
            ValueError: If the dimensions are not valid according to the regex pattern.
        """
        # check for valid dimensions
        if self._dims != "":
            if not self._validate(self.dims, cfg.WKNG_FDU_REGEX):
                _msg = f"Invalid dimensions in Parameter '{self.name}' "
                _msg += f"Check FDUs in: {self._dims}."
                _msg += " in acordance with the FDU precedence list:, "
                _msg += f"'{cfg.WKNG_FDU_PREC_LT}'/"
                raise ValueError(_msg)
            self._prepare_dims()

        if self.description != "":
            self.description = self.description.capitalize()

    def _prepare_dims(self) -> None:
        """*_prepare_dims()* Prepares the dimensions for analysis.
        """
        self._std_dims = self._standarize_dims(self._dims)
        self._std_dims = self._sort_dims(self._std_dims)
        self._sym_exp = self._setup_sym(self._std_dims)
        self._dim_col = self._setup_col(self._sym_exp)

    def _validate(self, dims: str, regex: str) -> bool:
        """*_validate()* Validates the dimensions using a regex pattern.

        Args:
            dims (str): Dimensions of the parameter. It is a string with the FDU formula of the parameter. e.g.: [T^2*L^-1]

        Returns:
            bool: True if the dimensions are valid, False otherwise.
        """
        return bool(re.match(regex, dims))

    def _standarize_dims(self, dims: str) -> str:
        """*_standarize_dims()* Standardizes the dimensions by adding parentheses and default exponents.

        Args:
            dims (str): Dimensions of the *Parameter*. e.g.: [T^2*L^-1]

        Returns:
            str: Standarized dimensions of the *Parameter*. e.g.: [T^(2)*L^(-1)]
        """
        # add parentheses to powers in dimensions
        _regex_pat = re.compile(cfg.WKNG_POW_REGEX)
        _dims = _regex_pat.sub(lambda m: f"({m.group(0)})", dims)
        # add ^1 to * and / operations in dimensions
        _regex_pat = re.compile(cfg.WKNG_NO_POW_REGEX)
        _dims = _regex_pat.sub(lambda m: f"{m.group(0)}^(1)", _dims)
        # return standarized dimensions
        return _dims

    def _sort_dims(self, dims: str) -> str:
        """*_sort_dims()* Sorts dimensions based on the FDU precedence list.

        NOTE: This function is crucial for result consistency.

        Args:
            dims (str): Dimensions of the *Parameter*. e.g.: [T^2*L^-1]

        Returns:
            str: Sorted dimensions of the *Parameter*.. e.g.: [L^(-1)*T^(2)]
        """
        # TODO maybe add a custom sort?
        # TODO move '*' as global operator to cfg module?
        # create a list of the dimensions
        _dims_lt = dims.split("*")
        # sort the dimensions in the dimensional precedence order
        _dims_lt.sort(key=lambda x: cfg.WKNG_FDU_PREC_LT.index(x[0]))
        # recreate the dimensions string
        _dims = "*".join(_dims_lt)
        return _dims

    def _setup_sym(self, dims: str) -> str:
        """*_setup_sym()* Converts dimensions to a Sympy-compatible format.

        Args:
            dims (str): Dimensions of the *Parameter*. e.g.: [T^2*L^-1]

        Returns:
            str: Sympy-compatible dimensions of the *Parameter*. e.g.: [T**2* L**(-1)]
        """
        # TODO move '*' and '* ' as global operator to cfg module?
        # TODO do I use also regex for this?
        # replace '*' with '* ' for sympy processing
        # # replace '^' with '**' for sympy processing
        return dims.replace("*", "* ").replace("^", "**")

    def _setup_col(self, dims: str) -> List[int]:
        """*_setup_col()* Generates the dimensional column (list of exponents) in the Dimensional Matrix.

        Args:
            dims (str): Standarized dimensions of the *Parameter*. e.g.: [T^(2)*L^(-1)]

        Returns:
            List[int]: Exponents with the dimensions of the *Parameter*. e.g.: [2, -1]
        """
        # split the sympy expression into a list of dimensions
        dims_list = dims.split("* ")
        # set the default list of zeros with the FDU length
        col = [0] * len(cfg.WKNG_FDU_PREC_LT)
        for dim in dims_list:
            # match the exponent of the dimension
            exponent = int(re.search(cfg.WKNG_POW_REGEX, dim).group(0))
            # match the symbol of the dimension
            symbol = re.search(cfg.WKNG_FDU_SYM_REGEX, dim).group(0)
            # update the column with the exponent of the dimension
            col[cfg.WKNG_FDU_PREC_LT.index(symbol)] = exponent
        return col

    @property
    def idx(self) -> int:
        """*idx* Get the index of the *Parameter* in the dimensional matrix.

        Returns:
            int: Index of the *Parameter*.
        """
        return self._idx

    @idx.setter
    def idx(self, val: int) -> None:
        """*idx* Sets the index of the *Parameter* in the dimensional matrix. It must be an integer.

        Args:
            val (int): Index of the *Parameter*.

        Raises:
            ValueError: If the Index is not a non-negative integer.
        """
        if not isinstance(val, int) or val < 0:
            _msg = "Precedence must be a non-negative integer. "
            _msg += f"Provided: {val}"
            raise ValueError(_msg)
        self._idx = val

    @property
    def sym(self) -> str:
        """*sym* Get the symbol of the *Parameter*.

        Returns:
            str: Symbol of the *Parameter*.
        """
        return self._sym

    @sym.setter
    def sym(self, val: str) -> None:
        """*sym* Sets the symbol of *Parameter*. It must be alphanumeric.

        Args:
            val (str): Symbol of the *Parameter*.

        Raises:
            ValueError: If the symbol is not alphanumeric.
        """
        # Regular expression to match valid LaTeX strings or alphanumeric strings
        if not (val.isalnum() or re.match(cfg.LATEX_REGEX, val)):
            _msg = "Symbol must be alphanumeric or a valid LaTeX string. "
            _msg += f"Provided: '{val}' "
            _msg += "Examples: 'V', 'd', '\\Pi_{0}', '\\rho'."
            raise ValueError(_msg)
        self._sym = val

    @property
    def fwk(self) -> str:
        """*fwk* Get the framework of the *Parameter*.

        Returns:
            str: Framework of the *Parameter*.
        """
        return self._fwk

    @fwk.setter
    def fwk(self, val: str) -> None:
        """*fwk* Sets the framework of the *Parameter*. It must be one of the allowed values. The allowed values are: `PHYSICAL`, `COMPUTATION`, `DIGITAL` or `CUSTOM`.

        Args:
            val (str): Framework of the *Parameter*. Must be the same as the FDU framework.

        Raises:
            ValueError: If the framework is not one of the allowed values.
        """
        if val not in cfg.FDU_FWK_DT.keys():
            _msg = f"Invalid framework: {val}. "
            _msg += "Framework must be one of the following: "
            _msg += f"{', '.join(cfg.FDU_FWK_DT.keys())}."
            raise ValueError(_msg)
        self._fwk = val

    @property
    def cat(self) -> str:
        """*cat* Get the category of the *Parameter*.

        Returns:
            str: Category of the *Parameter*.
        """
        return self._cat

    @cat.setter
    def cat(self, val: str) -> None:
        """*cat* Sets the category of the *Parameter*. It must be one of the allowed values. The allowed values are: `INPUT`, `OUTPUT`, or `CONTROL`.

        Args:
            val (str): Category of the *Parameter*.

        Raises:
            ValueError: If the category is not one of the allowed values.
        """
        if val.upper() not in cfg.PARAMS_CAT_DT.keys():
            _msg = f"Invalid category: {val}. "
            _msg += "Category must be one of the following: "
            _msg += f"{', '.join(cfg.PARAMS_CAT_DT.keys())}."
            raise ValueError(_msg)
        self._cat = val.upper()

    @property
    def dims(self) -> str:
        """*dims* Get the dimensions of the *Parameter*.

        Returns:
            str: Dimensions of the *Parameter*. e.g.: [T^2*L^-1]
        """
        return self._dims

    @dims.setter
    def dims(self, val: str) -> None:
        """*dims* Sets the dimensions of the *Parameter*.

        Args:
            val (str): Dimensions of the *Parameter*. e.g.: [T^2*L^-1]

        Raises:
            ValueError: If the string is empty.
        """
        if val is not None and not val.strip():
            raise ValueError("Dimensions cannot be empty.")
        self._dims = val
        # automatically prepare the dimensions for analysis
        self._prepare_dims()

    @property
    def std_dims(self) -> Optional[str]:
        """*std_dims* Get the standarized dimensions of the *Parameter*.

        Returns:
            Optional[str]: Dimensional expression of the *Parameter*. e.g.: [L^(-1)*T^(2)]
        """
        return self._std_dims

    @std_dims.setter
    def std_dims(self, val: str) -> None:
        """*std_dims* Sets the standarized dimensions of the *Parameter*.

        Args:
            val (str): Dimensional expression of the *Parameter*. e.g.: [L^(-1)*T^(2)]

        Raises:
            ValueError: If the string is empty.
        """
        if val is not None and not val.strip():
            raise ValueError("Dimensional expression cannot be empty.")
        self._std_dims = val

    @property
    def sym_exp(self) -> Optional[str]:
        """*sym_exp* Get the symbolic processed dimensional expression of the *Parameter*. It is suitable for Sympy processing.

        Returns:
            Optional[str]: Dimensional expression of the *Parameter* suitable for Sympy processing. e.g.: [T**2*L**(-1)]
        """
        return self._sym_exp

    @sym_exp.setter
    def sym_exp(self, val: str) -> None:
        """*sym_exp* Sets the symbolic processed dimensional expression of the *Parameter*. It is suitable for Sympy processing.

        Args:
            val (str): Dimensional expression of the *Parameter* suitable for Sympy processing. e.g.: [T**2*L**(-1)]

        Raises:
            ValueError: If the string is empty.
        """
        if val is not None and not val.strip():
            raise ValueError("Dimensional expression cannot be empty.")
        self._sym_exp = val

    @property
    def dim_col(self) -> Optional[List[int]]:
        """*dim_col* Get the dimensional column (list) of the *Parameter*.

        Returns:
            Optional[List[int]]: List with the exponents of the dimensions in the parameter. e.g.: [2, -1]
        """
        return self._dim_col

    @dim_col.setter
    def dim_col(self, val: List[int]) -> None:
        """*dim_col* Sets the dimensional column (list) of the *Parameter*.

        Args:
            val (List[int]): List with the exponents of the dimensions in the parameter. i.e..: [2, -1]

        Raises:
            ValueError: if the val is not a list of integers.
        """
        if val is not None and not isinstance(val, list):
            raise ValueError("Exponents list must be a list of integers.")
        self._dim_col = val

    @property
    def units(self) -> str:
        """*units* Get the Units of Measure of the *Parameter*.

        Returns:
            str: Units of measure of the *Parameter*. e.g.: `m/s`, `kg/m3`, etc.
        """
        return self._units

    @units.setter
    def units(self, val: str) -> None:
        """*units* Sets the Units of Measure of the *Parameter*. It must be a non-empty string.

        Args:
            val (str): Units of measure of the *Parameter*. i.e `m/s`, `kg/m3`, etc.

        Raises:
            ValueError: If the string is empty.
        """
        if val is not None and not val.strip():
            raise ValueError("Unit of Measure cannot be empty.")
        self._units = val

    def clear(self) -> None:
        """*clear()* Resets all attributes to their default values in the *Parameter* object.
        """
        self._idx = -1
        self._sym = ""
        self._fwk = "PHYSICAL"
        self._cat = "INPUT"
        self._dims = ""
        self._std_dims = None
        self._sym_exp = None
        self._dim_col = None
        self._units = ""
        self.name = ""
        self.description = ""
        self.relevant = False

    def __str__(self) -> str:
        """*__str__()* returns a string representation of the *Parameter* object.

        Returns:
            str: String representation of the *Parameter* object.
        """
        _attr_lt = []
        for attr, val in vars(self).items():
            # Skip private attributes starting with "__"
            if attr.startswith("__"):
                continue
            # Format attribute name and val
            _attr_name = attr.lstrip("_")
            _attr_lt.append(f"{_attr_name}={repr(val)}")
        # Format the string representation of the ArrayList class and its attributes
        _str = f"{self.__class__.__name__}({', '.join(_attr_lt)})"
        return _str

    def __repr__(self) -> str:
        """*__repr__* returns a string representation of the *Parameter* object.

        Returns:
            str: String representation of the *Parameter* object.
        """
        return self.__str__()


@dataclass
class Variable(Parameter[T]):
    """**Variable** Extends *Parameter* with additional attributes for sensitivity analysis and simulations.

    Args:
        Generic (T): Generic type for a Python data structure.
        Parameter (Parameter[T]): *PyDASA* *Parameter* class for Dimensional Analysis.

    Returns:
        Variable: A *Variable* object with the following attributes:
            - `_min`: The minimum range of the *Variable*. It is a float value.
            - `_max`: The maximum range of the *Variable*. It is a float value.
            - `_std_units`: The standardized Unit of Measure of the *Variable*. It is a string with the dimensional Units of Measure. e.g.: `m/s`, `kg/m3`, etc.
            - `_std_min`: The standardized minimum range of the *Variable*, after converting Units of Measure. It is a float value.
            - `_std_max`: The standardized maximum range of the *Variable*, after converting Units of Measure. It is a float value.
            - `_std_step`: The step value of the *Variable*. It is a very small float value. It is used for sensitivity analysis and simulations.
    """

    # Private attributes with validation logic
    # :attr: _min
    _min: Optional[float] = None
    """
    Minimum range of the *Variable*.
    """

    # :attr: _max
    _max: Optional[float] = None
    """
    Maximum range of the *Variable*.
    """

    # :attr: _std_units
    _std_units: Optional[str] = ""
    """
    Standarized Unit of Measure of the *Variable*. It is a string with the standarized dimensional Units of Measure. e.g from `km/h`, kByte/s` to `m/s`, `bit/s`.
    """

    # :attr: _std_min
    _std_min: Optional[float] = None
    """
    Standardized minimum range of the *Variable* after unit convertion.
    """

    # :attr: _std_max
    _std_max: Optional[float] = None
    """
    Standardized maximum varangelue of the *Variable* after unit convertion.
    """

    # :attr: _std_step
    _std_step: Optional[float] = 1 / 1000
    """
    Step of the *Variable* range. It is a very small float used for sensitivity analysis and simulations.
    """

    # :attr: _std_rng
    _std_rng: Optional[np.ndarray] = np.array([])
    """
    Range of the *Variable*. It is a numpy array with the data range of the variable used for sensitivity analysis and simulations.
    """

    def __post_init__(self):
        super().__post_init__()
        if all((self._std_min, self._std_max, self._std_step)):
            # set the data to the range of values between std_min and std_max with a step of std_step
            self.range = np.arange(self.std_min,
                                   self.std_max,
                                   self.std_step)

    @property
    def min(self) -> Optional[float]:
        """*min* Get the minimum range of the *Variable*.

        Returns:
            Optional[float]: minimum range of the *Variable*.
        """
        return self._min

    @min.setter
    def min(self, val: Optional[float]) -> None:
        """*min* Sets the minimum range of the *Variable*.

        Args:
            val (Optional[float]): Minimum range of the *Variable*.

        Raises:
            ValueError: If the minimum range is not a number.
            ValueError: If the minimum is greater than the maximum range.
        """
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError("Minimum range must be a number.")
        if val > self._max:
            _msg = f"Minimum range {val} cannot be greater"
            _msg = f" than maximum range {self._max}."
            raise ValueError(_msg)
        self._min = val

    @property
    def max(self) -> Optional[float]:
        """*max* Get the maximum range of the *Variable*.

        Returns:
            Optional[float]: maximum range of the *Variable*.
        """
        return self._max

    @max.setter
    def max(self, val: Optional[float]) -> None:
        """*max* Sets the maximum range of the *Variable*.

        Args:
            val (Optional[float]): maximum range of the *Variable*.

        Raises:
            ValueError: If the maximum range is not a number.
            ValueError: If the maximum is less than the minimum range.
        """
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError("Maximum val must be a number.")
        if val < self._min:
            _msg = f"Maximum range {val} cannot be less"
            _msg = f" than minimum range {self._min}."
            raise ValueError(_msg)
        self._max = val

    @property
    def std_units(self) -> Optional[str]:
        """*std_units* Get the standardized Unit of Measure of the *Variable*.

        Returns:
            Optional[str]: standardized Unit of Measure of the *Variable*.
        """
        return self._std_units

    @std_units.setter
    def std_units(self, val: Optional[str]) -> None:
        """*std_units* Sets the standardized Unit of Measure of the *Variable*. It must be a non-empty string.

        Args:
            val (Optional[str]): standardized Unit of Measure of the *Variable*.

        Raises:
            ValueError: If the string is empty.
        """
        if val is not None and not val.strip():
            raise ValueError("Standard Unit of Measure cannot be empty.")
        self._std_units = val

    @property
    def std_min(self) -> Optional[float]:
        """*std_min* Get the standardized minimum range of the *Variable*.

        Returns:
            Optional[float]: standardized minimum range of the *Variable*.
        """
        return self._std_min

    @std_min.setter
    def std_min(self, val: Optional[float]) -> None:
        """*std_min* Sets the standardized minimum range of the *Variable*.

        Args:
            val (Optional[float]): standardized minimum range of the *Variable*.

        Raises:
            ValueError: If the minimum range is not a number.
            ValueError: If the minimum is greater than the maximum range.
        """
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError("Standard minimum val must be a number.")
        if val > self._std_max:
            _msg = f"Standard minimum val {val} cannot be greater"
            _msg = f" than standard maximum val {self._std_max}."
            raise ValueError(_msg)
        self._std_min = val

    @property
    def std_max(self) -> Optional[float]:
        """*std_max* Get the standardized maximum range of the *Variable*.

        Returns:
            Optional[float]: standardized maximum range of the *Variable*.
        """
        return self._std_max

    @std_max.setter
    def std_max(self, val: Optional[float]) -> None:
        """*std_max* Sets the standardized maximum range of the *Variable*.

        Raises:
            ValueError: If the maximum range is not a number.
            ValueError: If the maximum is less than the minimum range.
        """
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError("Standard maximum val must be a number.")
        if val < self._std_min:
            _msg = f"Standard maximum *Variable* {val} cannot be less"
            _msg = f" than standard minimum *Variable* {self._std_min}."
            raise ValueError(_msg)
        self._std_max = val

    @property
    def std_step(self) -> Optional[float]:
        """*std_step* Get the standarized step of the *Variable*.
        It is used for sensitivity analysis and simulations.

        Returns:
            Optional[float]: standarized step of the *Variable*.
        """
        return self._std_step

    @std_step.setter
    def std_step(self, val: Optional[float]) -> None:
        """*std_step* Sets the standarized step of the *Variable*. It must be a number.

        Args:
            val (Optional[float]): standarized step of the *Variable*.

        Raises:
            ValueError: If the step is not a number.
            ValueError: If the step is zero.
            ValueError: If the step is greater than or equal to the range of values.
        """
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError("Step must be a number.")
        if val == 0:
            raise ValueError("Step cannot be zero.")
        if val >= self._std_max - self._std_min:
            _msg = f"Step {val} cannot be greater than or equal to"
            _msg = f" the range of values {self._std_max - self._std_min}."
            _msg += f"between {self._std_min} and {self._std_max}."
            raise ValueError(_msg)
        self._std_step = val

    @property
    def std_rng(self) -> np.ndarray:
        """*std_rng* Get the data array for the Variable.

        Returns:
            np.ndarray: Data array for the Variable.
        """
        return self._std_rng

    @std_rng.setter
    def std_rng(self, val: Optional[np.ndarray]) -> None:
        """*std_rng* Set the data array for the Variable." if `val` is None, it will be set to the range of values between `std_min` and `std_max` with a step of `std_step`.

        Args:
            val (Optional[np.ndarray]): Data array for the Variable.

        Raises:
            ValueError: if the data range is not a numpy array.
        """
        if val is None:
            self._std_rng = np.arange(self.std_min,
                                      self.std_max,
                                      self.std_step)
        elif not isinstance(val, np.ndarray):
            _msg = "Invalid data type. "
            _msg += "Range must be a numpy array."
            _msg += f" Provided: {type(val)}."
            raise ValueError(_msg)
        else:
            self._std_rng = val

    def clear(self) -> None:
        """*clear()* Resets all attributes to their default values in the *Variable* object. It extends from *Parameter* class.
        """
        super().clear()
        self._min = None
        self._max = None
        self._std_units = ""
        self._std_min = None
        self._std_max = None
        self._std_step = 1 / 1000
        self._std_rng = np.array([])

    def __str__(self) -> str:
        """*__str__()* Returns a string representation of the *Variable*. It extends from *Parameter* class.

        Returns:
            str: string representation of the *Variable*.
        """
        _str = super().__str__()
        return _str

    def __repr__(self) -> str:
        """*__repr__()* Returns a string representation of the *Variable*. It extends from *Parameter* class.

        Returns:
            str: string representation of the *Variable*.
        """
        _str = super().__repr__()
        return _str
