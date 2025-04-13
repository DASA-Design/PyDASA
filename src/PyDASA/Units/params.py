# -*- coding: utf-8 -*-
"""
Module for representing *Parameters* and Variables in Dimensional Analysis for *PyDASA*.

*IMPORTANT:* Based on the theory from:

    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""

# native python modules
import re
# import modules for defining Parameter and Variable types
from typing import Optional, List, Generic
# import dataclass for class attributes and validations
from dataclasses import dataclass, field

# custom modules
# generic error handling and type checking
from Src.PyDASA.Utils.err import error_handler as error
from Src.PyDASA.Utils.dflt import T

# importing PyDASA's custom regex for FDU
# using the 'as' allows shared variable edition
from Src.PyDASA.Utils import cfg as config

# importing the FDU class for creating the FDU object
# TODO do i need this import in future version?
# from Src.PyDASA.Units.fdu import FDU

# checking custom modules
assert error
assert config
assert T


@dataclass
class Parameter(Generic[T]):
    """*Parameter* class for creating a *Parameter* in *PyDASA*. The parameters are use in Dimensional Analysis to create Dimensionless Coefficients (DN).

    Args:
        Generic (T): Generic type for a Python data structure.

    Returns:
        Parameter: A *Parameter* object with the following attributes:
            - `_idx`: The ID of the Parameter.
            - `_sym`: The symbol of the Parameter.
            - `_fwk`: The framework of the Parameter. It can be one of the following: `PHYSICAL`, `DIGITAL`, or `CUSTOM`.
            - `_dims`: The dimensions of the Parameter.
            - `_units`: The Units of Measure of the Parameter.
            - `_cat`: The category of the Parameter. It can be one of the following: `INPUT`, `OUTPUT`, or `CONTROL`.
            - `name`: The name of the Parameter.
            - `description`: The description of the Parameter.
            - `relevance`: The relevance of the Parameter. It can be `True` or `False`.
    """

    # Private attributes with validation logic
    # :attr: _idx
    _idx: int = -1
    """
    Index of the *Parameter*. It is the unique integer for the column's order in the dimensional matrix.
    """

    # Symbol of the FDU
    # :attr: _sym
    _sym: str = ""
    """
    Symbol of the *Parameter*. It must be alphanumeric (preferably a single character + Latin or Greek letter). Useful for user-friendly representation of the instance.
    """

    # Working framework of the FDU
    # :attr: _fwk
    _fwk: str = "PHYSICAL"
    """
    Framework the *Parameter* follows in accordance with the FDU. It can be one of the following: `PHYSICAL`, `DIGITAL`, or `CUSTOM`. Parameters and FDUs must be in the same framework.
    """

    # Category of the parameter, can be: `INPUT`, `OUTPUT`, or `CONTROL`
    # :attr: _cat`
    _cat: str = "INPUT"
    """
    Category of the *Parameter*. It is a string specifing the parameter's behavior in the dimensional matrix. It can be one of the following: `INPUT`, `OUTPUT`, or `CONTROL` for fundamental parameters, desired parameters, or in-phenomena parameters respectively.
    """

    # user input dimensional expression for the parameter
    # :attr: _dims
    _dims: str = ""
    """
    Dimensional representation of the *Parameter*. It is a user-defined string with the FDU formula of the parameter. i.e.: [T^2*L^-1] or [T^2*L].
"""

    # user dimensional expression
    # :attr: _dim_exp
    _dim_exp: Optional[str] = None
    """
    Dimensional Expression of the *Parameter* for analysis. Iti is a standarized string with propper parenthesis and exponents. It is used to calculate the dimensional matrix columns. i.e.: from [T^2*L^-1] to [L^(-1)*T^(2)].
    """

    # dimensional expression for the sympy regex procesor
    # :attr: _sym_exp
    _sym_exp: Optional[str] = None
    """
    Symbolic Dimensional Expression of the *Parameter* for analysis. It is a string suitable for Sympy processing. It is used to calculate the dimensional matrix columns. i.e.: from [T^2*L^-1] to [T**2*L**(-1)].
    """

    # list with the dimensions exponent as integers
    # :attr: _dim_col
    _dim_col: Optional[List[int]] = field(default_factory=list)
    """
    Dimensional Column (list) of the *Parameter* for analysis. It is a list of integers with the exponents of the dimensions in the parameter. It is the actual column of the dimensional matrix. i.e.: from [T^2*L^-1] to [2, -1].
    """

    # public attributes
    # :attr: _units
    _units: str = ""
    """
    Units of Meassure of the *Parameter*. It is a user-defined string with the original dimensional Units of Measure parameter was defined in. i.e.: `m/s`, `kg/m3`, bit/s, etc.
    """

    # Public attributes
    # :attr: name
    name: str = ""
    """
    Name of the *Parameter*. User-friendly name of the parameter.
    """

    # :attr: description
    description: str = ""
    """
    Description of the *Parameter*. It is a small summary of the parameter.
    """

    # :attr: _relevance
    relevance: bool = False
    """
    Boolean value indicating if the *Parameter* is relevant or not. It is used to identify whether the parameter is inside the main dimensional matrix or not.
    """

    def __post_init__(self) -> None:
        """*__post_init__* method to initialize the *Parameter* object. It is called after the object is created. It validates the dimensions and sets up the dimensional expression, symbolic expression, and dimensional column.

        Raises:
            ValueError: error if the dimensions string doen't follow the FDU regex pattern.
        """
        # check for valid dimensions
        if self._dims != "":
            if not self._validate(self.dims, config.WKNG_FDU_REGEX):
                _msg = f"Invalid dimensions in Parameter '{self.name}' "
                _msg += f"Check FDUs in: {self._dims}."
                _msg += " in acordance with the FDU precedence list:, "
                _msg += f"'{config.WKNG_FDU_PREC_LT}'/"
                raise ValueError(_msg)

            # prepare the parameter for dimensional analysis
            self._prep_dims()

        # if description is not empty, capitalize it
        if self.description != "":
            self.description = self.description.capitalize()

    def _prep_dims(self) -> None:
        """*_prep_dims* prepares the dimensions of the *Parameter* for dimensional analysis. It validates the dimensions and sets up the dimensional expression, symbolic expression, and dimensional column.
        """
        # set up dimensions in uppercase
        self._dim_exp = self._std_dims(self._dims)

        # sort dimensions in the dimensional precedence order
        self._dim_exp = self._sort_dims(self._dim_exp)

        # set up expression for sympy
        self._sym_exp = self._set_sym(self._dim_exp)

        # setup dimension pow list for dimensional analysis
        self._dim_col = self._set_col(self._sym_exp)

    def _validate(self, dims: str, regex: str) -> bool:
        """*_validate* validates the dimensions of the *Parameter*.

        Args:
            dims (str): Dimensions of the parameter. It is a string with the FDU formula of the parameter. i.e.: [T^2*L^-1]

        Returns:
            bool: True if the dimensions are valid, False otherwise.
        """
        # check if the dimensions are valid
        _valid = bool(re.match(regex, dims))
        return _valid

    def _findall(self, sym: str, pattern: str) -> str:
        """*_findall* finds the pattern in the standarize symbolic string.

        Args:
            sym (str): Standarized symbolic string to search for the pattern.
            pattern (str): Regex pattern to match the dimensional exponents against the symbolic string.

        Returns:
            str: Matched pattern or None if no match is found.
        """
        find = re.compile(pattern)
        matches = find.findall(sym)
        if matches:
            return matches
        else:
            return None

    def _std_dims(self, dims: str) -> str:
        """*_std_dims* standarizes the dimensions of the *Parameter*. It adds parentheses to the powers, and adds ^1 to the * operations.

        Args:
            dims (str): Dimensions of the *Parameter*. It is a string with the FDU formula of the parameter. i.e.: [T^2*L^-1]

        Returns:
            str: Standarized dimensions of the *Parameter*. It is a string with the FDU formula of the parameter. i.e.: [T^(2)*L^(-1)]
        """
        # add parentheses to powers in dimensions
        _regex_pat = re.compile(config.WKNG_POW_REGEX)
        _dims = _regex_pat.sub(lambda m: f"({m.group(0)})", dims)
        # add ^1 to * and / operations in dimensions
        _regex_pat = re.compile(config.WKNG_NO_POW_REGEX)
        _dims = _regex_pat.sub(lambda m: f"{m.group(0)}^(1)", _dims)
        # return standarized dimensions
        return _dims

    def _sort_dims(self, dims: str) -> str:
        """*_sort_dims* sorts the dimensions of the *Parameter* in the dimensional precedence order. Crucrial to create a consistent dimensional matrix.

        Args:
            dims (str): Dimensions of the *Parameter*. It is a string with the FDU formula of the parameter. i.e.: [T^2*L^-1]

        Returns:
            str: Sorted dimensions of the *Parameter*. It is a string with the FDU formula of the parameter. i.e.: [L^(-1)*T^(2)]
        """
        # TODO maybe add a custom sort?
        # TODO move '*' as global operator to cfg module?
        # create a list of the dimensions
        _dims_lt = dims.split("*")
        # sort the dimensions in the dimensional precedence order
        _dims_lt.sort(key=lambda x: config.WKNG_FDU_PREC_LT.index(x[0]))
        # recreate the dimensions string
        _dims = "*".join(_dims_lt)
        return _dims

    def _set_sym(self, dims: str) -> str:
        """*_set_sym* sets the dimensional expression special characters such as '^' and '*' to '**' and '* ' respectively. for sympy processing.

        Args:
            dims (str): Dimensions of the *Parameter*. It is a string with the FDU formula of the parameter. i.e.: [T^2*L^-1]

        Returns:
            str: Symbolic dimensions of the *Parameter*. It is a string with the FDU formula of the parameter. i.e.: [T**2* L**(-1)]
        """
        # TODO move '*' and '* ' as global operator to cfg module?
        # TODO do I use also regex for this?
        # replace '*' with '* ' for sympy processing
        _dims = dims.replace("*", "* ")
        # replace '^' with '**' for sympy processing
        _dims = _dims.replace("^", "**")
        return _dims

    def _set_col(self, dims: str) -> List[int]:
        """*_set_col* sets the dimensional column of the *Parameter*. It is a list of integers with the exponents of the dimensions of the Parameter. It is used to create the dimensional matrix.

        Args:
            dims (str): Standarized dimensions of the *Parameter*. It is a string with the FDU formula of the parameter. i.e.: [T^(2)*L^(-1)]

        Returns:
            List[int]: list of integers with the exponents of the dimensions of the *Parameter*. It is used to create the dimensional matrix. i.e.: [2, -1]
        """
        # TODO check this algorithm for improvement
        # split the sympy expression into a list of dimensions
        _dims_lt = dims.split("* ")
        # set the default list of zeros with the FDU length
        _dimensional_col = len(config.WKNG_FDU_PREC_LT) * [0]
        # working vars
        i = 0
        while i < len(_dims_lt):
            # get the dimension
            _t_sym = _dims_lt[i]
            # match the exponent of the dimension
            _t_pow = self._findall(_t_sym,
                                   config.WKNG_POW_REGEX)
            # find the fdu and its index
            _t_dim = self._findall(_t_sym,
                                   config.WKNG_FDU_SYM_REGEX)
            _t_idx = config.WKNG_FDU_PREC_LT.index(_t_dim[0])
            # update the dimension column with the exponent of the dimension
            _dimensional_col[_t_idx] = int(_t_pow[0])
            # increment the index
            i += 1
        # return the list of powers
        return _dimensional_col

    @property
    def idx(self) -> str:
        """*idx* property to get *Parameter's* index in the dimensional matrix.

        Returns:
            str: ID of the *Parameter*.
        """
        return self._idx

    @idx.setter
    def idx(self, value: str) -> None:
        """*idx* property to set the *Parameter's* index in the dimensional matrix. It must be alphanumeric.

        Args:
            value (str): ID of the *Parameter*.

        Raises:
            ValueError: error if the Index is not an integer.
        """
        if not isinstance(value, int):
            _msg = "Index must be an integer, "
            _msg += f"Provided type: {type(value)}"
            raise ValueError(_msg)
        self._idx = value

    @property
    def sym(self) -> str:
        """*sym* property to get the symbol of the *Parameter*.

        Returns:
            str: Symbol of the *Parameter*. It is a string with the FDU formula of the parameter. i.e.: V, d, D, m, Q, \\rho, etc.
        """
        return self._sym

    @sym.setter
    def sym(self, value: str) -> None:
        """*sym* property to set the symbol of *Parameter*. It must be alphanumeric (preferably a single character, a Latin or Greek letter).

        Args:
            value (str): Symbol of the *Parameter*. . i.e.: V, d, D, m, Q, \\rho, etc.

        Raises:
            ValueError: error if the symbol is not alphanumeric.
        """
        if not value.isalnum():
            _msg = "Symbol must be alphanumeric. "
            _msg += f"Provided: {value}"
            _msg += "Preferably a Latin or Greek letter."
            raise ValueError(_msg)
        self._sym = value

    @property
    def fwk(self) -> str:
        """*fwk* property to get the framework of the *Parameter*.

        Returns:
            str: Framework of the *Parameter*. It can be one of the following: `PHYSICAL`, `DIGITAL`, or `CUSTOM`.
        """
        return self._fwk

    @fwk.setter
    def fwk(self, value: str) -> None:
        """*fwk* property of the framework of the *Parameter*. It must be one of the following: `PHYSICAL`, `DIGITAL`, or `CUSTOM`.

        Args:
            value (str): Framework of the *Parameter*. Must be the same as the FDU framework.

        Raises:
            ValueError: If the framework is not one of the allowed values.
        """
        if value not in config.FDU_FWK_DT.keys():
            _msg = f"Invalid framework: {value}. "
            _msg += "Framework must be one of the following: "
            _msg += f"{', '.join(config.FDU_FWK_DT.keys())}."
            raise ValueError(_msg)
        self._fwk = value

    @property
    def cat(self) -> str:
        """*cat* property to get the category of the *Parameter*.

        Returns:
            str: Category of the *Parameter*.
        """
        return self._cat

    @cat.setter
    def cat(self, value: str) -> None:
        """*cat* property to set the category of the *Parameter*.

        Args:
            value (str): Category of the *Parameter*. It can must one of the following: `INPUT`, `OUTPUT`, or `CONTROL`.

        Raises:
            ValueError: error if the category is not one of the allowed values.
        """
        if value.upper() not in config.PARAMS_FWK_DT.keys():
            _msg = f"Invalid category: {value}. "
            _msg += "Category must be one of the following: "
            _msg += f"{', '.join(config.PARAMS_FWK_DT.keys())}."
            raise ValueError(_msg)
        self._cat = value.upper()

    @property
    def dims(self) -> str:
        """*dims* property to get the dimensions of the *Parameter*. It is a string with the FDU (Fundamental Dimensional Unit) of the parameter.

        Returns:
            str: Dimensions of the *Parameter*. i.e.: [T^2*L^-1]
        """
        return self._dims

    @dims.setter
    def dims(self, value: str) -> None:
        """*dims* property to set the dimensions of the *Parameter*. It is a string with the FDU (Fundamental Dimensional Unit) of the parameter.

        Args:
            value (str): Dimensions of the *Parameter*. i.e.: [T^2*L^-1]

        Raises:
            ValueError: error if the dimensions string is empty.
        """
        if not value.strip():
            raise ValueError("Dimensions cannot be empty.")
        self._dims = value
        # automatically prepare the dimensions for analysis
        self._prep_dims()

    @property
    def dim_exp(self) -> Optional[str]:
        """*dim_exp* property to get the dimensional expression of the *Parameter*. It is a string with propper parenthesis and exponents.

        Returns:
            Optional[str]: Dimensional expression of the *Parameter*. i.e.: [L^(-1)*T^(2)]
        """
        return self._dim_exp

    @dim_exp.setter
    def dim_exp(self, value: str) -> None:
        """*dim_exp* property to set the dimensional expression of the *Parameter*. It is a string with propper parenthesis and exponents.

        Args:
            value (str): Dimensional expression of the *Parameter*. i.e.: [L^(-1)*T^(2)]

        Raises:
            ValueError: error if the dimensional expression string is empty.
        """
        # TODO complete with stricter regex validation!!!
        if value is not None and not value.strip():
            raise ValueError("Dimensional expression cannot be empty.")
        self._dim_exp = value

    @property
    def sym_exp(self) -> Optional[str]:
        """*sym_exp* property to get the symbolic processed dimensional expression of the *Parameter*. It is a string suitable for Sympy processing.

        Returns:
            Optional[str]: Dimensional expression of the *Parameter* for sympy processing. i.e.: [T**2*L**(-1)]
        """
        return self._sym_exp

    @sym_exp.setter
    def sym_exp(self, value: str) -> None:
        """*sym_exp* property to set the symbolic processed dimensional expression of the *Parameter*. It is a string suitable for Sympy processing.

        Args:
            value (str): Dimensional expression of the *Parameter* for sympy processing. i.e.: [T**2*L**(-1)]

        Raises:
            ValueError: error if the dimensional expression string is empty.
        """
        # TODO complete with stricter regex validation!!!
        if value is not None and not value.strip():
            raise ValueError("Dimensional expression cannot be empty.")
        self._sym_exp = value

    @property
    def dim_col(self) -> Optional[List[int]]:
        """*dim_col* property to get the dimensional column (list) of the *Parameter*. It is a list of integers with the exponents of the dimensions in the parameter. It is the actual column of the dimensional matrix.

        Returns:
            Optional[List[int]]: List of integers with the exponents of the dimensions in the parameter. i.e.: [2, -1]
        """
        return self._dim_col

    @dim_col.setter
    def dim_col(self, value: List[int]) -> None:
        """*dim_col* property to set the dimensional column (list) of the *Parameter*. It is a list of integers with the exponents of the dimensions in the parameter. It is the actual column of the dimensional matrix.

        Args:
            value (List[int]): List of integers with the exponents of the dimensions in the parameter. i.e..: [2, -1]

        Raises:
            ValueError: error if the value is not a list of integers.
        """
        # TODO complete with regex validation, and check if the elements are numbers!!!
        if value is not None and not isinstance(value, list):
            raise ValueError("Exponents list must be a list of integers.")
        self._dim_col = value

    @property
    def units(self) -> str:
        """*units* property with the Units of Measure of the *Parameter*. It is a string with the dimensional Units of Measure.

        Returns:
            str: Units of measure of the *Parameter*. i.e.: `m/s`, `kg/m3`, etc.
        """
        return self._units

    @units.setter
    def units(self, value: str) -> None:
        """*units* property to set the Units of Measure of the *Parameter*. It is a string with the dimensional Units of Measure.

        Args:
            value (str): Units of measure of the *Parameter*. i.e `m/s`, `kg/m3`, etc.

        Raises:
            ValueError: error if the Units of Measure string is empty.
        """
        if not value.strip():
            raise ValueError("Unit of Measure cannot be empty.")
        self._units = value

    def clear(self) -> None:
        """*clear* it resets all the attributes of the *Parameter* object to their default values. It is used to clear the object for reuse.
        """
        self._idx = -1
        self._sym = ""
        self._fwk = "PHYSICAL"
        self._cat = "INPUT"
        self._dims = ""
        self._dim_exp = None
        self._sym_exp = None
        self._dim_col = None
        self._units = ""
        self.name = ""
        self.description = ""
        self.relevance = False

    def __str__(self) -> str:
        """*__str__* returns a string representation of the *Parameter* object.
        It includes the ID, symbol, framework, dimensions, category, Units of Measure, name, description, and relevance.

        Returns:
            str: String representation of the *Parameter* object.
        """
        # # get class name
        # _class_name = self.__class__.__name__
        # _str = f"{_class_name}("
        # _str += f"idx='{self._idx}', "
        # _str += f"sym='{self._sym}', "
        # _str += f"fwk='{self._fwk}', "
        # _str += f"cat='{self._cat}', "
        # _str += f"dims='{self._dims}', "
        # _str += f"dim_exp='{self._dim_exp}', "
        # _str += f"sym_exp='{self._sym_exp}', "
        # _str += f"dim_col='{self._dim_col}', "
        # _str += f"units='{self._units}', "
        # _str += f"name='{self.name}', "
        # _str += f"description='{self.description}', "
        # _str += f"relevance={self.relevance}"
        # _str += ")"
        # return _str
        # get class name
        _str = f"{self.__class__.__name__}("
        for attr, value in vars(self).items():
            # Remove leading underscore from attribute names
            _prop = attr.lstrip("_")
            if isinstance(value, str):
                _str += f"{_prop}='{value}', "
            else:
                _str += f"{_prop}={value}, "
        # removingf last ', ' from the string
        _str = _str[:-2]
        _str += ")"
        return _str


@dataclass
class Variable(Parameter):
    """**Variable** extends *Parameter* with additional attributes for min/max values, step, and standard Unit of Measure. Useful for sensitivity analysis and simulations.

    Args:
        Parameter (Generic[T]): *PyDASA* *Parameter* class for processing parameters in dimensional analysis.

    Returns:
        Variable: A *Variable* object with the following attributes:
            - `_min`: The minimum range of the *Variable*. It is a float value.
            - `_max`: The maximum range of the *Variable*. It is a float value.
            - `_std_units`: The standardized Unit of Measure of the *Variable*. It is a string with the dimensional Units of Measure. i.e.: `m/s`, `kg/m3`, etc.
            - `_std_min`: The standardized minimum range of the *Variable*, after converting Units of Measure. It is a float value.
            - `_std_max`: The standardized maximum range of the *Variable*, after converting Units of Measure. It is a float value.
            - `_std_step`: The step value of the *Variable*. It is a very small float value. It is used for sensitivity analysis and simulations.
    """

    # Private attributes with validation logic
    # :attr: _min
    _min: Optional[float] = None
    """
    Minimum range of the *Variable*. It is a float value.
    """

    # :attr: _max
    _max: Optional[float] = None
    """
    Maximum range of the *Variable*. It is a float value.
    """

    # :attr: _std_units
    _std_units: Optional[str] = ""
    """
    Standarized Unit of Measure of the *Variable*. It is a string with the dimensional Units of Measure. e.g `m/s`, `kg/m3`, etc.
    """

    # :attr: _std_min
    _std_min: Optional[float] = None
    """
    Standardized minimum range of the *Variable*, after converting Units of Measure. It is a float value.
    """

    # :attr: _std_max
    _std_max: Optional[float] = None
    """
    Standardized maximum varangelue of the *Variable*, after converting Units of Measure. It is a float value.
    """

    # :attr: _std_step
    _std_step: Optional[float] = 1 / 1000
    """
    step value of the *Variable*. It is a very small float value. It is used for sensitivity analysis and simulations.
    """

    @property
    def min(self) -> Optional[float]:
        """*min* Property to get the minimum range of the *Variable*. It is a float value.

        Returns:
            Optional[float]: minimum range of the *Variable*.
        """
        return self._min

    @min.setter
    def min(self, value: Optional[float]) -> None:
        """*min* Property to set the minimum range of the *Variable*. It is a float value.

        Args:
            value (Optional[float]): Minimum value of the pa*Variable*ameter.

        Raises:
            ValueError: error if the value is not a number.
            ValueError: error if the value is greater than the maximum range.
        """
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError("Minimum range must be a number.")
        if value > self._max:
            _msg = f"Minimum range {value} cannot be greater"
            _msg = f" than maximum range {self._max}."
            raise ValueError(_msg)
        self._min = value

    @property
    def max(self) -> Optional[float]:
        """*max* Property to get the maximum value of the *Variable*. It is a float value.

        Returns:
            Optional[float]: maximum value of the *Variable*.
        """
        return self._max

    @max.setter
    def max(self, value: Optional[float]) -> None:
        """*max* Property to set the maximum value of the *Variable*. It is a float value.

        Args:
            value (Optional[float]): maximum value of the *Variable*.

        Raises:
            ValueError: error if the value is not a number.
            ValueError: error if the value is less than the minimum range.
        """
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError("Maximum value must be a number.")
        if value < self._min:
            _msg = f"Maximum range {value} cannot be less"
            _msg = f" than minimum range {self._min}."
            raise ValueError(_msg)
        self._max = value

    @property
    def std_units(self) -> Optional[str]:
        """*std_units* Property to get the standardized Unit of Measure of the *Variable*.

        Returns:
            Optional[str]: standardized Unit of Measure of the *Variable*.
        """
        return self._std_units

    @std_units.setter
    def std_units(self, value: Optional[str]) -> None:
        """*std_units* Property to set the standardized Unit of Measure of the *Variable*.

        Args:
            value (Optional[str]): standardized Unit of Measure of the *Variable*.

        Raises:
            ValueError: error if the value is an empty string.
        """
        if value is not None and not value.strip():
            raise ValueError("Standard Unit of Measure cannot be empty.")
        self._std_units = value

    @property
    def std_min(self) -> Optional[float]:
        """*std_min* Property to get the standardized minimum value of the *Variable*.

        Returns:
            Optional[float]: standardized minimum value of the *Variable*.
        """
        return self._std_min

    @std_min.setter
    def std_min(self, value: Optional[float]) -> None:
        """*std_min* Property to set the standardized minimum range of the *Variable*.

        Args:
            value (Optional[float]): standardized minimum range of the *Variable*.

        Raises:
            ValueError: error if the value is not a number.
            ValueError: error if the value is greater than the maximum range.
        """
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError("Standard minimum value must be a number.")
        if value > self._std_max:
            _msg = f"Standard minimum value {value} cannot be greater"
            _msg = f" than standard maximum value {self._std_max}."
            raise ValueError(_msg)
        self._std_min = value

    @property
    def std_max(self) -> Optional[float]:
        """*std_max* Property to get the standardized maximum range of the *Variable*.

        Returns:
            Optional[float]: standardized maximum value of the *Variable*.
        """
        return self._std_max

    @std_max.setter
    def std_max(self, value: Optional[float]) -> None:
        """*std_max* Property to set the standardized maximum range of the *Variable*.

        Raises:
            ValueError: error if the value is not a number.
            ValueError: error if the value is less than the minimum range.
        """
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError("Standard maximum value must be a number.")
        if value < self._std_min:
            _msg = f"Standard maximum *Variable* {value} cannot be less"
            _msg = f" than standard minimum *Variable* {self._std_min}."
            raise ValueError(_msg)
        self._std_max = value

    @property
    def std_step(self) -> Optional[float]:
        """*std_step* Property to get the standarized step value of the *Variable*. It is used for sensitivity analysis and simulations.

        Returns:
            Optional[float]: standarized step value of the *Variable*.
        """
        return self._std_step

    @std_step.setter
    def std_step(self, value: Optional[float]) -> None:
        """*std_step* Property to set the standarized step value of the *Variable*. It is used for sensitivity analysis and simulations.

        Args:
            value (Optional[float]): standarized step value of the *Variable*.

        Raises:
            ValueError: error if the value is not a number.
            ValueError: error if the value is zero.
            ValueError: error if the value is greater than or equal to the standarized range between minimum and maximum values.
        """
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError("Step must be a number.")
        if value == 0:
            raise ValueError("Step cannot be zero.")
        if value >= self._std_max - self._std_min:
            _msg = f"Step {value} cannot be greater than or equal to"
            _msg = f" the range of values {self._std_max - self._std_min}."
            _msg += f"between {self._std_min} and {self._std_max}."
            raise ValueError(_msg)
        self._std_step = value

    def clear(self) -> None:
        """*clear* it resets all the attributes of the *Variable* object to their default values. It is used to clear the object for reuse.
        """
        super().clear()
        self._min = None
        self._max = None
        self._std_units = ""
        self._std_min = None
        self._std_max = None
        self._std_step = 1 / 1000

    def __str__(self) -> str:
        """*__str__* returns a string representation of the *Variable* object.
        It includes the ID, symbol, framework, dimensions, category, Units of Measure, name, description, relevance, and min/max values.

        Returns:
            str: String representation of the *Variable* object.
        """
        # # get parent class name
        # _parent_class_name = super().__class__.__name__
        # # get class name
        # _class_name = self.__class__.__name__
        # # get the class representation
        # _str = super().__str__()
        # # replace the parent class name with the class name
        # _str = _str.replace(_parent_class_name, _class_name)
        # # remove last bracket
        # _str = _str[:-1]
        # # add the class name
        # _str += f", min: {self._min}, "
        # _str += f"max: {self._max}, "
        # _str += f"std_units: {self._std_units}, "
        # _str += f"std_min: {self._std_min}, "
        # _str += f"std_max: {self._std_max}, "
        # _str += f"std_step: {self._std_step}"
        # _str += ")"
        # return _str

        # get class name
        _str = f"{self.__class__.__name__}("
        for attr, value in vars(self).items():
            # Remove leading underscore from attribute names
            _prop = attr.lstrip("_")
            if isinstance(value, str):
                _str += f"{_prop}='{value}', "
            else:
                _str += f"{_prop}={value}, "
        # removingf last ', ' from the string
        _str = _str[:-2]
        _str += ")"
        return _str
