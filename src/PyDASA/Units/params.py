# -*- coding: utf-8 -*-
"""
Module to represent Parameters and Variables in Dimensional Analysis for *PyDASA*.

*IMPORTANT:* This code and its specifications for Python are based on the theory and subject developed by the following authors/books:

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

# TODO do i need this import???
# importing the FDU_FWK_TP for creating the FDU object
from Src.PyDASA.Utils.cfg import FDU_FWK_DT
from Src.PyDASA.Utils.cfg import PARAMS_FWK_DT

# checking custom modules
assert error
assert T


@dataclass
class Parameter(Generic[T]):
    """**Parameter** class for creating a **Parameter** in *PyDASA*. Fundamental for the process of Dimensional Analysis and creating Dimensionless Coefficients.

    Args:
        Generic (T): Generic type for a Python data structure.

    Returns:
        Parameter: A Parameter object with the following attributes:
            - `_idx`: The ID of the Parameter.
            - `_sym`: The symbol of the Parameter.
            - `_fwk`: The framework of the Parameter. It can be one of the following: `PHYSICAL`, `DIGITAL`, or `CUSTOM`.
            - `_dimensions`: The dimensions of the Parameter.
            - `_units`: The units of measure of the Parameter.
            - `_cat`: The category of the Parameter. It can be one of the following: `INPUT`, `OUTPUT`, or `CONTROL`.
            - `name`: The name of the Parameter.
            - `description`: The description of the Parameter.
            - `relevance`: The relevance of the Parameter. It can be `True` or `False`.
    """

    # Private attributes with validation logic
    # :attr: _idx
    _idx: int = -1
    """
    Index of the parameter. It is the unique integer for the order of the column of the parameter in the dimensional matrix.
    """

    # Symbol of the FDU
    # :attr: _sym
    _sym: str = ""
    """
    Symbol of the FDU. It must be alphanumeric (preferably a single character + Latin or Greek letter). Useful for user-friendly representation of the FDU.
    """

    # Working framework of the FDU
    # :attr: _fwk
    _fwk: str = "PHYSICAL"
    """
    Framework of the FDU. It can be one of the following: `PHYSICAL`, `DIGITAL`, or `CUSTOM`. Useful for identifying the framework of the FDU.
    """

    # Category of the parameter, can be: `INPUT`, `OUTPUT`, or `CONTROL`
    # :attr: _cat`
    _cat: str = "INPUT"
    """
    The parameter category. It can be one of the following: `INPUT`, `OUTPUT`, or `CONTROL`. Useful for identifying the order of its column in the dimensional matrix.
    """

    # user input dimensional expression for the parameter
    # :attr: _dimensions
    _dimensions: str = ""
    """
    User-defined dimensional expression for the Parameter. It is a string with the FDU of the parameter. i.e.: [T^2*L^-1] or [T^2*L].
    """

    # user dimensional expression
    # :attr: _dim_exp
    _dim_exp: Optional[str] = None
    """
    Dimensional expression for analysis. It is a string with propper parenthesis and exponents. It is used to calculate the dimensional matrix columns. i.e.: from [T^2*L^-1] to [T^(2)*L^(-1)].
    """

    # dimensional expression for the sympy regex procesor
    # :attr: _sym_exp
    _sym_exp: Optional[str] = None
    """
    Symbolic processed dimensional expression for analysis. It is a string suitable for Sympy processing. It is used to calculate the dimensional matrix columns. i.e.: from [T^(2)*L^(-1)] to [T**2*L**(-1)].
    """

    # list with the dimensions exponent coefficients as integers
    # :attr: _dim_col
    _dim_col: Optional[List[int]] = field(default_factory=list)
    """
    Dimensioan list of the parameter. It is a list of numbers (integers or floats) with the exponents of the dimensions in the parameter. It is used as the columns of the dimensional matrix. i.e.: from [T^2*L^-1] to [2, -1].
    """

    # public attributes
    # :attr: _units
    _units: str = ""
    """
    Original unit of meassure the phenomena parameter is measured in. It is a string with with dimensional units of measure. It is used to convert the parameter to the standard unit of measure.
    """

    # Public attributes
    # :attr: name
    name: str = ""
    """
    Name of the parameter. User-friendly name of the parameter.
    """

    # :attr: description
    description: str = ""
    """
    Description of the parameter. It is a string with a small summary of the parameter.
    """

    # :attr: _relevance
    relevance: bool = False
    """
    Boolean value indicating if the parameter is relevant or not. It is used to identify what parameter goes into creating the dimensional matrix
    """

    def __post_init__(self) -> None:
        # # check for valid dimensions
        if self._dimensions != "":
            if not self._validate_regex(self.dimensions, config.WKNG_FDU_REGEX):
                _msg = f"Invalid dimensions in Parameter '{self.name}' "
                _msg += f"Check FDUs in: {self._dimensions}."
                _msg += " in acordance with the FDU precedence list:, "
                _msg += f"'{config.WKNG_FDU_PREC_LT}'/"
                raise ValueError(_msg)

            # prepare the parameter for dimensional analysis
            # set up dimensions in uppercase
            self._dim_exp = self._standarize_dimensions(self._dimensions)
            print(self._dim_exp)

            # sort dimensions in the dimensional precedence order
            self._dim_exp = self._sort_dimensions(self._dim_exp)
            print(self._dim_exp)

            # set up expression for sympy
            self._sym_exp = self._set_sym_dimension(self._dim_exp)
            print(self._sym_exp)

            # setup dimension pow list for dimensional analysis
            self._dim_col = self._set_dimensional_col(self._sym_exp)
            print(self._dim_col)

        # if description is not empty, capitalize it
        if self.description != "":
            self.description = self.description.capitalize()

    def _validate_regex(self, dims: str, regex: str) -> bool:
        """*_validate_regex* validates the dimensions of the parameter.

        Args:
            dims (str): Dimensions of the parameter. It is a string with the FDU formula of the parameter. i.e.: [T^2*L^-1]

        Returns:
            bool: True if the dimensions are valid, False otherwise.
        """
        # check if the dimensions are valid
        _valid = bool(re.match(regex, dims))
        return _valid

    def _find_pattern(self, sym: str, pattern: str) -> str:
        """*_find_pattern* finds the pattern in the standarize symbolic string.

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

    def _standarize_dimensions(self, dims: str) -> str:
        """*_standarize_dimensions* standarizes the dimensions of the parameter. It adds parentheses to the powers, and adds ^1 to the * operations.

        Args:
            dims (str): Dimensions of the parameter. It is a string with the FDU formula of the parameter. i.e.: [T^2*L^-1]

        Returns:
            str: Standarized dimensions of the parameter. It is a string with the FDU formula of the parameter. i.e.: [T^(2)*L^(-1)]
        """
        # add parentheses to powers in dimensions
        _regex_pat = re.compile(config.WKNG_POW_REGEX)
        _dims = _regex_pat.sub(lambda m: f"({m.group(0)})", dims)
        # add ^1 to * and / operations in dimensions
        _regex_pat = re.compile(config.WKNG_NO_POW_REGEX)
        _dims = _regex_pat.sub(lambda m: f"{m.group(0)}^(1)", _dims)
        # return standarized dimensions
        return _dims

    def _sort_dimensions(self, dims: str) -> str:
        """*_sort_dimensions* sorts the dimensions of the parameter in the dimensional precedence order. Crucrial to create a consistent dimensional matrix.

        Args:
            dims (str): Dimensions of the parameter. It is a string with the FDU formula of the parameter. i.e.: [T^2*L^-1]

        Returns:
            str: Sorted dimensions of the parameter. It is a string with the FDU formula of the parameter. i.e.: [L^(-1)*T^(2)]
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

    def _set_sym_dimension(self, dims: str) -> str:
        """*_set_sym_dimension* sets the dimensional expression special characters such as '^' and '*' to '**' and '* ' respectively. for sympy processing.

        Args:
            dims (str): Dimensions of the parameter. It is a string with the FDU formula of the parameter. i.e.: [T^2*L^-1]

        Returns:
            str: Symbolic dimensions of the parameter. It is a string with the FDU formula of the parameter. i.e.: [T**2*L**(-1)]
        """
        # replace '*' with '* ' for sympy processing
        # TODO move '*' and '* ' as global operator to cfg module?
        _dims = dims.replace("*", "* ")
        # replace '^' with '**' for sympy processing
        # TODO move '^' and '**' as global operator to cfg module?
        _dims = _dims.replace("^", "**")
        return _dims

    def _set_dimensional_col(self, dims: str) -> List[int]:
        """*_set_dimensional_col* sets the dimensional column of the parameter. It is a list of integers with the exponents of the dimensions of the Parameter. It is used to create the dimensional matrix.

        Args:
            dims (str): Standarized dimensions of the parameter. It is a string with the FDU formula of the parameter. i.e.: [T^(2)*L^(-1)]

        Returns:
            List[int]: list of integers with the exponents of the dimensions of the parameter. It is used to create the dimensional matrix. i.e.: [2, -1]
        """
        # TODO check this algorithm for improvement
        # split the sympy expression into a list of dimensions
        _dims_lt = dims.split("* ")
        # set the default list of zeros with the FDU length
        _dimensional_col = len(config.WKNG_FDU_PREC_LT) * [0]
        print(_dimensional_col)
        # working vars
        i = 0
        print(config.WKNG_POW_REGEX)
        _regex_pat = re.compile(r"\-?\d+")
        while i < len(_dims_lt):
            # get the dimension
            _t_sym = _dims_lt[i]
            print(_t_sym)
            # match the exponent of the dimension
            _t_pow = _regex_pat.findall(_t_sym)
            print(_t_pow)
            # find the fdu and its index
            _t_dim = self._find_pattern(_t_sym, config.WKNG_FDU_SYM_REGEX)
            print(_t_dim)
            _t_idx = config.WKNG_FDU_PREC_LT.index(_t_dim[0])
            # update the dimension column with the exponent of the dimension
            print(_t_idx, _t_pow[0])
            _dimensional_col[_t_idx] = int(_t_pow[0])
            # increment the index
            i += 1
        # return the list of powers
        return _dimensional_col

    @property
    def idx(self) -> str:
        """*idx* property to get Parameter's index in the dimensional matrix.

        Returns:
            str: ID of the Parameter.
        """
        return self._idx

    @idx.setter
    def idx(self, value: str) -> None:
        """*idx* property to set the Parameter's index in the dimensional matrix. It must be alphanumeric.

        Args:
            value (str): ID of the Parameter.

        Raises:
            ValueError: error if the ID is not alphanumeric.
        """
        if not value.isalnum():
            raise ValueError("ID must be alphanumeric.")
        self._idx = value

    @property
    def sym(self) -> str:
        """*sym* property to get the symbol of the FDU.

        Returns:
            str: Symbol of the FDU.
        """
        return self._sym

    @sym.setter
    def sym(self, value: str) -> None:

        if not value.isalnum():
            _msg = "Symbol must be alphanumeric. "
            _msg += f"Provided: {value}"
            _msg += "Preferably a Latin or Greek letter."
            raise ValueError(_msg)
        self._sym = value

    @property
    def fwk(self) -> str:
        """*fwk* property to get the framework of the FDU.

        Returns:
            str: Framework of the FDU. It can be one of the following: `PHYSICAL`, `DIGITAL`, or `CUSTOM`.
        """
        return self._fwk

    @fwk.setter
    def fwk(self, value: str) -> None:
        """*fwk* property of the allowed framework of the FDU.

        Args:
            value (str): Framework of the parameter (related to the FDU). It can be one of the following: `PHYSICAL`, `DIGITAL`, or `CUSTOM`.

        Raises:
            ValueError: If the framework is not one of the allowed values.
        """
        if value not in FDU_FWK_DT.keys():
            _msg = f"Invalid framework: {value}. "
            _msg += "Framework must be one of the following: "
            _msg += f"{', '.join(FDU_FWK_DT.keys())}."
            raise ValueError(_msg)
        self._fwk = value

    @property
    def cat(self) -> str:
        """*cat* property to get the category of the parameter.

        Returns:
            str: Category of the parameter. It can be one of the following: `INPUT`, `OUTPUT`, or `CONTROL`.
        """
        return self._cat

    @cat.setter
    def cat(self, value: str) -> None:
        """*cat* property to set the category of the parameter. It must be one of the following: `INPUT`, `OUTPUT`, or `CONTROL`.

        Args:
            value (str): Category of the parameter. It can be one of the following: `INPUT`, `OUTPUT`, or `CONTROL`.

        Raises:
            ValueError: error if the category is not one of the allowed values.
        """
        if value.upper() not in PARAMS_FWK_DT.keys():
            _msg = f"Invalid category: {value}. "
            _msg += "Category must be one of the following: "
            _msg += f"{', '.join(PARAMS_FWK_DT.keys())}."
            raise ValueError(_msg)
        self._cat = value.upper()

    @property
    def dimensions(self) -> str:
        """*dimensions* property to get the dimensions of the parameter.

        Returns:
            str: Dimensions of the parameter. It is a string with the FDU (Fundamental Dimensional Unit) of the parameter.
        """
        return self._dimensions

    @dimensions.setter
    def dimensions(self, value: str) -> None:
        """*dimensions* property to set the dimensions of the parameter. It is a string with the FDU (Fundamental Dimensional Unit) of the parameter.

        Args:
            value (str): Dimensions of the parameter. It is a string with the FDU (Fundamental Dimensional Unit) of the parameter.

        Raises:
            ValueError: error if the dimensions string is empty.
        """
        if not value.strip():
            raise ValueError("Dimensions cannot be empty.")
        self._dimensions = value

    @property
    def dim_exp(self) -> Optional[str]:
        """*dim_exp* property to get the dimensional expression of the parameter.

        Returns:
            Optional[str]: Dimensional expression of the parameter. It is a string with propper parenthesis and exponents.
        """
        return self._dim_exp

    @dim_exp.setter
    def dim_exp(self, value: str) -> None:
        """*dim_exp* _summary_

        Args:
            value (str): _description_

        Raises:
            ValueError: _description_
        """
        # TODO complete with regex validation!!!
        if value is not None and not value.strip():
            raise ValueError("Dimensional expression cannot be empty.")
        self._dim_exp = value

    @property
    def sym_exp(self) -> Optional[str]:
        """*sym_exp* property to get the symbolic processed dimensional expression of the parameter.

        Returns:
            Optional[str]: Dimensional expression of the parameter. It is a string suitable for Sympy processing.
        """
        return self._sym_exp

    @sym_exp.setter
    def sym_exp(self, value: str) -> None:
        """*sym_exp* _summary_

        Args:
            value (str): _description_

        Raises:
            ValueError: _description_
        """
        # TODO complete with regex validation!!!
        if value is not None and not value.strip():
            raise ValueError("Dimensional expression cannot be empty.")
        self._sym_exp = value

    @property
    def dim_col(self) -> Optional[List[int]]:
        """*dim_col* property to get the list of dimensions exponent coefficients as integers.

        Returns:
            Optional[List[int]]: List of integers with the exponents of the dimensions in the parameter.
        """
        return self._dim_col

    @dim_col.setter
    def dim_col(self, value: List[int]) -> None:
        """*dim_col* _summary_

        Args:
            value (List[int]): _description_

        Raises:
            ValueError: _description_
        """
        # TODO complete with regex validation, and check if the elements are numbers!!!
        if value is not None and not isinstance(value, list):
            raise ValueError("Exponents list must be a list of integers.")
        self._dim_col = value

    @property
    def units(self) -> str:
        """*units* property to get the units of measure of the parameter.

        Returns:
            str: Units of measure of the parameter. It is a string with the dimensional units of measure.
        """
        return self._units

    @units.setter
    def units(self, value: str) -> None:
        """*units* property to set the units of measure of the parameter. It is a string with the dimensional units of measure.

        Args:
            value (str): Units of measure of the parameter. It is a string with the dimensional units of measure i.e `m/s`, `kg/m3`, etc.

        Raises:
            ValueError: error if the units of measure string is empty.
        """
        if not value.strip():
            raise ValueError("Unit of measure cannot be empty.")
        self._units = value

    def __str__(self) -> str:
        """*__str__* returns a string representation of the Parameter object.
        It includes the ID, symbol, framework, dimensions, category, units of measure, name, description, and relevance.

        Returns:
            str: String representation of the Parameter object.
        """
        # get class name
        _class_name = self.__class__.__name__
        _str = f"{_class_name}("
        _str += f"idx='{self._idx}', "
        _str += f"sym='{self._sym}', "
        _str += f"fwk='{self._fwk}', "
        _str += f"cat='{self._cat}', "
        _str += f"dimensions='{self._dimensions}', "
        _str += f"dim_exp='{self._dim_exp}', "
        _str += f"sym_exp='{self._sym_exp}', "
        _str += f"dim_col='{self._dim_col}', "
        _str += f"unit_of_meassure='{self._units}', "
        _str += f"name='{self.name}', "
        _str += f"description='{self.description}', "
        _str += f"relevance={self.relevance}"
        _str += ")"
        return _str


@dataclass
class Variable(Parameter):
    """Extends Parameter with additional attributes for min/max values, step, and standard unit of measure. Useful for sensitivity analysis and simulations."""

    # Private attributes with validation logic
    # :attr: _min
    _min: Optional[float] = 0.0
    """
    Minimum value of the parameter. It is a float value.
    """

    # :attr: _max
    _max: Optional[float] = 0.0
    """
    Maximum value of the parameter. It is a float value.
    """

    # :attr: _std_units
    _std_units: Optional[str] = ""
    """
    Standarized unit of measure of the parameter. It is a string with the dimensional units of measure. e.g `m/s`, `kg/m3`, etc.
    """

    # :attr: _std_min
    _std_min: Optional[float] = 0.0
    """
    Standardized minimum value of the parameter, after converting units of measure. It is a float value.
    """

    # :attr: _std_max
    _std_max: Optional[float] = 0.0
    """
    Standardized maximum value of the parameter, after converting units of measure. It is a float value.
    """

    # :attr: _step
    _step: Optional[float] = 1 / 1000
    """
    step value of the parameter. It is a very small float value. It is used for sensitivity analysis and simulations.
    """

    @property
    def min(self) -> Optional[float]:
        """*min* Property to get the minimum value of the parameter. It is a float value.

        Returns:
            Optional[float]: minimum value of the parameter.
        """
        return self._min

    @min.setter
    def min(self, value: Optional[float]) -> None:
        """*min* Property to set the minimum value of the parameter. It is a float value.

        Args:
            value (Optional[float]): Minimum value of the parameter.

        Raises:
            ValueError: error if the value is not a number.
            ValueError: error if the value is greater than the maximum value.
        """
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError("Minimum value must be a number.")
        if value > self._max:
            _msg = f"Minimum value {value} cannot be greater"
            _msg = f" than maximum value {self._max}."
            raise ValueError(_msg)
        self._min = value

    @property
    def max(self) -> Optional[float]:
        """*max* Property to get the maximum value of the parameter. It is a float value.

        Returns:
            Optional[float]: maximum value of the parameter.
        """
        return self._max

    @max.setter
    def max(self, value: Optional[float]) -> None:
        """*max* Property to set the maximum value of the parameter. It is a float value.

        Args:
            value (Optional[float]): maximum value of the parameter.

        Raises:
            ValueError: error if the value is not a number.
            ValueError: error if the value is less than the minimum value.
        """
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError("Maximum value must be a number.")
        if value < self._min:
            _msg = f"Maximum value {value} cannot be less"
            _msg = f" than minimum value {self._min}."
            raise ValueError(_msg)
        self._max = value

    @property
    def step(self) -> Optional[float]:
        """*step* Property to get the step value of the parameter. It is used for sensitivity analysis and simulations.

        Returns:
            Optional[float]: step value of the parameter.
        """
        return self._step

    @step.setter
    def step(self, value: Optional[float]) -> None:
        """*step* Property to set the step value of the parameter. It is used for sensitivity analysis and simulations.

        Args:
            value (Optional[float]): step value of the parameter.

        Raises:
            ValueError: error if the value is not a number.
            ValueError: error if the value is zero.
            ValueError: error if the value is greater than or equal to the range of values.
        """
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError("Step must be a number.")
        if value == 0:
            raise ValueError("Step cannot be zero.")
        if value >= self._max - self._min:
            _msg = f"Step {value} cannot be greater than or equal to"
            _msg = f" the range of values {self._max - self._min}."
            _msg += f"between {self._min} and {self._max}."
            raise ValueError(_msg)
        self._step = value

    @property
    def std_units(self) -> Optional[str]:
        """*std_units* Property to get the standardized unit of measure of the parameter.

        Returns:
            Optional[str]: standardized unit of measure of the parameter.
        """
        return self._std_units

    @std_units.setter
    def std_units(self, value: Optional[str]) -> None:
        """*std_units* Property to set the standardized unit of measure of the parameter.

        Args:
            value (Optional[str]): standardized unit of measure of the parameter.

        Raises:
            ValueError: error if the value is an empty string.
        """
        if value is not None and not value.strip():
            raise ValueError("Standard unit of measure cannot be empty.")
        self._std_units = value

    @property
    def std_min(self) -> Optional[float]:
        """*std_min* Property to get the standardized minimum value of the parameter.

        Returns:
            Optional[float]: standardized minimum value of the parameter.
        """
        return self._std_min

    @std_min.setter
    def std_min(self, value: Optional[float]) -> None:
        """*std_min* Property to set the standardized minimum value of the parameter.

        Args:
            value (Optional[float]): standardized minimum value of the parameter.

        Raises:
            ValueError: error if the value is not a number.
            ValueError: error if the value is greater than the maximum value.
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
        """*std_max* Property to get the standardized maximum value of the parameter.

        Returns:
            Optional[float]: standardized maximum value of the parameter.
        """
        return self._std_max

    @std_max.setter
    def std_max(self, value: Optional[float]) -> None:
        """*std_max* Property to set the standardized maximum value of the parameter.

        Raises:
            ValueError: error if the value is not a number.
            ValueError: error if the value is less than the minimum value.
        """
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError("Standard maximum value must be a number.")
        if value < self._std_min:
            _msg = f"Standard maximum value {value} cannot be less"
            _msg = f" than standard minimum value {self._std_min}."
            raise ValueError(_msg)
        self._std_max = value

    def __str__(self) -> str:
        """*__str__* returns a string representation of the Variable object.
        It includes the ID, symbol, framework, dimensions, category, units of measure, name, description, relevance, and min/max values.

        Returns:
            str: String representation of the Variable object.
        """
        # get parent class name
        _parent_class_name = super().__class__.__name__
        # get class name
        _class_name = self.__class__.__name__
        # get the class representation
        _str = super().__str__()
        # replace the parent class name with the class name
        _str = _str.replace(_parent_class_name, _class_name)
        # remove last bracket
        _str = _str[:-1]
        # add the class name
        _str += f", min: {self._min}, "
        _str += f"max: {self._max}, "
        _str += f"step: {self._step}, "
        _str += f"std_units: {self._std_units}, "
        _str += f"std_min: {self._std_min}, "
        _str += f"std_max: {self._std_max}"
        _str += ")"
        return _str
