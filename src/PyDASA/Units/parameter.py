# -*- coding: utf-8 -*-
"""
Module to represent Parameters and Variables in Dimensional Analysis for *PyDASA*.

*IMPORTANT:* This code and its specifications for Python are based on the theory and subject developed by the following authors/books:

    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""

# native python modules
# import dataclass for defining the node class
from typing import Optional
from dataclasses import dataclass
# import modules for defining the MapEntry type
from typing import Generic

# custom modules
# generic error handling and type checking
from Src.PyDASA.Utils.err import error_handler as error
from Src.PyDASA.Utils.dflt import T

# importing the FDU class for creating the FDU object
# TODO do i need this import latter?
# from Src.PyDASA.Units.fdu import FDU

# importing the FDU_FWK_TP for creating the FDU object
from Src.PyDASA.Units.fdu import FDU_FWK_DT

# checking custom modules
assert error
assert T

# Set of supported Fundamental Dimensional Units (FDU)
# :data: FDU_FWK_TP
PARAMS_FWK_DT = {
    "INPUT": "Input parameters, what I know affects the system",
    "OUTPUT": "Output parameters, Usually the result of the analysis",
    "CONTROL": "Control parameters, including constants",
}
"""
Dictionary with the supported Fundamental Dimensional Units (FDU) in *PyDASA*.
"""


@dataclass
class Parameter(Generic[T]):
    """**Parameter** class for creating a **Parameter** in *PyDASA*. Fundamental for the process of Dimensional Analysis and creating Dimensionless Coefficients.

    Args:
        Generic (T): Generic type for a Python data structure.

    Returns:
        Parameter: A Parameter object with the following attributes:
            - `_id`: The ID of the Parameter.
            - `_symbol`: The symbol of the Parameter.
            - `_framework`: The framework of the Parameter. It can be one of the following: `PHYSICAL`, `DIGITAL`, or `CUSTOM`.
            - `_dimensions`: The dimensions of the Parameter.
            - `_units_of_measure`: The units of measure of the Parameter.
            - `_category`: The category of the Parameter. It can be one of the following: `INPUT`, `OUTPUT`, or `CONTROL`.
            - `name`: The name of the Parameter.
            - `description`: The description of the Parameter.
            - `relevance`: The relevance of the Parameter. It can be `True` or `False`.
    """

    # Private attributes with validation logic
    # :attr: _id
    _id: str = ""
    """
    ID of the FDU. It must be alphanumeric. Useful for identifying the FDU in the system and dimensional matrix construction.
    """

    # :attr: _symbol
    _symbol: str = ""
    """
    Symbol of the FDU. It must be alphanumeric (preferably a single character + Latin or Greek letter). Useful for user-friendly representation of the FDU.
    """

    # :attr: _framework
    _framework: str = ""
    """
    Framework of the FDU. It can be one of the following: `PHYSICAL`, `DIGITAL`, or `CUSTOM`. Useful for identifying the framework of the FDU.
    """

    # :attr: _category`
    _category: str = ""
    """
    The parameter category. It can be one of the following: `INPUT`, `OUTPUT`, or `CONTROL`. Useful for identifying the order of its column in the dimensional matrix.
    """

    # :attr: _dimensions
    _dimensions: str = ""
    """
    Dimensions of the parameter. It is a string with the FDU (Fundamental Dimensional Unit) of the parameter. It is used to calculate the dimensional matrix rows.
    """

    # :attr: _units_of_measure
    _units_of_measure: str = ""
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

    @property
    def id(self) -> str:
        """*id* property to get the ID of the Parameter.

        Returns:
            str: ID of the Parameter.
        """
        return self._id

    @id.setter
    def id(self, value: str) -> None:
        """*id* property to set the ID of the Parameter. It must be alphanumeric.

        Args:
            value (str): ID of the Parameter.

        Raises:
            ValueError: error if the ID is not alphanumeric.
        """
        if not value.isalnum():
            raise ValueError("ID must be alphanumeric.")
        self._id = value

    @property
    def symbol(self) -> str:
        """symbol property to get the symbol of the FDU.

        Returns:
            str: Symbol of the FDU.
        """
        return self._symbol

    @symbol.setter
    def symbol(self, value: str) -> None:

        if not value.isalnum():
            _msg = "Symbol must be alphanumeric. "
            _msg += f"Provided: {value}"
            _msg += "Preferably a Latin or Greek letter."
            raise ValueError(_msg)
        self._symbol = value

    @property
    def framework(self) -> str:
        """*framework* property to get the framework of the FDU.

        Returns:
            str: Framework of the FDU. It can be one of the following: `PHYSICAL`, `DIGITAL`, or `CUSTOM`.
        """
        return self._framework

    @framework.setter
    def framework(self, value: str) -> None:
        """*framework* property of the allowed framework of the FDU.

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
        self._framework = value

    @property
    def category(self) -> str:
        """*category* property to get the category of the parameter.

        Returns:
            str: Category of the parameter. It can be one of the following: `INPUT`, `OUTPUT`, or `CONTROL`.
        """
        return self._category

    @category.setter
    def category(self, value: str) -> None:
        """*category* property to set the category of the parameter. It must be one of the following: `INPUT`, `OUTPUT`, or `CONTROL`.

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
        self._category = value.upper()

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
    def units_of_measure(self) -> str:
        """*units_of_measure* property to get the units of measure of the parameter.

        Returns:
            str: Units of measure of the parameter. It is a string with the dimensional units of measure.
        """
        return self._units_of_measure

    @units_of_measure.setter
    def units_of_measure(self, value: str) -> None:
        """*units_of_measure* property to set the units of measure of the parameter. It is a string with the dimensional units of measure.

        Args:
            value (str): Units of measure of the parameter. It is a string with the dimensional units of measure i.e `m/s`, `kg/m3`, etc.

        Raises:
            ValueError: error if the units of measure string is empty.
        """
        if not value.strip():
            raise ValueError("Unit of measure cannot be empty.")
        self._units_of_measure = value

    def __str__(self) -> str:
        """*__str__* returns a string representation of the Parameter object.
        It includes the ID, symbol, framework, dimensions, category, units of measure, name, description, and relevance.

        Returns:
            str: String representation of the Parameter object.
        """
        _str = f"Parameter ID: {self._id}, "
        _str += f"Symbol: {self._symbol}, "
        _str += f"Framework: {self._framework}, "
        _str += f"Dimensions: {self._dimensions}, "
        _str += f"Category: {self._category}, "
        _str += f"Units of Measure: {self._units_of_measure}, "
        _str += f"Name: {self.name}, "
        _str += f"Description: {self.description}, "
        _str += f"Relevance: {self.relevance}"
        return _str


@dataclass
class Variable(Parameter):
    """Extends Parameter with additional attributes for min/max values, step, and standard unit of measure. Useful for sensitivity analysis and simulations."""

    # Private attributes with validation logic
    # :attr: _max_value
    _min_value: Optional[float] = 0.0
    """
    Minimum value of the parameter. It is a float value.
    """

    # :attr: _max_value
    _max_value: Optional[float] = 0.0
    """
    Maximum value of the parameter. It is a float value.
    """

    # :attr: _std_unit_of_meassure
    _std_unit_of_meassure: Optional[str] = ""
    """
    Standarized unit of measure of the parameter. It is a string with the dimensional units of measure. e.g `m/s`, `kg/m3`, etc.
    """

    # :attr: _std_min_value
    _std_min_value: Optional[float] = 0.0
    """
    Standardized minimum value of the parameter, after converting units of measure. It is a float value.
    """

    # :attr: _std_max_value
    _std_max_value: Optional[float] = 0.0
    """
    Standardized maximum value of the parameter, after converting units of measure. It is a float value.
    """

    # :attr: _step
    _step: Optional[float] = 1 / 10000
    """
    step value of the parameter. It is a very small float value. It is used for sensitivity analysis and simulations.
    """

    @property
    def min_value(self) -> Optional[float]:
        """*min_value* Property to get the minimum value of the parameter. It is a float value.

        Returns:
            Optional[float]: minimum value of the parameter.
        """
        return self._min_value

    @min_value.setter
    def min_value(self, value: Optional[float]) -> None:
        """*min_value* Property to set the minimum value of the parameter. It is a float value.

        Args:
            value (Optional[float]): Minimum value of the parameter.

        Raises:
            ValueError: error if the value is not a number.
            ValueError: error if the value is greater than the maximum value.
        """
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError("Minimum value must be a number.")
        if value > self._max_value:
            _msg = f"Minimum value {value} cannot be greater"
            _msg = f" than maximum value {self._max_value}."
            raise ValueError(_msg)
        self._min_value = value

    @property
    def max_value(self) -> Optional[float]:
        """*max_value* Property to get the maximum value of the parameter. It is a float value.

        Returns:
            Optional[float]: maximum value of the parameter.
        """
        return self._max_value

    @max_value.setter
    def max_value(self, value: Optional[float]) -> None:
        """*max_value* Property to set the maximum value of the parameter. It is a float value.

        Args:
            value (Optional[float]): maximum value of the parameter.

        Raises:
            ValueError: error if the value is not a number.
            ValueError: error if the value is less than the minimum value.
        """
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError("Maximum value must be a number.")
        if value < self._min_value:
            _msg = f"Maximum value {value} cannot be less"
            _msg = f" than minimum value {self._min_value}."
            raise ValueError(_msg)
        self._max_value = value

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
        if value >= self._max_value - self._min_value:
            _msg = f"Step {value} cannot be greater than or equal to"
            _msg = f" the range of values {self._max_value - self._min_value}."
            _msg += f"between {self._min_value} and {self._max_value}."
            raise ValueError(_msg)
        self._step = value

    @property
    def std_unit_of_meassure(self) -> Optional[str]:
        """*std_unit_of_meassure* Property to get the standardized unit of measure of the parameter.

        Returns:
            Optional[str]: standardized unit of measure of the parameter.
        """
        return self._std_unit_of_meassure

    @std_unit_of_meassure.setter
    def std_unit_of_meassure(self, value: Optional[str]) -> None:
        """*std_unit_of_meassure* Property to set the standardized unit of measure of the parameter.

        Args:
            value (Optional[str]): standardized unit of measure of the parameter.

        Raises:
            ValueError: error if the value is an empty string.
        """
        if value is not None and not value.strip():
            raise ValueError("Standard unit of measure cannot be empty.")
        self._std_unit_of_meassure = value

    @property
    def std_min_value(self) -> Optional[float]:
        """*std_min_value* Property to get the standardized minimum value of the parameter.

        Returns:
            Optional[float]: standardized minimum value of the parameter.
        """
        return self._std_min_value

    @std_min_value.setter
    def std_min_value(self, value: Optional[float]) -> None:
        """*std_min_value* Property to set the standardized minimum value of the parameter.

        Args:
            value (Optional[float]): standardized minimum value of the parameter.

        Raises:
            ValueError: error if the value is not a number.
            ValueError: error if the value is greater than the maximum value.
        """
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError("Standard minimum value must be a number.")
        if value > self._std_max_value:
            _msg = f"Standard minimum value {value} cannot be greater"
            _msg = f" than standard maximum value {self._std_max_value}."
            raise ValueError(_msg)
        self._std_min_value = value

    @property
    def std_max_value(self) -> Optional[float]:
        """*std_max_value* Property to get the standardized maximum value of the parameter.

        Returns:
            Optional[float]: standardized maximum value of the parameter.
        """
        return self._std_max_value

    @std_max_value.setter
    def std_max_value(self, value: Optional[float]) -> None:
        """*std_max_value* Property to set the standardized maximum value of the parameter.

        Raises:
            ValueError: error if the value is not a number.
            ValueError: error if the value is less than the minimum value.
        """
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError("Standard maximum value must be a number.")
        if value < self._std_min_value:
            _msg = f"Standard maximum value {value} cannot be less"
            _msg = f" than standard minimum value {self._std_min_value}."
            raise ValueError(_msg)
        self._std_max_value = value

    def __str__(self) -> str:
        """*__str__* returns a string representation of the Variable object.
        It includes the ID, symbol, framework, dimensions, category, units of measure, name, description, relevance, and min/max values.

        Returns:
            str: String representation of the Variable object.
        """
        _str = super().__str__()
        _str += f", Min Value: {self._min_value}, "
        _str += f"Max Value: {self._max_value}, "
        _str += f"Step: {self._step}, "
        _str += f"Standard Unit of Measure: {self._std_unit_of_meassure}, "
        _str += f"Standard Min Value: {self._std_min_value}, "
        _str += f"Standard Max Value: {self._std_max_value}"
        return _str
