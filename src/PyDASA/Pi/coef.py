# -*- coding: utf-8 -*-
"""
Module for representing Dimensionless Coefficients (DN) in *PyDASA*. Defines the *PiCoefficient* class, which models DN in dimensional analysis.

*IMPORTANT:* Based on the theory from:

    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""

# native python modules
# import dataclass for class attributes and validations
from typing import Optional, List, Generic
from dataclasses import dataclass, field

# custom modules
from Src.PyDASA.Utils.dflt import T
# generic error handling and type checking
from Src.PyDASA.Utils.err import error_handler as error

# importing PyDASA's regex for managing FDUs
# using the 'as' allows shared variable edition
from Src.PyDASA.Utils import cfg as config

# checking custom modules
assert error
assert config
assert T


@dataclass
class PiCoefficient(Generic[T]):
    """*PiCoefficient* class for creating a Dimensionless Coefficient (DN) in *PyDASA*. It is used to represent the coefficients in the dimensional model.

    Args:
        Generic (T): Generic type for a Python data structure.

    Returns:
        PiCoefficient: A *PiCoefficient* object for Dimensional Coefficients (DCs) with the following attributes:
            - `_idx`: The ID of the *PiCoefficient*.
            - `_sym`: The symbol of the *PiCoefficient*.
            - `_fwk`: The framework of the *PiCoefficient*. It can be one of the following: `PHYSICAL`, `COMPUTATION`, `DIGITAL` or `CUSTOM`.
            - `_cat`: The category of the *PiCoefficient*. It can be one of the following: `COMPUTED` or `DERIVED`.
            - `_param_lt`: List of parameter symbols used in the *PiCoefficient*.
            - `_dim_col`: Dimensional column of the *PiCoefficient*.
            - `_pivot_lt`: List of pivot indices of the dimensional matrix.
            - `name`: The name of the *PiCoefficient*.
            - `description`: The description of the *PiCoefficient*.
            - `relevance`: Boolean value indicating if the *PiCoefficient* is relevant or not.
    """
    # Private attributes with validation logic
    # :attr: _idx
    _idx: int = -1
    """
    Index of the *PiCoefficient*. It is the unique integer if the Dimensionless Coefficient (DN). It is used to identify the precedence of the DN in the dimensional model.
    """

    # Symbol of the FDU
    # :attr: _sym
    _sym: str = "\\Pi_{}"
    """
    Symbol of the *PiCoefficient*. It must be alphanumeric (by default is the Greek letter *Pi*). It is used to identify the DN in the dimensional model. The Latex default symbol is `\\Pi_{}` and the Index is use to complete the symbol. i.e.: `\\Pi_{1}`.
    """

    # Working framework of the FDU
    # :attr: _fwk
    _fwk: str = "PHYSICAL"
    """
    Framework the *PiCoefficient* follows in accordance with the FDU. It can be one of the following: `PHYSICAL`, `COMPUTATION`, `DIGITAL` or `CUSTOM`. Coefficients, Parameters, and FDUs must be in the same framework.
    """

    # Category of the parameter, can be: `INPUT`, `OUTPUT`, or `CONTROL`
    # :attr: _cat`
    _cat: str = "COMPUTED"
    """
    Category of the *PiCoefficient*. It is a string specifing the coefficient's origin. It can be one of the following: `COMPUTED` or `DERIVED`. for example, `COMPUTED` are coefficients that are computed from the Buckingham Pi theorem and the dimensional matrix. `DERIVED` are coefficients that are derived from other computed coefficients.
    """

    # TODO do I use Data Structures for this?
    _param_lt: List[str] = field(default_factory=list)
    """
    List of parameter symbols used in the *PiCoefficient*. It is a list of `str` objects. It is used to identify the parameters used to calculate the coefficient.
    """

    # list with the coefficient's parameters dimensional exponent as integers
    # :attr: _dim_col
    _dim_col: List[int] = field(default_factory=list)
    """
    Dimensional Column (list) of the *PiCoefficient*. A list of integers representing the exponents of the *Parameters* used to calculate the Dimensionless Coefficient. i.e.: `[1, 2, -1]`.
    """

    # list with the pivot indices of the dimensional matrix
    # :attr: _pivot_lt
    # TODO check if I really need this!!!
    _pivot_lt: Optional[List[int]] = None
    """
    List of pivot indices of the dimensional matrix. It is a list of integers representing the indices of the pivots in the dimensional matrix. It is used to identify the pivots in the dimensional matrix.
    """

    # Public attributes
    # :attr: name
    name: str = ""
    """
    Name of the *PiCoefficient*. User-friendly name of the parameter.
    """

    # :attr: description
    description: str = ""
    """
    Description of the *PiCoefficient*. It is a small summary of the parameter.
    """

    # :attr: _relevance
    relevance: bool = False
    """
    Boolean value indicating if the *PiCoefficient* is relevant or not. It is used to identify whether the parameter is inside the main dimensional matrix or not.
    """

    def __post_init__(self) -> None:
        """__post_init__ _summary_
        """
        # TODO implement this method
        # TODO write docstring
        # TODO check if I really need this!!!
        pass

    @property
    def idx(self) -> int:
        """*idx* property to get *PiCoefficient* index in the dimensional model.

        Returns:
            str: ID of the *PiCoefficient*.
        """
        return self._idx

    @idx.setter
    def idx(self, value: str) -> None:
        """*idx* property to set the *PiCoefficient* index in the dimensional model. It must be an integer.

        Args:
            value (str): ID of the *PiCoefficient*.

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
        """*sym* property to get the symbol of the *PiCoefficient*.

        Returns:
            str: Symbol of the *PiCoefficient*. It is a string with the FDU formula of the parameter. i.e.: V, d, D, m, Q, \\rho, etc.
        """
        return self._sym

    @sym.setter
    def sym(self, value: str) -> None:
        """*sym* property to set the symbol of *PiCoefficient*. It must be alphanumeric (preferably a single character, a Latin or Greek letter).

        Args:
            value (str): Symbol of the *PiCoefficient*. . i.e.: V, d, D, m, Q, \\rho, etc.

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
        """*fwk* property to get the framework of the *PiCoefficient*.

        Returns:
            str: Framework of the *PiCoefficient*. It can be one of the following: `PHYSICAL`, `COMPUTATION`, `DIGITAL` or `CUSTOM`.
        """
        return self._fwk

    @fwk.setter
    def fwk(self, value: str) -> None:
        """*fwk* property of the framework of the *PiCoefficient*. It must be one of the following: `PHYSICAL`, `COMPUTATION`, `DIGITAL` or `CUSTOM`.

        Args:
            value (str): Framework of the *PiCoefficient*. Must be the same as the FDU framework.

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
        """*cat* property to get the category of the *PiCoefficient*.

        Returns:
            str: Category of the *PiCoefficient*.
        """
        return self._cat

    @cat.setter
    def cat(self, value: str) -> None:
        """*cat* property to set the category of the *PiCoefficient*.

        Args:
            value (str): Category of the *PiCoefficient*. It can must one of the following: `COMPUTED` or `DERIVED`. It is used to identify the origin of the coefficient.

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
    def param_lt(self) -> List[str]:
        """*param_lt* property to get the list of parameters used in the *PiCoefficient*.

        Returns:
            List[Parameter]: List of parameters used in the *PiCoefficient*.
        """
        return self._param_lt

    @param_lt.setter
    def param_lt(self, value: List[str]) -> None:
        """*param_lt* property to set the list of parameters used in the *PiCoefficient*. It must be a list of `Parameter` objects.

        Args:
            value (List[Parameter]): List of parameters used in the *PiCoefficient*.
        """
        # validating input parameter list
        self._validate_param_lt(value)
        # setting the new parameter list
        self._param_lt = value

    def _validate_param_lt(self, param_lt: List[str]) -> None:
        """*_validate_param_lt* validates the list of parameters used in the *PiCoefficient*.

        Args:
            param_lt (List[str]): List of parameters used in the *PiCoefficient*.

        Raises:
            ValueError: error if the list is not a Python list.
            ValueError: error if the list is empty.
            ValueError: error if the list is not a list of strings.
            ValueError: error if the list length is not equal to the FDU precedence list length.
        """
        if not isinstance(param_lt, list):
            _msg = "Parameters must be a list. "
            _msg += f"Provided: {type(param_lt)}"
            raise ValueError(_msg)
        if len(param_lt) == 0:
            _msg = "Parameters list cannot be empty. "
            _msg += f"Provided: {param_lt}"
            raise ValueError(_msg)
        if not all(isinstance(x, str) for x in param_lt):
            _msg = "Parameters must be a list of strings. "
            _msg += f"Provided: {[type(x).__name__ for x in param_lt]}"
            raise ValueError(_msg)
        if not len(param_lt) == len(config.WKNG_FDU_PREC_LT):
            _msg = "Parameters list length must match the FDU precedence list. "
            _msg += f"Provided: {len(param_lt)}, "
            _msg += f"Expected: {len(config.WKNG_FDU_PREC_LT)}"
            raise ValueError(_msg)

    @property
    def dim_col(self) -> List[int]:
        """*dim_col* property to get the dimensional column with the parameter's exponents of the *PiCoefficient*. It is a list of integers.

        Returns:
            List[int]: Dimensional column of the *PiCoefficient*.
        """
        return self._dim_col

    @dim_col.setter
    def dim_col(self, value: List[int]) -> None:
        """*dim_col* property to set the dimensional column with the parameter's exponents of the *PiCoefficient*. It must be a list of integers.

        Args:
            value (List[int]): Dimensional column of the *PiCoefficient*.
        """
        # validating input dimensional column
        self._validate_dim_col(value)
        self._dim_col = value

    def _validate_dim_col(self, dim_col: List[int]) -> None:
        """*_validate_dim_col* the dimensional column of the *PiCoefficient*.

        Args:
            dim_col (List[int]): Dimensional column of the *PiCoefficient*.

        Raises:
            ValueError: error if the list is not a Python list.
            ValueError: error if the list is empty.
            ValueError: error if the list is not a list of integers.
            ValueError: error if the list length is not equal to the FDU precedence list length.
        """
        if not isinstance(dim_col, list):
            _msg = "Dimensional column must be a list. "
            _msg += f"Provided: {type(dim_col)}"
            raise ValueError(_msg)
        if len(dim_col) == 0:
            _msg = "Dimensional column cannot be empty. "
            _msg += f"Provided: {dim_col}"
            raise ValueError(_msg)
        if not all(isinstance(x, int) for x in dim_col):
            _msg = "Dimensional column must be a list of integers. "
            _msg += f"Provided: {[type(x).__name__ for x in dim_col]}"
            raise ValueError(_msg)
        if not len(dim_col) == len(config.WKNG_FDU_PREC_LT):
            _msg = "Dimensional column length must match the FDU precedence list. "
            _msg += f"Provided: {len(dim_col)}, "
            _msg += f"Expected: {len(config.WKNG_FDU_PREC_LT)}"
            raise ValueError(_msg)

    def __str__(self) -> str:
        """*__str__()* get the string representation of the *PiCoefficient*.

        Returns:
            str: String representation of the *PiCoefficient*.
        """
        _attr_lt = []
        for attr, value in vars(self).items():
            # Skip private attributes starting with "__"
            if attr.startswith("__"):
                continue
            # Format attribute name and value
            _attr_name = attr.lstrip("_")
            _attr_lt.append(f"{_attr_name}={repr(value)}")
        # Format the string representation of the ArrayList class and its attributes
        _str = f"{self.__class__.__name__}({', '.join(_attr_lt)})"
        return _str

    def __repr__(self) -> str:
        """*__repr__()* get the string representation of the *PiCoefficient*.

        Returns:
            str: String representation of the *PiCoefficient*.
        """
        return self.__str__()
