# -*- coding: utf-8 -*-
"""
Module for representing Dimensionless Coefficients (DN) in *PyDASA*. Defines the *PiCoefficient* class, which models DN in dimensional analysis.

*IMPORTANT:* Based on the theory from:

    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""

# native python modules
from typing import Optional, List, Generic
from dataclasses import dataclass, field
import re

# custom modules
from Src.PyDASA.Utils.dflt import T
from Src.PyDASA.Utils.err import error_handler as error

# importing PyDASA's regex for managing FDUs
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
            - `relevance`: Boolean flag indicating if the *PiCoefficient* is relevant or not.
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
    _pivot_lt: Optional[List[int]] = None
    """
    List of pivot indices of the dimensional matrix. It is a list of integers representing the indices of the pivots in the dimensional matrix. It is used to identify the pivots in the dimensional matrix.
    """

    # :attr: _pi_expr
    _pi_expr: Optional[str] = None
    """
    Symbolic expression of the *PiCoefficient* formula. It is a string in the LateX format. e.g.: `\\Pi_{1} = \\frac{u* L}{\\rho}`.
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
    Boolean indicating if the *PiCoefficient* is relevant or not. It is used to identify whether the parameter is inside the main dimensional matrix or not.
    """

    def __post_init__(self) -> None:
        """__post_init__ _summary_
        """
        # force check all properties and attrs after the object is created
        self.idx = self._idx
        self.sym = self._sym
        self.fwk = self._fwk
        self.cat = self._cat
        print(type(self._param_lt), self._param_lt)
        self.param_lt = self._param_lt
        self.dim_col = self._dim_col
        self._pivot_lt = self._pivot_lt
        self.pi_expr = self._build_expression(self._param_lt, self._dim_col)

    def _validate_list(self, lt: List, exp_type: List[type], name: str) -> bool:
        """*_validate_list()* validates the list of parameters used in the *PiCoefficient* with the expected type.

        Args:
            lt (List): list to validate.
            exp_type (List[type]): expected possible types of the list elements.
            exp_type (type): expected type of the list elements.
            name (str): name of the list to validate.

        Raises:
            ValueError: if the list is not a Python list.
            ValueError: if the elements of the list are not of the expected type.
            ValueError: if the list is empty.
            ValueError: if the list length is not equal to the FDU precedence list length.

        Returns:
            bool: True if the list is valid, Raise ValueError otherwise.
        """
        if not isinstance(lt, list):
            _msg = f"{name} must be a list. "
            _msg += f"Provided: {type(lt)}"
            raise ValueError(_msg)
        if not all(isinstance(x, exp_type) for x in lt):
            _msg = f"{name} must contain {exp_type.__name__} elements. "
            _msg += f"Provided: {[type(x).__name__ for x in lt]}"
            raise ValueError(_msg)
        if len(lt) == 0:
            _msg = f"{name} cannot be empty. "
            _msg += f"Provided: {lt}"
            raise ValueError(_msg)
        return True

    def _build_expression(self, param_lt: List[str], dim_col: List[int]) -> str:
        """*_build_expression()* builds the LaTeX format expression for *PiCoefficient* using the parameter symbols and the their respective dimensions.

        Args:
            param_lt (List[str]): List of parameter symbols used in the *PiCoefficient*.
            dim_col (List[int]): List of integers representing the exponents of the parameters used to calculate the *PiCoefficient*.

        Raises:
            ValueError: if the length of the parameter list and the dimensional column are not equal.

        Returns:
            str: LaTeX format expression for the *PiCoefficient*. e.g.: `\\Pi_{1} = \\frac{u* L}{\\rho}`.
        """
        print("param_lt", param_lt, "dim_col", dim_col)
        if len(param_lt) != len(dim_col):
            _msg = "Invalid input length. "
            _msg += "Parameters and dimensional size must be the same. "
            _msg += f"Parameters: {len(param_lt)}, "
            _msg += f"Dimensional column: {len(dim_col)}"
            raise ValueError(_msg)

        mumerator = []
        denominator = []
        for sym, exp in zip(param_lt, dim_col):
            if exp > 0:
                part = f"{sym}" if exp == 1 else f"{sym}^{exp}"
                mumerator.append(part)
            elif exp < 0:
                part = f"{sym}" if exp == -1 else f"{sym}^{-exp}"
                denominator.append(part)
            # ignore zero exponents
        num_str = "1" if not mumerator else "*".join(mumerator)
        if not denominator:
            return f"${num_str}$"
        else:
            denom_str = "*".join(denominator)
            return f"$\\frac{{{num_str}}}{{{denom_str}}}$"

    @property
    def idx(self) -> int:
        """*idx* property to get *PiCoefficient* index in the dimensional model.

        Returns:
            str: ID of the *PiCoefficient*.
        """
        return self._idx

    @idx.setter
    def idx(self, val: int) -> None:
        """*idx* property to set the *PiCoefficient* index in the dimensional model. It must be an integer.

        Args:
            val (int): ID of the *PiCoefficient*.

        Raises:
            ValueError: error if the Index is not an integer.
        """
        if not isinstance(val, int):
            _msg = "Index must be an integer, "
            _msg += f"Provided type: {type(val)}"
            raise ValueError(_msg)
        self._idx = val

    @property
    def sym(self) -> str:
        """*sym* property to get the symbol of the *PiCoefficient*.

        Returns:
            str: Symbol of the *PiCoefficient*. It is a string with the FDU formula of the parameter. i.e.: V, d, D, m, Q, \\rho, etc.
        """
        return self._sym

    @sym.setter
    def sym(self, val: str) -> None:
        """*sym* property to set the symbol of *PiCoefficient*. It must be alphanumeric (preferably a single character, a Latin or Greek letter).

        Args:
            val (str): Symbol of the *PiCoefficient*. . i.e.: V, d, D, m, Q, \\rho, etc.

        Raises:
            ValueError: error if the symbol is not alphanumeric.
        """
        # Regular expression to match valid LaTeX strings or alphanumeric strings
        if not (val.isalnum() or re.match(config.LATEX_REGEX, val)):
            _msg = "Symbol must be alphanumeric or a valid LaTeX string. "
            _msg += f"Provided: '{val}' "
            _msg += "Examples: 'V', 'd', '\\Pi_{0}', '\\rho'."
            raise ValueError(_msg)
        self._sym = val

    @property
    def fwk(self) -> str:
        """*fwk* property to get the framework of the *PiCoefficient*.

        Returns:
            str: Framework of the *PiCoefficient*. It can be one of the following: `PHYSICAL`, `COMPUTATION`, `DIGITAL` or `CUSTOM`.
        """
        return self._fwk

    @fwk.setter
    def fwk(self, val: str) -> None:
        """*fwk* property of the framework of the *PiCoefficient*. It must be one of the following: `PHYSICAL`, `COMPUTATION`, `DIGITAL` or `CUSTOM`.

        Args:
            val (str): Framework of the *PiCoefficient*. Must be the same as the FDU framework.

        Raises:
            ValueError: If the framework is not one of the allowed values.
        """
        if val not in config.FDU_FWK_DT.keys():
            _msg = f"Invalid framework: {val}. "
            _msg += "Framework must be one of the following: "
            _msg += f"{', '.join(config.FDU_FWK_DT.keys())}."
            raise ValueError(_msg)
        self._fwk = val

    @property
    def cat(self) -> str:
        """*cat* property to get the category of the *PiCoefficient*.

        Returns:
            str: Category of the *PiCoefficient*.
        """
        return self._cat

    @cat.setter
    def cat(self, val: str) -> None:
        """*cat* property to set the category of the *PiCoefficient*.

        Args:
            val (str): Category of the *PiCoefficient*. It can must one of the following: `COMPUTED` or `DERIVED`. It is used to identify the origin of the coefficient.

        Raises:
            ValueError: error if the category is not one of the allowed values.
        """
        if val.upper() not in config.DC_CAT_DT.keys():
            _msg = f"Invalid category: {val}. "
            _msg += "Category must be one of the following: "
            _msg += f"{', '.join(config.DC_CAT_DT.keys())}."
            raise ValueError(_msg)
        self._cat = val.upper()

    @property
    def param_lt(self) -> List[str]:
        """*param_lt* property to get the list of parameters used in the *PiCoefficient*.

        Returns:
            List[Parameter]: List of parameters used in the *PiCoefficient*.
        """
        return self._param_lt

    @param_lt.setter
    def param_lt(self, val: List[str]) -> None:
        """*param_lt* property to set the list of parameters used in the *PiCoefficient*. It must be a list of `Parameter` objects.

        Args:
            val (List[Parameter]): List of parameters used in the *PiCoefficient*.
        """
        # if the list is valid, set the parameter list
        if self._validate_list(val, (str,), "param_lt"):
            self._param_lt = val

    @property
    def dim_col(self) -> List[int]:
        """*dim_col* property to get the dimensional column with the parameter's exponents of the *PiCoefficient*. It is a list of integers.

        Returns:
            List[int]: Dimensional column of the *PiCoefficient*.
        """
        return self._dim_col

    @dim_col.setter
    def dim_col(self, val: List[int]) -> None:
        """*dim_col* property to set the dimensional column with the parameter's exponents of the *PiCoefficient*. It must be a list of integers.

        Args:
            val (List[int]): Dimensional column of the *PiCoefficient*.
        """
        # if the dimensional column is valid, set the dimensional column
        if self._validate_list(val, (int, float), "dim_col"):
            self._dim_col

    @property
    def pi_expr(self) -> str:
        """*pi_expr* property to get the symbolic expression of the *PiCoefficient*.

        Returns:
            str: Symbolic expression of the *PiCoefficient*.
        """
        return self._pi_expr

    @pi_expr.setter
    def pi_expr(self, val: str) -> None:
        """*pi_expr* property to set the symbolic expression of the *PiCoefficient*. It must be a string in the LaTeX format.

        Args:
            val (str): Symbolic expression of the *PiCoefficient*. It must be a string in the LaTeX format.

        Raises:
            ValueError: error if the symbolic expression is not in the LaTeX format.
        """
        if not isinstance(val, str):
            _msg = "Symbolic expression must be a string. "
            _msg += f"Provided: {type(val)}"
            raise ValueError(_msg)
        self._pi_expr = val

    def __str__(self) -> str:
        """*__str__()* get the string representation of the *PiCoefficient*.

        Returns:
            str: String representation of the *PiCoefficient*.
        """
        _attr_lt = []
        for attr, val in vars(self).items():
            # Skip private attributes starting with "__"
            if attr.startswith("__"):
                continue
            # Format attribute name and value
            _attr_name = attr.lstrip("_")
            _attr_lt.append(f"{_attr_name}={repr(val)}")
        # Format the string representation of the ArrayList class and its attributes
        _str = f"{self.__class__.__name__}({', '.join(_attr_lt)})"
        return _str

    def __repr__(self) -> str:
        """*__repr__()* get the string representation of the *PiCoefficient*.

        Returns:
            str: String representation of the *PiCoefficient*.
        """
        return self.__str__()


@dataclass
class PiNumber(PiCoefficient[T]):
    pass