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

# Third-party modules
import numpy as np

# custom modules
from Src.PyDASA.Utils.dflt import T
from Src.PyDASA.Utils.err import error_handler as _error
from Src.PyDASA.Utils.err import inspect_name as _insp_var

# import the 'cfg' module to allow global variable edition
from Src.PyDASA.Utils import cfg

# checking custom modules
assert _error
assert _insp_var
assert cfg
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
    Unique identifier/index of the *PiCoefficient*. It is the order of in which the coefficient is calculated in the dimensional model.
    """

    # :attr: _sym
    _sym: str = "\\Pi_{}"
    """
    Symbol of the *PiCoefficient*. It is a LaTeX or an alphanumeric string (preferably a single Latin or Greek letter). It is used for user-friendly representation of the instance. The default LaTeX symbol is `\\Pi_{}`. e.g.: `\\Pi_{1}`.
    """

    # :attr: _fwk
    _fwk: str = "PHYSICAL"
    """
    Framework of the *PiCoefficient*. It must be the same as the FDU framework. It must be the same as the *FDU* and *Parameter* framework. It must be the same as the FDU framework. Can be: `PHYSICAL`, `COMPUTATION`, `DIGITAL` or `CUSTOM`.
    """

    # :attr: _cat`
    _cat: str = "COMPUTED"
    """
    Category of the *PiCoefficient*. It is used to identify the origin of the coefficient. Can be `COMPUTED` or `DERIVED` for the coefficient computed from the Buckingham Pi in the dimensional matrix and those derived from other `COMPUTED` coefficients.
    """

    _param_lt: List[str] = field(default_factory=list)
    """
    Parameter symbols used in the *PiCoefficient*. It is a list of `str` objects to identify the parameters used to calculate the coefficient.
    """

    # :attr: _dim_col
    _dim_col: List[int] = field(default_factory=list)
    """
    Dimensional Column (list) of the *PiCoefficient*. It is a list of integers representing the exponents of the *Parameters* used to calculate the Dimensionless Coefficient. i.e.: `[1, 2, -1]`.
    """

    # :attr: _pivot_lt
    _pivot_lt: Optional[List[int]] = None
    """
    List of pivot indices of the Dimensional Matrix. It represent the indices of the lineary independent parameters in the dimensional matrix.
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

    # :attr: relevance
    relevance: bool = True
    """
    Boolean indicating if the *PiCoefficient* is relevant or not. It is used to identify whether the parameter is inside the main dimensional matrix or not.
    """

    # :attr: pi_dims
    pi_dims: Optional[dict] = None
    """
    Dimensional parameter exponents in the *PiCoefficient*. It is a dictionary for the parameter exponents used to calculate the coefficient. e.g.: `{'u': 1, 'L': 1, '\\rho': -1}`.
    """

    def __post_init__(self) -> None:
        """*__post_init__()* Initializes the *PiCoefficient* object and builds the LaTeX expression for the coefficient using the parameter symbols and their respective dimensional exponents.
        """
        # force check all properties and attrs after the object is created
        self.idx = self._idx
        self.sym = self._sym
        self.fwk = self._fwk
        self.cat = self._cat
        self.param_lt = self._param_lt
        self.dim_col = self._dim_col
        self._pivot_lt = self._pivot_lt
        self.pi_expr, self.pi_dims = self._build_expression(self._param_lt,
                                                            self._dim_col)

    def _validate_list(self, lt: List, exp_type: List[type]) -> bool:
        """*_validate_list()* validates the list of parameters used in the *PiCoefficient* with the expected type.

        Args:
            lt (List): list to validate.
            exp_type (List[type]): expected possible types of the list elements.

        Raises:
            ValueError: if the list is not a Python list.
            ValueError: if the elements of the list are not of the expected type.
            ValueError: if the list is empty.

        Returns:
            bool: True if the list is valid, Raise ValueError otherwise.
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
        """_build_expression()* builds the LaTeX format expression for *PiCoefficient* using the parameter symbols and the their respective dimensions.

        Args:
            param_lt (List[str]): List of parameter symbols used in the *PiCoefficient*.
            dim_col (List[int]): List of integers representing the exponents of the parameters used to calculate the *PiCoefficient*.

        Raises:
            ValueError:  if the length of the parameter list and the dimensional column are not equal.

        Returns:
            tuple[str, dict]: LaTeX format expression for the *PiCoefficient* and a dictionary with the parameters and their exponents. e.g.: `\\Pi_{1} = \\frac{u* L}{\\rho}`, `{'u': 1, 'L': 1, '\\rho': -1}`.
        """
        # check if the parameter list and the dimensional column are valid
        if len(param_lt) != len(dim_col):
            _msg = "Invalid input length. "
            _msg += "Parameters and dimensional size must be the same. "
            _msg += f"Parameters: {len(param_lt)}, "
            _msg += f"Dimensional column: {len(dim_col)}"
            raise ValueError(_msg)

        # working variables
        mumerator = []
        denominator = []
        parameters = {}

        # iterate over the parameters and their respective dimensions
        # and build the LaTeX expression
        for sym, exp in zip(param_lt, dim_col):
            # check if the symbol is a valid LaTeX string for the numerator
            if exp > 0:
                part = f"{sym}" if exp == 1 else f"{sym}^{exp}"
                mumerator.append(part)
            # check if the symbol is a valid LaTeX string for the denominator
            elif exp < 0:
                part = f"{sym}" if exp == -1 else f"{sym}^{-exp}"
                denominator.append(part)
            # ignore zero exponents
            else:
                continue
            # add the symbol to the parameters dictionary
            parameters[sym] = exp
        # check if the numerator is empty
        num_str = "1" if not mumerator else "*".join(mumerator)
        # if the denominator is empty, return the numerator only
        if not denominator:
            return f"${num_str}$"
        # otherwise, build the LaTeX expression for the denominator
        else:
            # build the LaTeX expression for the denominator
            denom_str = "*".join(denominator)
            # create the LaTeX expression for the whole coefficient
            return f"$\\frac{{{num_str}}}{{{denom_str}}}$", parameters

    def clear(self) -> None:
        """*clear()* Resets all attributes to their default values in the *PiCoefficient* object.
        """
        self._idx = -1
        self._sym = "\\Pi_{}"
        self._fwk = "PHYSICAL"
        self._cat = "COMPUTED"
        self._param_lt = []
        self._dim_col = []
        self._pivot_lt = None
        self._pi_expr = None
        self.name = ""
        self.description = ""
        self.relevance = True
        self.pi_dims = None

    @property
    def idx(self) -> int:
        """*idx* get *PiCoefficient* index in the dimensional model.

        Returns:
            str: ID of the *PiCoefficient*.
        """
        return self._idx

    @idx.setter
    def idx(self, val: int) -> None:
        """*idx* sets the *PiCoefficient* index in the dimensional model. It must be an integer.

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
        """*sym* get the symbol of the *PiCoefficient*.

        Returns:
            str: Symbol of the *PiCoefficient*. It is a string with the FDU formula of the parameter. i.e.: V, d, D, m, Q, \\rho, etc.
        """
        return self._sym

    @sym.setter
    def sym(self, val: str) -> None:
        """*sym* sets the symbol of *PiCoefficient*. It must be alphanumeric (preferably a single character, a Latin or Greek letter).

        Args:
            val (str): Symbol of the *PiCoefficient*. . i.e.: V, d, D, m, Q, \\rho, etc.

        Raises:
            ValueError: error if the symbol is not alphanumeric.
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
        """*fwk* get the framework of the *PiCoefficient*.

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
        if val not in cfg.FDU_FWK_DT.keys():
            _msg = f"Invalid framework: {val}. "
            _msg += "Framework must be one of the following: "
            _msg += f"{', '.join(cfg.FDU_FWK_DT.keys())}."
            raise ValueError(_msg)
        self._fwk = val

    @property
    def cat(self) -> str:
        """*cat* get the category of the *PiCoefficient*.

        Returns:
            str: Category of the *PiCoefficient*.
        """
        return self._cat

    @cat.setter
    def cat(self, val: str) -> None:
        """*cat* sets the category of the *PiCoefficient*.

        Args:
            val (str): Category of the *PiCoefficient*. It can must one of the following: `COMPUTED` or `DERIVED`. It is used to identify the origin of the coefficient.

        Raises:
            ValueError: error if the category is not one of the allowed values.
        """
        if val.upper() not in cfg.DC_CAT_DT.keys():
            _msg = f"Invalid category: {val}. "
            _msg += "Category must be one of the following: "
            _msg += f"{', '.join(cfg.DC_CAT_DT.keys())}."
            raise ValueError(_msg)
        self._cat = val.upper()

    @property
    def param_lt(self) -> List[str]:
        """*param_lt* get the list of parameters used in the *PiCoefficient*.

        Returns:
            List[Parameter]: List of parameters used in the *PiCoefficient*.
        """
        return self._param_lt

    @param_lt.setter
    def param_lt(self, val: List[str]) -> None:
        """*param_lt* sets the list of parameters used in the *PiCoefficient*. It must be a list of *Parameter* objects.

        Args:
            val (List[Parameter]): List of parameters used in the *PiCoefficient*.
        """
        # if the list is valid, set the parameter list
        if self._validate_list(val, (str,)):
            self._param_lt = val

    @property
    def dim_col(self) -> List[int]:
        """*dim_col* get the dimensional column with the parameter's exponents of the *PiCoefficient*. It is a list of integers.

        Returns:
            List[int]: Dimensional column of the *PiCoefficient*.
        """
        return self._dim_col

    @dim_col.setter
    def dim_col(self, val: List[int]) -> None:
        """*dim_col* sets the dimensional column with the parameter's exponents of the *PiCoefficient*. It must be a list of integers.

        Args:
            val (List[int]): Dimensional column of the *PiCoefficient*.
        """
        # if the dimensional column is valid, set the dimensional column
        if self._validate_list(val, (int, float)):
            self._dim_col

    @property
    def pi_expr(self) -> str:
        """*pi_expr* get the symbolic expression of the *PiCoefficient*.

        Returns:
            str: Symbolic expression of the *PiCoefficient*.
        """
        return self._pi_expr

    @pi_expr.setter
    def pi_expr(self, val: str) -> None:
        """*pi_expr* sets the symbolic expression of the *PiCoefficient*. It must be a string in the LaTeX format.

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

    # Private attributes with validation logic
    # :attr: _min
    _min: Optional[float] = None
    """
    Minimum value the Dimensionless Number, based on associated variable's minimum range (std_min).

    """

    # :attr: _max
    _max: Optional[float] = None
    """
    Maximum value of the Dimensionless Number, based on associated variable's maximum range (std_max).
    """

    # :attr: _avg
    _avg: Optional[float] = None
    """
    Average value of the Dimensionless Number, based on associated variable's average range (std_avg = (std_min + std_max) / 2).
    """

    # :attr: _step
    _step: Optional[float] = -1.0
    """
    Step value of the Dimensionless Number, based on associated variable's step range (std_step).
    """

    # :attr: _data
    _data: Optional[np.ndarray] = np.array([])
    """
    Data of the *PiNumber*. It is a numpy array of the possible values of the Dimensionless Number.
    """

    @property
    def min(self) -> Optional[float]:
        """*min* Get the minimum range of the *PiNumber*.

        Returns:
            Optional[float]: minimum range of the *PiNumber*.
        """
        return self._min

    @min.setter
    def min(self, val: Optional[float]) -> None:
        """*min* Sets the minimum range of the *PiNumber*.

        Args:
            val (Optional[float]): Minimum range of the *PiNumber*.

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
        """*max* Get the maximum range of the *PiNumber*.

        Returns:
            Optional[float]: maximum range of the *PiNumber*.
        """
        return self._max

    @max.setter
    def max(self, val: Optional[float]) -> None:
        """*max* Sets the maximum range of the *PiNumber*.

        Args:
            val (Optional[float]): maximum range of the *PiNumber*.

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
    def step(self) -> Optional[float]:
        """*step* Get the step of the *PiNumber*.
        It is used for sensitivity analysis and simulations.

        Returns:
            Optional[float]: step of the *PiNumber*.
        """
        return self._step

    @step.setter
    def step(self, val: Optional[float]) -> None:
        """*step* Sets the step of the *PiNumber*. It must be a number.

        Args:
            val (Optional[float]): step of the *PiNumber*.

        Raises:
            ValueError: If the step is not a number.
            ValueError: If the step is zero.
            ValueError: If the step is greater than or equal to the range of values.
        """
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError("Step must be a number.")
        if val == 0:
            raise ValueError("Step cannot be zero.")
        if val >= self._max - self._min:
            _msg = f"Step {val} cannot be greater than or equal to"
            _msg = f" the range of values {self._max - self._min}."
            _msg += f"between {self._min} and {self._max}."
            raise ValueError(_msg)
        self._step = val

    @property
    def data(self) -> np.ndarray:
        """*data* Get the data array for the Variable.

        Returns:
            np.ndarray: Data array for the Variable.
        """
        return self.data

    @data.setter
    def data(self, val: Optional[np.ndarray]) -> None:
        """*data* Set the data array for the Variable." if `val` is None, it will be set to the range of values between `min` and `max` with a step of `step`.

        Args:
            val (Optional[np.ndarray]): Data array for the Variable.

        Raises:
            ValueError: if the data range is not a numpy array.
        """
        if val is None:
            self.data = np.arange(self.min,
                                  self.max,
                                  self.step)
        elif not isinstance(val, np.ndarray):
            _msg = "Invalid data type. "
            _msg += "Range must be a numpy array."
            _msg += f" Provided: {type(val)}."
            raise ValueError(_msg)
        else:
            self.data = val

    def clear(self) -> None:
        """*clear()* Resets all attributes to their default values in the *PiNumber* object. It extends from *PiCoefficient* class.
        """
        super().clear()
        self._min = None
        self._max = None
        self._avg = None
        self._step = -1.0
        self._data = np.array([])

    def __str__(self) -> str:
        """*__str__()* Returns a string representation of the *PiNumber*. It extends from *PiCoefficient* class.

        Returns:
            str: string representation of the *PiNumber*.
        """
        _str = super().__str__()
        return _str

    def __repr__(self) -> str:
        """*__repr__()* Returns a string representation of the *PiNumber*. It extends from *PiCoefficient* class.

        Returns:
            str: string representation of the *PiNumber*.
        """
        _str = super().__repr__()
        return _str
