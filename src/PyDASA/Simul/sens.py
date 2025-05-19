# -*- coding: utf-8 -*-
"""
Module for **Sensitivity** analysis in *PyDASA*.

This module provides the `Sensitivity` class for performing sensitivity analysis on Dimensional Coefficients and Pi Numbers.
"""
# native python modules
from typing import Optional, List, Generic, Callable, Union
from dataclasses import dataclass, field
# import inspect
import re

# Third-party modules
import numpy as np
from sympy.parsing.latex import parse_latex
from sympy import diff, lambdify
from SALib.sample.fast_sampler import sample
from SALib.analyze.fast import analyze

# Custom modules
# Dimensional Analysis modules

# Utils modules
from Src.PyDASA.Utils.dflt import T
from Src.PyDASA.Utils.dflt import parse_latex_symbols
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
class Sensitivity(Generic[T]):
    """**Sensitivity** Class for performing sensitivity analysis.

    Args:
        Generic (T): Generic type for a Python data structure.

    Returns:
        Sensitivity: An object with the following attributes:
            - _idx (int): The Index of the *Sensitivity*.
            - _sym (str): The Symbol of the *Sensitivity*.
            - _fwk (str): The Framework of the *Sensitivity*.
            - _cat (str): The Category of the *Sensitivity*.
            - _pi_expr (str): The LaTeX expression of the *Sensitivity*.
            - _sym_fun (Callable): The Sympy function of the *Sensitivity*.
            - _variables (list): The Variables of the *Sensitivity*.
            - var_val (Union[int, float, list, np.ndarray]): The Value for the *Sensitivity* analysis.
            - name (str): The Name of the *Sensitivity*.
            - description (str): The Description of the *Sensitivity*.
            - result (dict): The Report of the *Sensitivity*.
    """

    # Private attributes with validation logic
    # :attr: _idx
    _idx: int = -1
    """
    Unique identifier/index of the *Sensitivity*.
    """

    # :attr: _sym
    _sym: str = "SEN ANSYS \\Pi_{}"
    """
    Symbol of the *Sensitivity*. It is a LaTeX or an alphanumeric string (preferably a single Latin or Greek letter). The default LaTeX symbol is `SEN ANSYS \\Pi_{}`. e.g.: `SEN ANSYS \\Pi_{0}`.
    """

    # :attr: _fwk
    _fwk: str = "PHYSICAL"
    """
    Framework of the *Sensitivity*. It must be the same as the *FDU* framework. Can be: `PHYSICAL`, `COMPUTATION`, `DIGITAL` or `CUSTOM`.
    """

    # :attr: _cat`
    _cat: str = "SYMBOLIC"
    """
    Category of the *Sensitivity*. It is used to identify the type of analysis. It can be one of the following: `SYMBOLIC`, `NUMERIC`, `HYBRID` or `CUSTOM`. by default is `SYMBOLIC`.
    """

    # :attr: _pi_expr
    _pi_expr: Optional[str] = None
    """
    Symbolic expression of the *Sensitivity* formula. It is a string in the LateX format. e.g.: `\\Pi_{1} = \\frac{u* L}{\\rho}`.
    """

    # dimensional expression in sympy regex procesor
    _sym_fun: Optional[Callable] = None
    """
    Sympy function of the *Sensitivity*. It is a callable function that takes the variables as input and returns the value of the *Sensitivity*.
    """

    # Public attributes
    # :attr: _variables
    _variables: Optional[list] = None
    """
    Variables of the *Sensitivity*. It is a list with the names of the variables used in the *Sensitivity* formula. e.g.: `['u', 'L', '\\rho']`.
    """

    # :attr: var_val
    var_val: Union[int, float, list, np.ndarray] = None
    """
    Value for the *Sensitivity* analysis. for `SYMBOLIC` analysis it uses `int` or `float` values. for `NUMERIC` analysis it uses `list` or `numpy.ndarray` values.
    """

    # :attr: name
    name: str = ""
    """
    Name of the *Sensitivity*. User-friendly name of the parameter.
    """

    # :attr: description
    description: str = ""
    """
    Description of the *Sensitivity*. It is a small summary of the parameter.
    """

    # :attr: result
    result: dict = field(default_factory=dict)
    """
    Report of the *Sensitivity*. It is a dictionary with the results of the sensitivity analysis.
    """

    def __post_init__(self) -> None:
        """*__post_init__* initializes the *Sensitivity* instance."""
        # Initialize the sympy function and variables
        self.idx = self._idx
        self.sym = self._sym
        self.fwk = self._fwk
        self.cat = self._cat
        if self._pi_expr:
            self._parse_expression(self.pi_expr)

    def _parse_expression(self, expr: str) -> None:
        """*_parse_expression()* Parse the LaTeX expression into a sympy function.
        """
        # TODO fix parse_latex to accept the expression as a string
        # clean the expression, remove {, } and \\
        # self._sym_fun = parse_latex(expr)
        # self._exe_fun = lambdify(self._variables, self.sym_fun, "numpy")
        # parse the expression and generate the function
        self._sym_fun, self._variables = parse_latex_symbols(expr)
        # create sympy symbols dict for all variables
        self._variables = sorted([str(v) for v in self.sym_fun.free_symbols])
        # create sympy function
        self._exe_fun = lambdify(self._variables, self.sym_fun, "numpy")

    def analyze_symbolically(self, values: dict) -> dict:
        """
        Perform symbolic sensitivity analysis.

        Args:
            values (dict): Variable values for the analysis.

        Returns:
            dict: Sensitivity results.
        """
        self.result = {
            var: lambdify(self.variables, diff(self.sym_fun, var), "numpy")(
                *[values[v] for v in self.variables]
            )
            for var in self.variables
        }
        return self.result

    def analyze_numerically(self,
                            bounds: List[List[Union[int, float]]],
                            num_samples: int = 1000) -> dict:
        """
        Perform NUMERIC sensitivity analysis using SALib.

        Args:
            bounds (list): Bounds for the variables.
            num_samples (int): Number of samples for the analysis. Default is 1000.

        Returns:
            dict: Sensitivity results.
        """
        problem = {
            "num_vars": len(self.variables),
            "names": self.variables,
            "bounds": bounds,
        }
        self.var_val = sample(
            problem, num_samples).reshape(-1, len(self.variables))
        Y = np.apply_along_axis(lambda v: self._exe_fun(*v), 1, self.var_val)
        self.result = analyze(problem, Y)
        return self.result

    @property
    def idx(self) -> int:
        """*idx* Get the *Sensitivity* index in the program.

        Returns:
            int: ID of the *Sensitivity*.
        """
        return self._idx

    @idx.setter
    def idx(self, val: int) -> None:
        """*idx* Sets the *Sensitivity* index in the program. It must be an integer.

        Args:
            val (int): Index of the *Sensitivity*.

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
        """*sym* get the symbol of the *Sensitivity*.

        Returns:
            str: Symbol of the *Sensitivity*. It is a string with the FDU formula of the parameter. i.e.: V, d, D, m, Q, \\rho, etc.
        """
        return self._sym

    @sym.setter
    def sym(self, val: str) -> None:
        """*sym* sets the symbol of *Sensitivity*. It must be alphanumeric (preferably a single character, a Latin or Greek letter).

        Args:
            val (str): Symbol of the *Sensitivity*. . i.e.: V, d, D, m, Q, \\rho, etc.

        Raises:
            ValueError: error if the symbol is not alphanumeric.
        """
        # Regular expression to match valid LaTeX strings or alphanumeric strings
        if not (val.isalnum() or re.match(cfg.LATEX_RE, val)):
            _msg = "Symbol must be alphanumeric or a valid LaTeX string. "
            _msg += f"Provided: '{val}' "
            _msg += "Examples: 'V', 'd', '\\Pi_{0}', '\\rho'."
            raise ValueError(_msg)
        self._sym = val

    @property
    def fwk(self) -> str:
        """*fwk* get the framework of the *Sensitivity*.

        Returns:
            str: Framework of the *Sensitivity*. It can be one of the following: `PHYSICAL`, `COMPUTATION`, `DIGITAL` or `CUSTOM`.
        """
        return self._fwk

    @fwk.setter
    def fwk(self, val: str) -> None:
        """*fwk* property of the framework of the *Sensitivity*. It must be one of the following: `PHYSICAL`, `COMPUTATION`, `DIGITAL` or `CUSTOM`.

        Args:
            val (str): Framework of the *Sensitivity*. Must be the same as the FDU framework.

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
        """*cat* get the category of the *Sensitivity*.

        Returns:
            str: Category of the *Sensitivity*.
        """
        return self._cat

    @cat.setter
    def cat(self, val: str) -> None:
        """*cat* sets the category of the *Sensitivity*.

        Args:
            val (str): Category of the *Sensitivity*. It can be one of the following: `SYMBOLIC`, `NUMERIC` `HYBRID` or `CUSTOM`.

        Raises:
            ValueError: error if the category is not one of the allowed values.
        """
        if val.upper() not in cfg.SENS_ANSYS_DT.keys():
            _msg = f"Invalid category: {val}. "
            _msg += "Category must be one of the following: "
            _msg += f"{', '.join(cfg.SENS_ANSYS_DT.keys())}."
            raise ValueError(_msg)
        self._cat = val.upper()

    @property
    def pi_expr(self) -> str:
        """*pi_expr* Get the LaTeX expression of the *Sensitivity*.

        Returns:
            str: LaTeX expression of the *Sensitivity*.
        """
        return self._pi_expr

    @pi_expr.setter
    def pi_expr(self, val: str) -> None:
        """*pi_expr* Sets the LaTeX expression of the *Sensitivity*. It must be a valid LaTeX expression.

        Args:
            val (str): LaTeX expression of the *Sensitivity*.

        Raises:
            ValueError: error if the LaTeX expression is not valid.
        """
        # Regular expression to match valid LaTeX strings or alphanumeric strings
        if not (val.isalnum() or re.match(cfg.LATEX_RE, val)):
            _msg = "LaTeX expression must be a valid string. "
            _msg += f"Provided: '{val}' "
            _msg += "Examples: 'V', 'd', '\\Pi_{0}', '\\rho'."
            # FIXME REGEX not working!!!!
            # raise ValueError(_msg)
        self._pi_expr = val
        # automatically parse the expression and generate the function
        self._parse_expression(self._pi_expr)

    @property
    def variables(self) -> dict:
        """*variables* Get the variables of the *Sensitivity*.

        Returns:
            dict: Dimensional expression of the *Sensitivity*.
        """
        return self._variables

    @variables.setter
    def variables(self, val: list) -> None:
        """*variables* Sets the variables of the *Sensitivity*. It must be a list with the variable names.

        Args:
            val (dict): Dimensional expression of the *Sensitivity*.

        Raises:
            ValueError: error if the dimensional expression is not valid.
        """
        if not isinstance(val, list):
            _msg = "Dimensional expression must be a list. "
            _msg += f"Provided: {type(val)}"
            raise ValueError(_msg)
        self._variables = val

    @property
    def sym_fun(self) -> Callable:
        """*sym_fun* Get the sympy function of the *Sensitivity*.

        Returns:
            Callable: Sympy function of the *Sensitivity*.
        """
        return self._sym_fun

    @sym_fun.setter
    def sym_fun(self, val: Callable) -> None:
        """*sym_fun* Sets the sympy function of the *Sensitivity*. It must be a valid sympy function.

        Args:
            val (Callable): Sympy function of the *Sensitivity*.

        Raises:
            ValueError: error if the sympy function is not valid.
        """
        if not callable(val):
            _msg = "Sympy function must be callable. "
            _msg += f"Provided: {type(val)}"
            raise ValueError(_msg)
        self._sym_fun = val

    def clear(self) -> None:
        """*clear* clears the *Sensitivity* instance."""
        self._idx = -1
        self._sym = "SEN ANSYS \\Pi_{}"
        self._fwk = "PHYSICAL"
        self._cat = "SYMBOLIC"
        self._pi_expr = None
        self._variables = None
        self.var_val = None
        self.name = ""
        self.description = ""
        self.result = {}

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
