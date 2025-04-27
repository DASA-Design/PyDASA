# -*- coding: utf-8 -*-
"""
Module for the **Sensitivity** analysis in *PyDASA*.

*Sensitivity* hold the variables and methods for performing sensitivity analysis on the Dimensional Coefficients and Pi Numbers.
"""
# native python modules
from typing import Optional, List, Generic, Callable, Union
from dataclasses import dataclass, field
import inspect
import re

# Third-party modules
import numpy as np
import sympy as sp
from sympy.parsing.latex import parse_latex
from sympy import symbols, diff, lambdify
import SALib
from SALib.sample.fast_sampler import sample
from SALib.analyze.fast import analyze

# Custom modules
# Dimensional Analysis modules
from Src.PyDASA.Measure.fdu import FDU
from Src.PyDASA.Measure.params import Variable
from Src.PyDASA.Pi.coef import PiCoefficient, PiNumber

# Data Structures
from Src.PyDASA.DStruct.Tables.scht import SCHashTable

# Utils modules
from Src.PyDASA.Util.dflt import T
from Src.PyDASA.Util.err import error_handler as _error
from Src.PyDASA.Util.err import inspect_name as _insp_var

# import the 'cfg' module to allow global variable edition
from Src.PyDASA.Util import cfg

# checking custom modules
assert _error
assert _insp_var
assert cfg
assert T


@dataclass
class Sensitivity(Generic[T]):
    """
    Class to store the results of the sensitivity analysis.
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
    Category of the *Sensitivity*. It is used to identify the type of analysis. It can be one of the following: `SYMBOLIC`, `NUMERICAL`, `HYBRID` or `CUSTOM`. by default is `SYMBOLIC`.
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
    Value for the *Sensitivity* analysis. for `SYMBOLIC` analysis it uses `int` or `float` values. for `NUMERICAL` analysis it uses `list` or `numpy.ndarray` values.
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

    # :attr: report
    report: dict = field(default_factory=dict)
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
        self.pi_expr = self._pi_expr
        self.variables = self._variables
        self.name = self.name
        self.description = self.description
        self.report = self.report

    def _parce_expr(self, expr: str) -> None:
        """*parse_expr* parses the LaTeX expression into a sympy expression.

        Args:
            expr (str): The LaTeX expression to convert.
        """
        # Parse the LaTeX expression into a sympy expression
        self._sym_fun = parse_latex(expr)

    def _extract_variables(self) -> None:
        """*extract_variables* extracts variables from the sympy expression.
        """
        # Extract variables from the sympy expression
        self.variables = sorted(self.sym_fun.free_symbols, key=lambda s: s.name)
        self.variables = [str(v) for v in self.variables]

    def _generate_function(self) -> None:
        """*generate_function* generates a callable function using lambdify.
        """
        # Generate a callable function using lambdify
        self._exe_fun = lambdify(self.variables, self._sym_fun, "numpy")

    def analyze_symbolically(self, values: dict) -> dict:
        """*analyze_symbolically* performs symbolic sensitivity analysis.

        Args:
            values (dict): Dictionary with the variable values for the analysis.

        Returns:
            dict: Dictionary with the results of the sensitivity analysis.
        """
        sens = {}
        for p in self.variables:
            part_der = diff(self._sym_fun, p)
            derf = lambdify(self.variables, part_der, "numpy")
            # print(f"values[p]: {variables.get(p)}")
            sens_v = derf(*[values[p] for p in self.variables])
            sens[p] = sens_v
        self.report = sens
        return sens

    def analyze_numerically(self, bounds: list[Union[int, float]],
                            num_samples: int = 1000) -> None:

        problem = {
            'num_vars': len(self.variables),
            'names': self.variables,
            'bounds': [bounds] * len(self.variables),
        }

        self.var_val = sample(problem, num_samples)
        # Reshape the samples to match the expected input format for the custom function
        self.var_val = self.var_val.reshape(-1, len(self.variables))
        # Evaluate the custom function for all samples
        Y = np.apply_along_axis(lambda j: self._exe_fun(*j), 1, self.var_val)
        # Perform sensitivity analysis using FAST
        n_ansys = analyze(problem, Y)
        self.report = n_ansys
        return n_ansys

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
        if not (val.isalnum() or re.match(cfg.LATEX_REGEX, val)):
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
            val (str): Category of the *Sensitivity*. It can be one of the following: `SYMBOLIC`, `NUMERICAL` `HYBRID` or `CUSTOM`.

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
        if not (val.isalnum() or re.match(cfg.LATEX_REGEX, val)):
            _msg = "LaTeX expression must be a valid string. "
            _msg += f"Provided: '{val}' "
            _msg += "Examples: 'V', 'd', '\\Pi_{0}', '\\rho'."
            # FIXME REGEX not working!!!!
            # raise ValueError(_msg)
        self._pi_expr = val
        # automatically parse the expression and generate the function
        self._parce_expr(self._pi_expr)
        self._extract_variables()
        self._generate_function()

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
        #  TODO check if the function is a valid sympy function
        if not callable(val):
            _msg = "Sympy function must be callable. "
            _msg += f"Provided: {type(val)}"
            raise ValueError(_msg)
        self._sym_fun = val

    def clear(self) -> None:
        """*clear* clears the *Sensitivity* instance."""
        self._idx = -1
        self._sym = "\\Pi_{}"
        self._fwk = "PHYSICAL"
        self._cat = "SYMBOLIC"
        self._pi_expr = None
        self._variables = None
        self.var_val = None
        self.name = ""
        self.description = ""
        self.report = {}

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





















