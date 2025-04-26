# -*- coding: utf-8 -*-
"""
Module for sensitivity analysis in *PyDASA*.

Uses SymPy for analytical sensitivity analysis (derivatives) and SALib for numerical sensitivity analysis (FAST).

The *SensitivityAnalysis* class computes sensitivities for *PiNumbers* based on *Variables* and ranks them in *SensitivitReport*.
"""
# native python modules
from typing import Optional, List, Generic, Callable
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

    # TODO add attributes to store the results of the analysis
    # :attr: _pi_expr
    _pi_expr: Optional[str] = None
    """
    Symbolic expression of the *PiCoefficient* formula. It is a string in the LateX format. e.g.: `\\Pi_{1} = \\frac{u* L}{\\rho}`.
    """

    # :attr: _latex_expr
    _var_lt: List[str] = field(default_factory=list)
    """
    Parameter symbols used in the *PiCoefficient*. It is a list of `str` objects to identify the parameters used to calculate the coefficient.
    """

    sympy_expr: Callable = None
    numpy_func: Callable = None

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

    report: dict = field(default_factory=dict)

    def _parce_expr(self, expr: str) -> None:
        """*parse_expr* parses the LaTeX expression into a sympy expression.

        Args:
            expr (str): The LaTeX expression to convert.
        """
        # Parse the LaTeX expression into a sympy expression
        self.sympy_expr = parse_latex(expr)

    def _extract_variables(self) -> None:
        """*extract_variables* extracts variables from the sympy expression."""
        # Extract variables from the sympy expression
        self.variables = sorted(self.sympy_expr.free_symbols, key=lambda s: s.name)
        self.variables = [str(v) for v in self.variables]

    def _generate_function(self) -> None:
        """*generate_function* generates a callable function using lambdify."""
        # Generate a callable function using lambdify
        self.numpy_func = lambdify(self.variables, self.sympy_expr, "numpy")

    def analyze_symbolically(self, variables: SCHashTable) -> None:
        sens = {}
        print(f"Analyzing Symbolically: {self.latex_expr}, {self.variables}")
        for p in self.variables:
            part_der = diff(self.sympy_expr, p)
            print(f"Partial Derivative: {part_der}")
            self.numpy_func = lambdify(self.variables, part_der, "numpy")
            print(f"values[p]: {variables.get_entry(p)}")
            sens_v = self.numpy_func(*[values[p] for p in self.variables])
            sens[p] = sens_v
        self.report = sens

    def analyze_numerically(self,
                            bounds: dict,
                            num_samples: int = 1000) -> None:
        # Generate samples using the FAST method
        p_vals = sample(bounds, num_samples)

        # Reshape the samples to match the expected input format for the custom function
        p_vals = p_vals.reshape(-1, bounds["num_vars"])

        # Evaluate the custom function for all samples
        Y = np.apply_along_axis(lambda row: self.numpy_func(*row), 1, p_vals)

        # Perform sensitivity analysis using FAST
        n_ansys = analyze(bounds, Y)
        self.report = n_ansys

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
    def latex_expr(self) -> str:
        """*latex_expr* Get the LaTeX expression of the *Sensitivity*.

        Returns:
            str: LaTeX expression of the *Sensitivity*.
        """
        return self._latex_expr

    @latex_expr.setter
    def latex_expr(self, val: str) -> None:
        """*latex_expr* Sets the LaTeX expression of the *Sensitivity*. It must be a valid LaTeX expression.

        Args:
            val (str): LaTeX expression of the *Sensitivity*.

        Raises:
            ValueError: error if the LaTeX expression is not valid.
        """
        # FIXME REGEX not working!!!!
        # Regular expression to match valid LaTeX strings or alphanumeric strings
        if not (val.isalnum() or re.match(cfg.LATEX_REGEX, val)):
            _msg = "LaTeX expression must be a valid string. "
            _msg += f"Provided: '{val}' "
            _msg += "Examples: 'V', 'd', '\\Pi_{0}', '\\rho'."
            # raise ValueError(_msg)
        self._latex_expr = val
        # automatically parse the expression and generate the function
        self._parce_expr(self._latex_expr)
        self._extract_variables()
        self._generate_function()


@dataclass
class SensitivityAnalysis(Generic[T]):

    # Private attributes with validation logic
    # :attr: _idx
    _idx: int = -1
    """
    Unique identifier/index of the *SensitivityAnalysis*.
    """

    # :attr: _sym
    _sym: str = "SEN ANSYS_{x}"
    """
    Symbol of the *SensitivityAnalysis*. It must be alphanumeric (preferably a single character + Latin or Greek letter). Useful for user-friendly representation of the instance.
    """

    # :attr: _fwk
    _fwk: str = "PHYSICAL"
    """
    Framework of the *SensitivityAnalysis*, can be one of the following: `PHYSICAL`, `COMPUTATION`, `DIGITAL` or `CUSTOM`. By default, it is set to `PHYSICAL`.
    """

    # :attr: _cat`
    _cat: str = "SYMBOLIC"
    """
    Category of the *SensitivityAnalysis*. It can be one of the following: `SYMBOLIC`, `NUMERICAL` or `CUSTOM` and decides which method to use for the analysis. By default, it is set to `SYMBOLIC`.
    """

    # :attr: _relevance_lt
    _relevance_lt: List[Variable] = field(default_factory=list)
    """
    List of relevant *Variable* objects. Need to be defined before performing the sensitivity analysis with the *DimensionalAnalyzer*.
    """

    # :attr: _coefficient_lt
    _coefficient_lt: List[PiNumber] = field(default_factory=list)
    """
    List of relevant *PiCoefficients* objects. Need to be defined before performing the sensitivity analysis with the *DimensionalSolver*.
    """

    # :attr: _relevance_mp
    _relevance_mp: Optional[SCHashTable] = None
    """
    Hash table for storing the relevant *Variable* objects. It is used to quickly access and manage the variables in the sensitivity analysis.
    """

    # :attr: _coefficient_mp
    _coefficient_mp: Optional[SCHashTable] = None
    """
    Hash table for storing *PiCoefficients* objects. It is used to quickly access and manage the Pi coefficients in the sensitivity analysis.
    """

    # :attr: _pi_numbers
    _pi_numbers: List[PiNumber] = field(default_factory=list)
    """
    List of resulting *PiNumbers* after performing the sensitivity analysis. It contains the computed Pi numbers and their corresponding sensitivity order in the `sensitivity` attribute.
    """

    # Public attributes
    # :attr: name
    name: str = "Sensitivity Analysis"
    """
    User-friendly name of the *SensitivityAnalysis*.
    """

    # :attr: description
    description: str = ""
    """
    Small summary of the *SensitivityAnalysis*.
    """

    # :attr: sensitivity_rpt
    sensitivity_rpt: Optional[List[Sensitivity]] = None
    """
    Report of the sensitivity analysis results. It contains the computed sensitivity according to the Fourier Amplitude Sensitivity Testing (FAST) method.
    """

    # TODO add the __post_init__ method to validate the attributes.
    def __post_init__(self) -> None:
        """__post_init__ _summary_
        """
        if self._relevance_mp is None:
            self._relevance_mp = SCHashTable()
        if self._coefficient_mp is None:
            self._coefficient_mp = SCHashTable()
        # initialize hash tables
        if self._relevance_lt:
            self._setup_variables_map(self._relevance_lt)
        if self._coefficient_lt:
            self._setup_coefficients_map(self._coefficient_lt)

        # validate the attributes
        self.idx = self._idx
        self.sym = self._sym
        self.fwk = self._fwk
        self.relevance_lt = self._relevance_lt
        self.coefficient_lt = self._coefficient_lt
        if self._pi_numbers:
            self.pi_numbers = self._pi_numbers

    # TODO: add private methods.
    def _setup_variables_map(self, var_lt: List[Variable]) -> None:
        """*setup_vars_map* sets up the hash table for the *Variable* objects.

        Args:
            var_lt (List[Variable]): List of *Variable* objects.
        """
        # self._relevance_mp = SCHashTable()
        for var in var_lt:
            self._relevance_mp.insert(var.sym, var)

    def _setup_coefficients_map(self, pi_lt: List[PiCoefficient]) -> None:
        """*setup_coef_map* sets up the hash table for the *PiCoefficient* objects.

        Args:
            pi_lt (List[PiCoefficient]): List of *PiCoefficient* objects.
        """
        self._coefficient_mp = SCHashTable()
        for pi in pi_lt:
            self._coefficient_mp.insert(pi.sym, pi)

    def _error_handler(self, err: Exception) -> None:
        """*_error_handler()* to process the context (package/class), function name (method), and the error (exception) that was raised to format a detailed error message and traceback.

        Args:
            err (Exception): Python raised exception.
        """
        # TODO check the usefulness of this method
        _context = self.__class__.__name__
        _function_name = inspect.currentframe().f_code.co_name
        _error(_context, _function_name, err)

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

    # TODO add public methods to get the results of the analysis.
    @property
    def idx(self) -> int:
        """*idx* Get the *SensitivityAnalysis* index in the program.

        Returns:
            int: ID of the *SensitivityAnalysis*.
        """
        return self._idx

    @idx.setter
    def idx(self, val: int) -> None:
        """*idx* Sets the *SensitivityAnalysis* index in the program. It must be an integer.

        Args:
            val (int): Index of the *SensitivityAnalysis*.

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
        """*sym* Get the symbol of the *SensitivityAnalysis*.

        Returns:
            str: Symbol of the *SensitivityAnalysis*.
        """
        return self._sym

    @sym.setter
    def sym(self, val: str) -> None:
        """*sym* Sets the symbol of the *SensitivityAnalysis*. It must be alphanumeric (preferably a single character, a Latin or Greek letter).

        Args:
            val (str): Symbol of the *SensitivityAnalysis*.

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
        """*fwk* Gets the working framework of the *SensitivityAnalysis*.

        Returns:
            str: Working framework of the *SensitivityAnalysis*.
        """
        return self._fwk

    @fwk.setter
    def fwk(self, val: str) -> None:
        """*fwk* Sets the working framework of the *SensitivityAnalysis*.

        Args:
            val (str): Worjing Framework of the *SensitivityAnalysis*. It must be a supported FDU framework

        Raises:
            ValueError: error if the framework is not valid.
        """
        if val not in cfg.FDU_FWK_DT.keys():
            _msg = f"Invalid framework: {val}. "
            _msg += "Must be one of the following: "
            _msg += f"{', '.join(cfg.FDU_FWK_DT.keys())}."
            raise ValueError(_msg)
        self._fwk = val

    @property
    def cat(self) -> str:
        """*cat* Get the category of the *SensitivityAnalysis*.

        Returns:
            str: Category of the *SensitivityAnalysis*.
        """
        return self._cat

    @cat.setter
    def cat(self, val: str) -> None:
        """*cat* Sets the category of the *SensitivityAnalysis*. It can be one of the following: `SYMBOLIC`, `NUMERICAL` or `CUSTOM`.

        Args:
            val (str): Category of the *SensitivityAnalysis*.

        Raises:
            ValueError: error if the category is not valid.
        """
        if val not in cfg.SENS_ANSYS_DT.keys():
            _msg = f"Invalid category: {val}. "
            _msg += "Must be one of the following: "
            _msg += f"{', '.join(cfg.SENS_ANSYS_DT.keys())}."
            raise ValueError(_msg)
        self._cat = val

    @property
    def relevance_lt(self) -> List[Variable]:
        """*relevance_lt* property to get the list of relevant parameters in the *SensitivityAnalysis*.

        Returns:
            List[Variable]: List of relevant *Variable* objects.
        """
        return self._relevance_lt

    @relevance_lt.setter
    def relevance_lt(self, val: List[Variable]) -> None:
        """*relevance_lt* sets the relevance list of the *SensitivityAnalysis*.

        Args:
            val (List[Variable]): List of *Variable* objects.
        """
        # if the list is valid, set the parameter list
        if self._validate_list(val, (Variable,)):
            self._relevance_lt = val

    @property
    def coefficient_lt(self) -> List[PiCoefficient]:
        """*coefficient_lt* Get the list of coefficients in the *SensitivityAnalysis*.

        Returns:
            List[PiCoefficient]: List of *PiCoefficient* objects.
        """
        return self._coefficient_lt

    @coefficient_lt.setter
    def coefficient_lt(self, val: List[PiCoefficient]) -> None:
        """*set_coefficient_lt()* sets the coefficient list of the *SensitivityAnalysis*.

        Args:
            val (List[PiCoefficient]): List of *PiCoefficient* objects.
        """
        # if the list is valid, set the parameter list
        if self._validate_list(val, (PiCoefficient,)):
            self._coefficient_lt = val

    @property
    def pi_numbers(self) -> List[PiNumber]:
        """*pi_numbers* Get the list of coefficients in the *SensitivityAnalysis*.

        Returns:
            List[PiNumber]: List of *PiNumber* objects.
        """
        return self._pi_numbers

    @pi_numbers.setter
    def pi_numbers(self, val: List[PiNumber]) -> None:
        """*set_coefficient_lt()* sets the coefficient list of the *SensitivityAnalysis*.

        Args:
            val (List[PiNumber]): List of *PiNumber* objects.
        """
        # if the list is valid, set the parameter list
        if self._validate_list(val, (PiNumber,)):
            self._pi_numbers = val

    def analyze_pi_sensitivity(self, cutoff: str = "avg") -> None:

        # prepare the sensitivity analysis
        for coef in self.coefficient_lt:
            print(f"Analyzing Pi Coefficient: {coef.sym}: {coef.pi_expr}")
            val = self._relevance_mp.get_entry(coef.sym)
            if val is None:
                _msg = f"Variable {coef.sym} not found in the relevance list."
                raise ValueError(_msg)
            
            pi_sen = Sensitivity()
            pi_sen.latex_expr = coef.pi_expr
            print(pi_sen)
            pi_sen._parce_expr(coef.pi_expr)
            pi_sen._extract_variables()
            pi_sen._generate_function()
            pi_sen.analyze_symbolically(val)
            print(pi_sen)

    def __str__(self) -> str:
        """*__str__()* returns a string representation of the *SensitivityAnalysis* object.

        Returns:
            str: String representation of the *SensitivityAnalysis* object.
        """
        _attr_lt = []
        for attr, val in vars(self).items():
            # Skip private attributes starting with "__"
            # if attr.startswith("__"):
            #    continue
            # Format attribute name and value
            _attr_name = attr.lstrip("_")
            _attr_lt.append(f"{_attr_name}={repr(val)}")
        # Format the string representation of the ArrayList class and its attributes
        _str = f"{self.__class__.__name__}({', '.join(_attr_lt)})"
        return _str

    def __repr__(self) -> str:
        """*__repr__()* returns a string representation of the *SensitivityAnalysis* object.

        Returns:
            str: String representation of the *SensitivityAnalysis* object.
        """
        return self.__str__()
