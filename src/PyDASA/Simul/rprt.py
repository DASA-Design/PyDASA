# -*- coding: utf-8 -*-
"""
Module for sensitivity analysis in *PyDASA*.

Uses SymPy for analytical sensitivity analysis (derivatives) and SALib for numerical sensitivity analysis (FAST).

The *SensitivityAnalysis* class computes sensitivities for *PiNumbers* based on *Variables* and ranks them in *SensitivitReport*.
"""
# native python modules
from typing import Optional, List, Generic, Union
from dataclasses import dataclass, field
import inspect
import re

# # Third-party modules
# import numpy as np
# import sympy as sp
# from sympy.parsing.latex import parse_latex
# from sympy import symbols, diff, lambdify
# import SALib
# from SALib.sample.fast_sampler import sample
# from SALib.analyze.fast import analyze

# Custom modules
# Dimensional Analysis modules
# from Src.PyDASA.Measure.fdu import FDU
from Src.PyDASA.Measure.params import Parameter, Variable
from Src.PyDASA.Pi.coef import PiCoefficient, PiNumber
from Src.PyDASA.Simul.sens import Sensitivity

# Data Structures
from Src.PyDASA.DStruct.Tables.scht import SCHashTable

# Utils modules
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
class SensitivityAnalysis(Generic[T]):

    # Private attributes with validation logic
    # :attr: _idx
    _idx: int = -1
    """
    Unique identifier/index of the *SensitivityAnalysis*.
    """

    # :attr: _sym
    _sym: str = "SEN ANSYS RPRT_{x}"
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
    Category of the *SensitivityAnalysis*. It can be one of the following: `SYMBOLIC`, `NUMERIC` or `CUSTOM` and decides which method to use for the analysis. By default, it is set to `SYMBOLIC`.
    """

    # :attr: _relevance_lt
    _relevance_lt: List[Union[Parameter, Variable]] = field(default_factory=list)
    """
    List of relevant *Variable* objects. Need to be defined before performing the sensitivity analysis with the *DimensionalAnalyzer*.
    """

    # :attr: _coefficient_lt
    _coefficient_lt: List[Union[PiCoefficient, PiNumber]] = field(default_factory=list)
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

    # :attr: report
    report: List[Sensitivity] = field(default_factory=list)
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

    # TODO: add private methods.
    def _setup_variables_map(self, var_lt: List[Variable]) -> None:
        """*setup_vars_map* sets up the hash table for the *Variable* objects.

        Args:
            var_lt (List[Variable]): List of *Variable* objects.
        """
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
        """*cat* Sets the category of the *SensitivityAnalysis*. It can be one of the following: `SYMBOLIC`, `NUMERIC` or `CUSTOM`.

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
        if self._validate_list(val, (PiCoefficient, PiNumber)):
            self._coefficient_lt = val

    def _setup_report(self) -> None:
        """*setup_report* sets up the report for the *SensitivityAnalysis*."""
        # prepare the sensitivity analysis
        for coef in self.coefficient_lt:
            entry = self._coefficient_mp.get_entry(str(coef.sym)).value
            if entry is None:
                _msg = f"Pi Coefficient {coef.sym} not found in the relevance list."
                raise ValueError(_msg)
            entry = list(entry.par_dims.keys())
            sens = Sensitivity(_idx=coef.idx,
                               _sym=coef.sym,
                               _fwk=self.fwk,
                               _pi_expr=coef.pi_expr,
                               _variables=entry,)
            self.report.append(sens)

    def analyze_pi_sensitivity(self,
                               category: str = "SYMBOLIC",
                               cutoff: str = "avg") -> None:
        self._setup_report()
        for sen in self.report:
            if category == "SYMBOLIC":
                td = {}
                for param in sen.variables:
                    details = self._relevance_mp.get_entry(param).value
                    if cutoff == "avg":
                        td[param] = details.std_avg
                    elif cutoff == "max":
                        td[param] = details.std_max
                    elif cutoff == "min":
                        td[param] = details.std_min
                    else:
                        _msg = f"Invalid cutoff method: {cutoff}. "
                        _msg += "Must be one of the following: "
                        _msg += f"{', '.join(cfg.CUTOFF_METHODS.keys())}."
                        raise ValueError(_msg)
                sen.analyze_symbolically(td)
            elif category == "NUMERIC":
                rg_lt = []
                for param in sen.variables:
                    details = self._relevance_mp.get_entry(param).value
                    trg = (details.std_min, details.std_max)
                    rg_lt.append(trg)
                # Perform numerical analysis using the FAST method
                sen.analyze_numerically(rg_lt)
            else:
                _msg = f"Invalid category: {category}. "
                _msg += "Must be one of the following: "
                _msg += f"{', '.join(cfg.SENS_ANSYS_DT.keys())}."
                raise ValueError(_msg)

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
