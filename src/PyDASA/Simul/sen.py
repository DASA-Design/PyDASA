# -*- coding: utf-8 -*-
"""
Module for sensitivity analysis using Fourier Amplitude Sensitivity Testing (FAST).

This module provides the `SensitivityAnalysis` class, which performs sensitivity analysis on a list of *PiNumbers* and *Variables*.

It computes the sensitivity of each *PiNumber* based on the provided *Variable* samples and ranks them accordingly.
"""

# native python modules
from typing import Optional, List, Generic
from dataclasses import dataclass, field
import inspect
import re

# Third-party modules
import numpy as np

# Custom modules
# Dimensional Analysis modules
from Src.PyDASA.Units.fdu import FDU
from Src.PyDASA.Units.params import Variable
from Src.PyDASA.Pi.coef import PiCoefficient, PiNumber

# Data Structures
from Src.PyDASA.DStructs.Tables.scht import SCHashTable

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

# TODO maybe i need a Sensitivity class to store the results of the analysis


@dataclass
class SensitivityAnalysis(Generic[T]):

    # Private attributes with validation logic
    # :attr: _idx
    _idx: int = -1
    """
    Unique identifier/index of the *SensitivityAnalysis*.
    """

    # :attr: _sym
    _sym: str = "DA_{x}"
    """
    Symbol of the *SensitivityAnalysis*. It must be alphanumeric (preferably a single character + Latin or Greek letter). Useful for user-friendly representation of the instance.
    """

    # :attr: _fwk
    _fwk: str = "PHYSICAL"
    """
    Framework of the *SensitivityAnalysis*, can be one of the following: `PHYSICAL`, `COMPUTATION`, `DIGITAL` or `CUSTOM`. By default, it is set to `PHYSICAL`.
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
    sensitivity_rpt: Optional[List[dict]] = None
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


    # OLD METHODS + TODOs + CODE IMPROVEMENTS + TODOs
    def perform_analysis(self, samples: int = 1000) -> None:
        """
        Performs sensitivity analysis using FAST and ranks the PiNumbers.

        Args:
            samples (int): Number of samples for the analysis. Default is 1000.
        """
        # FIXME old code, deprecate!!!
        # Generate random samples for each variable
        variable_samples = {
            var.name: np.random.uniform(var.min, var.max, samples)
            for var in self.variables
        }

        # Analyze each PiNumber
        for pi in self.pi_numbers:
            pi_values = self._compute_pi_values(pi, variable_samples)
            pi.sensitivity = self._compute_sensitivity(pi_values)
            pi.max_value = np.max(pi_values)
            pi.min_value = np.min(pi_values)
            pi.avg_value = np.mean(pi_values)

        # Sort PiNumbers by sensitivity in descending order
        self.pi_numbers.sort(key=lambda x: x.sensitivity, reverse=True)

    def _compute_pi_values(self, pi: PiNumber, variable_samples: dict) -> np.ndarray:
        """
        Computes the values of a PiNumber based on variable samples.

        Args:
            pi (PiNumber): The PiNumber object.
            variable_samples (dict): Dictionary of variable samples.

        Returns:
            np.ndarray: Array of computed Pi values.
        """
        # FIXME old code, deprecate!!!
        pi_values = np.ones(len(next(iter(variable_samples.values()))))
        for var, exp in zip(pi.param_lt, pi.dim_col):
            if var in variable_samples:
                pi_values *= variable_samples[var] ** exp
        return pi_values

    def _compute_sensitivity(self, pi_values: np.ndarray) -> float:
        """
        Computes the sensitivity of a PiNumber using FAST.

        Args:
            pi_values (np.ndarray): Array of computed Pi values.

        Returns:
            float: Sensitivity score.
        """
        # FIXME old code, deprecate!!!
        # Perform Fourier Transform
        fft_values = np.fft.fft(pi_values)
        amplitudes = np.abs(fft_values)

        # Compute sensitivity as the ratio of the first harmonic to the total amplitude
        sensitivity = amplitudes[1] / np.sum(amplitudes)
        return sensitivity

    def get_results(self) -> List[dict]:
        # FIXME old code, deprecate!!!
        """
        Returns the ranked PiNumbers with their sensitivity, max, min, and avg values.

        Returns:
            List[dict]: List of dictionaries containing PiNumber analysis results.
        """
        # TODO tthis are the attrs for the report class
        return [
            {
                "pi_number": pi.sym,
                "sensitivity": pi.sensitivity,
                "max": pi.max_value,
                "min": pi.min_value,
                "avg": pi.avg_value,
            }
            for pi in self.pi_numbers
        ]
