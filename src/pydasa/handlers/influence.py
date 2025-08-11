# -*- coding: utf-8 -*-
"""
Module influence.py
===========================================

Module for **SensitivityHandler** to manage sensitivity analysis in *PyDASA*

This module provides the SensitivityHandler class for coordinating multiple sensitivity analyses and generating reports on which variables have the most significant impact on dimensionless coefficients.

Classes:
    **SensitivityHandler**: Manages sensitivity analyses for multiple coefficients, processes results, and generates reports on variable impacts.

*IMPORTANT:* Based on the theory from:

    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Generic
# import re

# Import validation base classes
from src.pydasa.core.basic import Validation

# Import related classes
from src.pydasa.core.parameter import Variable
from src.pydasa.buckingham.vashchy import Coefficient
from src.pydasa.analysis.scenario import DimSensitivity

# Import utils
from src.pydasa.utils.default import T
from src.pydasa.utils.error import inspect_var
from src.pydasa.utils.latex import latex_to_python

# Import global configuration
# Import the 'cfg' module to allow global variable editing
from src.pydasa.utils import config as cfg


@dataclass
class SensitivityHandler(Validation, Generic[T]):
    """**SensitivityHandler** class for managing multiple sensitivity analyses in *PyDASA*.

    Coordinates sensitivity analyses for multiple coefficients, processes their results, and generates comprehensive reports on variable impacts.

    Attributes:
        # Identification and Classification
        name (str): User-friendly name of the sensitivity handler.
        description (str): Brief summary of the sensitivity handler.
        _idx (int): Index/precedence of the sensitivity handler.
        _sym (str): Symbol representation (LaTeX or alphanumeric).
        _alias (str): Python-compatible alias for use in code.
        _fwk (str): Framework context (PHYSICAL, COMPUTATION, SOFTWARE, CUSTOM).
        _cat (str): Category of analysis (SYM, NUM, HYB).

        # Analysis Components
        _variables (Dict[str, Variable]): all available parameters/variables in the model (*Variable*).
        _coefficients (Dict[str, Coefficient]): all available coefficients in the model (*Coefficient*).

        # Analysis Results
        _analyses (Dict[str, DimSensitivity]): all sensitivity analyses performed.
        _results (Dict[str, Dict[str, Any]]): all consolidated results of analyses.
    """

    # Category attribute
    # :attr: _cat
    _cat: str = "SYM"
    """Category of sensitivity analysis (SYM, NUM)."""

    # Variable management
    # :attr: _variables
    _variables: Dict[str, Variable] = field(default_factory=dict)
    """Dictionary of all parameters/variables in the model (*Variable*)."""

    # :attr: _coefficients
    _coefficients: Dict[str, Coefficient] = field(default_factory=dict)
    """Dictionary of all coefficients in the model (*Coefficient*)."""

    # Analysis results
    # :attr: _analyses
    _analyses: Dict[str, DimSensitivity] = field(default_factory=dict)
    """Dictionary of sensitivity analyses performed."""

    # :attr: _results
    _results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Consolidated results of analyses."""

    def __post_init__(self) -> None:
        """*__post_init__()* Initializes the sensitivity handler.

        Validates basic properties and sets up component maps.
        """
        # Initialize from base class
        super().__post_init__()

        # Set default symbol if not specified
        if not self._sym:
            self._sym = f"SH_{{\\Pi_{{{self._idx}}}}}" if self._idx >= 0 else "SH_\\Pi_{-1}"

        if not self._alias:
            self._alias = latex_to_python(self._sym)

        # Set name and description if not already set
        if not self.name:
            self.name = f"sensitivity Analysis Handler {self._idx}"

        if not self.description:
            self.description = f"Manages sensitivity analyses for [{self._coefficients.keys()}] coefficients."

    def _validate_dict(self, dt: dict, exp_type: List[type]) -> bool:
        """*_validate_dict()* Validates a dictionary with expected value types.

        Args:
            dt (dict): Dictionary to validate.
            exp_type (List[type]): Expected types for dictionary values.

        Raises:
            ValueError: If the object is not a dictionary.
            ValueError: If the dictionary is empty.
            ValueError: If the dictionary contains values of unexpected types.

        Returns:
            bool: True if the dictionary is valid.
        """
        if not isinstance(dt, dict):
            _msg = f"{inspect_var(dt)} must be a dictionary. "
            _msg += f"Provided: {type(dt)}"
            raise ValueError(_msg)
        if len(dt) == 0:
            _msg = f"{inspect_var(dt)} cannot be empty. "
            _msg += f"Provided: {dt}"
            raise ValueError(_msg)
        if not all(isinstance(v, exp_type) for v in dt.values()):
            _msg = f"{inspect_var(dt)} must contain {exp_type} values."
            _msg += f" Provided: {[type(v).__name__ for v in dt.values()]}"
            raise ValueError(_msg)
        return True

    def _create_analyses(self) -> None:
        """*_create_analyses()* Creates sensitivity analyses for each coefficient.

        Sets up DimSensitivity objects for each coefficient to be analyzed.
        """
        self._analyses.clear()

        for i, (pi, coef) in enumerate(self._coefficients.items()):
            # Create sensitivity analysis
            analysis = DimSensitivity(
                _idx=i,
                _sym=f"SEN_{{{coef.sym}}}",
                _fwk=self._fwk,
                _cat=self._cat,
                name=f"Sensitivity for {coef.name}",
                description=f"Sensitivity analysis for {coef.sym}",
                # _pi_expr=coef._pi_expr
            )

            # Configure with coefficient
            analysis.set_coefficient(coef)

            # Add to list
            self._analyses[pi] = analysis

    def _get_variable_value(self,
                            var_sym: str,
                            val_type: str = "mean") -> float:
        """*_get_variable_value()* Gets a value for a variable based on value type.

        Args:
            var_sym (str): Symbol of the variable.
            val_type (str, optional): Type of value to return (mean, min, max). Defaults to "mean".

        Returns:
            float: Variable value.

        Raises:
            ValueError: If the variable is not found.
            ValueError: If the value type is invalid.
        """
        # Check if the variable symbol exists in our variable map
        if var_sym not in self._variables:
            _msg = f"Variable '{var_sym}' not found in variables."
            _msg += f" Available variables: {list(self._variables.keys())}"
            raise ValueError(_msg)

        # Get the Variable object from the map
        var = self._variables[var_sym]

        # CASE 1: Return average value
        if val_type == "mean":
            # First check if standardized average exists
            if var.std_mean is None:
                # If no standardized average, try regular average
                # If thats also None, use default value -1.0
                return var.mean if var.mean is not None else -1.0
            # Return standardized average if it exists
            return var.std_mean

        # CASE 2: Return minimum value
        elif val_type == "min":
            # First check if standardized minimum exists
            if var.std_min is None:
                # If no standardized minimum, try regular minimum
                # If thats also None, use default value -0.1
                return var.min if var.min is not None else -0.1
            # Return standardized minimum if it exists
            return var.std_min

        # CASE 3: Return maximum value
        elif val_type == "max":
            # First check if standardized maximum exists
            if var.std_max is None:
                # If no standardized maximum, try regular maximum
                # If thats also None, use default value -10.0
                return var.max if var.max is not None else -10.0
            # Return standardized maximum if it exists
            return var.std_max

        # CASE 4: Invalid value type
        else:
            # Build error message
            _msg = f"Invalid value type: {val_type}. "
            _msg += "Must be one of: mean, min, max."
            raise ValueError(_msg)

    def analyze_symbolic(self,
                         val_type: str = "mean") -> Dict[str, Dict[str, float]]:
        """*analyze_symbolic()* Performs symbolic sensitivity analysis.

        Analyzes each coefficient using symbolic differentiation at specified values.

        Args:
            val_type (str, optional): Type of value to use (mean, min, max). Defaults to "mean".

        Returns:
            Dict[str, Dict[str, float]]: Sensitivity results by coefficient and variable.
        """
        # Create analyses if not already done
        if not self._analyses:
            self._create_analyses()

        # Clear previous results
        self._results.clear()

        # Process each analysis
        for analysis in self._analyses.values():
            # Get variable values
            values = {}
            for var_sym in analysis._latex_to_py.keys():
                # Ensure symbol is a string
                values[var_sym] = self._get_variable_value(var_sym, val_type)
            # Perform analysis
            result = analysis.analyze_symbolically(values)

            # Store results
            self._results[analysis.sym] = result

        return self._results

    def analyze_numeric(self,
                        n_samples: int = 1000) -> Dict[str, Dict[str, Any]]:
        """*analyze_numeric()* Performs numerical sensitivity analysis.

        Analyzes each coefficient using Fourier Amplitude Sensitivity Test (FAST).

        Args:
            n_samples (int, optional): Number of samples to use. Defaults to 1000.

        Returns:
            Dict[str, Dict[str, Any]]: Sensitivity results by coefficient.
        """
        # Create analyses if not already done
        if not self._analyses:
            self._create_analyses()

        # Clear previous results
        self._results.clear()

        # Process each analysis
        for analysis in self._analyses.values():
            # Get variable bounds
            vals = []
            bounds = []
            for var_sym in analysis._latex_to_py.keys():
                var = self._variables[var_sym]
                min_val = var.std_min if var.std_min is not None else (var.min if var.min is not None else -0.1)
                max_val = var.std_max if var.std_max is not None else (var.max if var.max is not None else -10.0)
                bounds.append([min_val, max_val])
                vals.append(var.sym)

            # Perform analysis
            result = analysis.analyze_numerically(vals, bounds, n_samples)

            # Store results
            self._results[analysis.sym] = result
        return self._results

    # Property getters and setters

    @property
    def cat(self) -> str:
        """*cat* Get the analysis category.

        Returns:
            str: Category (SYM, NUM, HYB).
        """
        return self._cat

    @cat.setter
    def cat(self, val: str) -> None:
        """*cat* Set the analysis category.

        Args:
            val (str): Category value.

        Raises:
            ValueError: If category is invalid.
        """
        if val.upper() not in cfg.SENS_ANSYS_DT:
            raise ValueError(
                f"Invalid category: {val}. "
                f"Must be one of: {', '.join(cfg.SENS_ANSYS_DT.keys())}"
            )
        self._cat = val.upper()

    @property
    def variables(self) -> Dict[str, Variable]:
        """*variables* Get the dictionary of variables.

        Returns:
            Dict[str, Variable]: Dictionary of variables.
        """
        return self._variables.copy()

    @variables.setter
    def variables(self, val: Dict[str, Variable]) -> None:
        """*variables* Set the dictionary of variables.

        Args:
            val (Dict[str, Variable]): Dictionary of variables.

        Raises:
            ValueError: If dictionary is invalid.
        """
        if self._validate_dict(val, (Variable,)):
            self._variables = val

            # Clear existing analyses
            self._analyses.clear()

    @property
    def coefficients(self) -> Dict[str, Coefficient]:
        """*coefficients* Get the dictionary of coefficients.

        Returns:
            Dict[str, Coefficient]: Dictionary of coefficients.
        """
        return self._coefficients.copy()

    @coefficients.setter
    def coefficients(self, val: Dict[str, Coefficient]) -> None:
        """*coefficients* Set the dictionary of coefficients.

        Args:
            val (Dict[str, Coefficient]): Dictionary of coefficients.

        Raises:
            ValueError: If dictionary is invalid.
        """
        if self._validate_dict(val, (Coefficient,)):
            self._coefficients = val

            # Clear existing analyses
            self._analyses.clear()

    @property
    def analyses(self) -> Dict[str, DimSensitivity]:
        """*analyses* Get the dictionary of sensitivity analyses.

        Returns:
            Dict[str, DimSensitivity]: Dictionary of sensitivity analyses.
        """
        return self._analyses.copy()

    @property
    def results(self) -> Dict[str, Dict[str, Any]]:
        """*results* Get the analysis results.

        Returns:
            Dict[str, Dict[str, Any]]: Analysis results.
        """
        return self._results.copy()

    def clear(self) -> None:
        """*clear()* Reset all attributes to default values.

        Resets all handler properties to their initial state.
        """
        # Reset base class attributes
        self._idx = -1
        self._sym = "SENS_Pi_{-1}"
        self._fwk = "PHYSICAL"
        self.name = ""
        self.description = ""

        # Reset handler-specific attributes
        self._cat = "SYM"
        self._variables = {}
        self._coefficients = {}
        self._analyses = {}
        self._results = {}

    def to_dict(self) -> Dict[str, Any]:
        """*to_dict()* Convert sensitivity handler to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation of sensitivity handler.
        """
        return {
            "name": self.name,
            "description": self.description,
            "idx": self._idx,
            "sym": self._sym,
            "fwk": self._fwk,
            "cat": self._cat,
            "variables": [var.to_dict() for var in self._variables],
            "coefficients": [coef.to_dict() for coef in self._coefficients],
            "results": self._results
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SensitivityHandler:
        """*from_dict()* Create sensitivity handler from dictionary representation.

        Args:
            data (Dict[str, Any]): Dictionary representation of sensitivity handler.

        Returns:
            SensitivityHandler: New sensitivity handler instance.
        """
        # Create variables and coefficients from dicts
        variables = []
        if "variables" in data:
            variables = [Variable.from_dict(var) for var in data["variables"]]

        coefficients = []
        if "coefficients" in data:
            coefficients = [Coefficient.from_dict(coef) for coef in data["coefficients"]]

        # Remove list items from data
        handler_data = data.copy()
        for key in ["variables", "coefficients", "results"]:
            if key in handler_data:
                del handler_data[key]

        # Create handler
        handler = cls(
            **handler_data,
            _variables=variables,
            _coefficients=coefficients
        )

        # Set results if available
        if "results" in data:
            handler._results = data["results"]

        return handler
