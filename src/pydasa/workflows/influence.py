# -*- coding: utf-8 -*-
"""
Module influence.py
===========================================

Module for **SensitivityAnalysis** to manage sensitivity analysis in *PyDASA*

This module provides the SensitivityAnalysis class for coordinating multiple sensitivity analyses and generating reports on which variables have the most significant impact on dimensionless coefficients.

Classes:
    **SensitivityAnalysis**: Manages sensitivity analyses for multiple coefficients, processes results, and generates reports on variable impacts.

*IMPORTANT:* Based on the theory from:

    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""

from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import List, Dict, Any, Union, Tuple
# import re

# Import validation base classes
from pydasa.core.basic import Foundation

# Import related classes
from pydasa.elements.parameter import Variable
from pydasa.dimensional.buckingham import Coefficient
from pydasa.analysis.scenario import Sensitivity

# Import utils
from pydasa.validations.error import inspect_var
from pydasa.serialization.parser import latex_to_python

# Import validation decorators
from pydasa.validations.decorators import validate_type
from pydasa.validations.decorators import validate_choices
from pydasa.validations.decorators import validate_emptiness
# Import global configuration
from pydasa.core.setup import AnaliticMode
from pydasa.core.setup import PYDASA_CFG
# from pydasa.validations.patterns import LATEX_RE


@dataclass
class SensitivityAnalysis(Foundation):
    """**SensitivityAnalysis** class for managing multiple sensitivity analyses in *PyDASA*.

    Coordinates sensitivity analyses for multiple coefficients, processes their results, and generates comprehensive reports on variable impacts.

    Attributes:
        # Identification and Classification
        name (str): User-friendly name of the sensitivity handler.
        description (str): Brief summary of the sensitivity handler.
        _idx (int): Index/precedence of the sensitivity handler.
        _sym (str): Symbol representation (LaTeX or alphanumeric).
        _alias (str): Python-compatible alias for use in code.
        _fwk (str): Frameworks context (PHYSICAL, COMPUTATION, SOFTWARE, CUSTOM).
        _cat (str): Category of analysis (SYM, NUM, HYB).

        # Analysis Components
        _variables (Dict[str, Variable]): all available parameters/variables in the model (*Variable*).
        _coefficients (Dict[str, Coefficient]): all available coefficients in the model (*Coefficient*).

        # Analysis Results
        _analyses (Dict[str, Sensitivity]): all sensitivity analyses performed.
        _results (Dict[str, Dict[str, Any]]): all consolidated results of analyses.
    """

    # Category attribute
    # :attr: _cat
    _cat: str = AnaliticMode.SYM.value
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
    _analyses: Dict[str, Sensitivity] = field(default_factory=dict)
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
            self._sym = f"SANSYS_{{\\Pi_{{{self._idx}}}}}" if self._idx >= 0 else "SANSYS_\\Pi_{-1}"

        if not self._alias:
            self._alias = latex_to_python(self._sym)

        # Set name and description if not already set
        if not self.name:
            self.name = f"sensitivity Analysis Handler {self._idx}"

        if not self.description:
            self.description = f"Manages sensitivity analyses for [{self._coefficients.keys()}] coefficients."

    def _validate_dict(self, dt: dict,
                       exp_type: Union[type, List[type], Tuple[type, ...]]) -> bool:
        """*_validate_dict()* Validates a dictionary with expected value types.

        Args:
            dt (dict): Dictionary to validate.
            exp_type (Union[type, List[type], Tuple[type, ...]]): Expected type(s) for dictionary values.

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

        # Convert exp_type to tuple for isinstance()
        if isinstance(exp_type, (list, tuple)):
            type_tuple = tuple(exp_type)
        else:
            type_tuple = (exp_type,)

        if not all(isinstance(v, type_tuple) for v in dt.values()):
            _msg = f"{inspect_var(dt)} must contain {exp_type} values."
            _msg += f" Provided: {[type(v).__name__ for v in dt.values()]}"
            raise ValueError(_msg)

        return True

    def _create_analyses(self) -> None:
        """*_create_analyses()* Creates sensitivity analyses for each coefficient.

        Sets up Sensitivity objects for each coefficient to be analyzed.
        """
        self._analyses.clear()

        for i, (pi, coef) in enumerate(self._coefficients.items()):
            # Create sensitivity analysis
            analysis = Sensitivity(
                _idx=i,
                _sym=f"SEN_{{{coef.sym}}}",
                _fwk=self._fwk,
                _cat=self._cat,
                _name=f"Sensitivity for {coef.name}",
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
    @validate_choices(PYDASA_CFG.analitic_modes, case_sensitive=False)
    def cat(self, val: str) -> None:
        """*cat* Set the analysis category.

        Args:
            val (str): Category value.

        Raises:
            ValueError: If category is invalid.
        """
        self._cat = val.upper()

    @property
    def variables(self) -> Dict[str, Variable]:
        """*variables* Get the dictionary of variables.

        Returns:
            Dict[str, Variable]: Dictionary of variables.
        """
        return self._variables.copy()

    @variables.setter
    @validate_type(dict, allow_none=False)
    @validate_emptiness()
    def variables(self, val: Dict[str, Variable]) -> None:
        """*variables* Set the dictionary of variables.

        Args:
            val (Dict[str, Variable]): Dictionary of variables.

        Raises:
            ValueError: If dictionary is invalid.
        """
        # Validate dictionary values are Variable instances
        if not all(isinstance(v, Variable) for v in val.values()):
            _msg = "All dictionary values must be Variable instances"
            raise ValueError(_msg)

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
    @validate_type(dict, allow_none=False)
    @validate_emptiness()
    def coefficients(self, val: Dict[str, Coefficient]) -> None:
        """*coefficients* Set the dictionary of coefficients.

        Args:
            val (Dict[str, Coefficient]): Dictionary of coefficients.

        Raises:
            ValueError: If dictionary is invalid.
        """
        # Validate dictionary values are Coefficient instances
        if not all(isinstance(v, Coefficient) for v in val.values()):
            _msg = "All dictionary values must be Coefficient instances"
            raise ValueError(_msg)

        self._coefficients = val
        # Clear existing analyses
        self._analyses.clear()

    @property
    def analyses(self) -> Dict[str, Sensitivity]:
        """*analyses* Get the dictionary of sensitivity analyses.

        Returns:
            Dict[str, Sensitivity]: Dictionary of sensitivity analyses.
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
        # Reset parent class attributes (Foundation)
        super().clear()
        self._sym = f"SANSYS_{{\\Pi_{{{self._idx}}}}}"

        # Reset handler-specific attributes
        self._cat = AnaliticMode.SYM.value
        self._variables = {}
        self._coefficients = {}
        self._analyses = {}
        self._results = {}

    def to_dict(self) -> Dict[str, Any]:
        """*to_dict()* Convert sensitivity handler to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation of sensitivity handler.
        """
        result = {}

        # Get all dataclass fields
        for f in fields(self):
            attr_name = f.name
            attr_value = getattr(self, attr_name)

            # Handle special dictionary types (Variable and Coefficient dicts)
            if attr_name == "_variables" and isinstance(attr_value, dict):
                attr_value = [var.to_dict() for var in attr_value.values()]
            elif attr_name == "_coefficients" and isinstance(attr_value, dict):
                attr_value = [coef.to_dict() for coef in attr_value.values()]
            elif attr_name == "_analyses" and isinstance(attr_value, dict):
                # Skip analyses - they are transient and recreated as needed
                continue

            # Skip callables (can't be serialized)
            if callable(attr_value) and not isinstance(attr_value, type):
                continue

            # Remove leading underscore from private attributes
            if attr_name.startswith("_"):
                clean_name = attr_name[1:]  # Remove first character
            else:
                clean_name = attr_name

            result[clean_name] = attr_value

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SensitivityAnalysis:
        """*from_dict()* Create sensitivity handler from dictionary representation.

        Args:
            data (Dict[str, Any]): Dictionary representation of sensitivity handler.

        Returns:
            SensitivityAnalysis: New sensitivity handler instance.
        """
        # Get all valid field names from the dataclass
        field_names = {f.name for f in fields(cls)}

        # Map keys without underscores to keys with underscores
        mapped_data = {}

        for key, value in data.items():
            # Handle special list conversions back to dictionaries
            if key == "variables" and isinstance(value, list):
                mapped_data["_variables"] = {
                    var["name"] if "name" in var else var.get("_name", f"var_{i}"): Variable.from_dict(var)
                    for i, var in enumerate(value)
                }
                continue
            elif key == "coefficients" and isinstance(value, list):
                mapped_data["_coefficients"] = {
                    coef["name"] if "name" in coef else coef.get("_name", f"coef_{i}"): Coefficient.from_dict(coef)
                    for i, coef in enumerate(value)
                }
                continue

            # Try the key as-is first (handles both _idx and name)
            if key in field_names:
                mapped_data[key] = value
            # Try adding underscore prefix (handles idx -> _idx)
            elif f"_{key}" in field_names:
                mapped_data[f"_{key}"] = value
            # Try removing underscore prefix (handles _name -> name if needed)
            elif key.startswith("_") and key[1:] in field_names:
                mapped_data[key[1:]] = value
            else:
                # Use as-is for unknown keys (will be validated by dataclass)
                mapped_data[key] = value

        return cls(**mapped_data)
