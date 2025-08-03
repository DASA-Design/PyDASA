# -*- coding: utf-8 -*-
"""
Module for **SensitivityHandler** to manage sensitivity analysis in *PyDASA*

This module provides the SensitivityHandler class for coordinating multiple sensitivity analyses and generating reports on which variables have the most significant impact on dimensionless coefficients.

*IMPORTANT:* Based on the theory from:
    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Generic
# import re

# Import validation base classes
from new.pydasa.core.basic import Validation

# Import related classes
from new.pydasa.core.parameter import Variable
from new.pydasa.buckingham.vashchy import Coefficient
from new.pydasa.analysis.scenario import DimSensitivity

# Import utils
from new.pydasa.utils.default import T
from new.pydasa.utils.error import inspect_var
from new.pydasa.utils import config as cfg


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
        _fwk (str): Framework context (PHYSICAL, COMPUTATION, DIGITAL, CUSTOM).
        _cat (str): Category of analysis (SYM, NUM, HYB).

        # Analysis Components
        _variables (List[Variable]): List of variables used in analysis.
        _coefficients (List[Coefficient]): List of coefficients to analyze.
        _variable_map (Dict[str, Variable]): Map of variable symbols to objects.
        _coefficient_map (Dict[str, Coefficient]): Map of coefficient symbols to objects.

        # Analysis Results
        _analyses (List[DimSensitivity]): List of sensitivity analyses performed.
        _results (Dict[str, Dict[str, Any]]): Consolidated results of analyses.
    """

    # Category attribute
    # :attr: _cat
    _cat: str = "SYM"
    """Category of sensitivity analysis (SYM, NUM, HYB)."""

    # Analysis components
    # :attr: _variables
    _variables: List[Variable] = field(default_factory=list)
    """List of variables used in analysis."""

    # :attr: _coefficients
    _coefficients: List[Coefficient] = field(default_factory=list)
    """List of coefficients to analyze."""

    # :attr: _variable_map
    _variable_map: Dict[str, Variable] = field(default_factory=dict)
    """Map of variable symbols to objects."""

    # :attr: _coefficient_map
    _coefficient_map: Dict[str, Coefficient] = field(default_factory=dict)
    """Map of coefficient symbols to objects."""

    # Analysis results
    # :attr: _analyses
    _analyses: List[DimSensitivity] = field(default_factory=list)
    """List of sensitivity analyses performed."""

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
            self._sym = f"SENS HDL \\Pi_{{{self._idx}}}" if self._idx >= 0 else "SENS HDL"

        # Initialize component maps
        if self._variables:
            self._setup_variable_map()

        if self._coefficients:
            self._setup_coefficient_map()

    def _setup_variable_map(self) -> None:
        """*_setup_variable_map()* Creates a map of variable symbols to variable objects.
        """
        self._variable_map.clear()
        for var in self._variables:
            self._variable_map[var.sym] = var

    def _setup_coefficient_map(self) -> None:
        """*_setup_coefficient_map()* Creates a map of coefficient symbols to coefficient objects.
        """
        self._coefficient_map.clear()
        for coef in self._coefficients:
            self._coefficient_map[coef.sym] = coef

    def _validate_list(self, lt: List, exp_type: tuple) -> bool:
        """*_validate_list()* Validates a list with expected element types.

        Args:
            lt (List): List to validate.
            exp_type (tuple): Expected types for list elements.

        Raises:
            ValueError: If list is empty or contains invalid types.

        Returns:
            bool: True if the list is valid.
        """
        if not isinstance(lt, list):
            _msg = f"{inspect_var(lt)} must be a list. "
            _msg += f"Provided: {type(lt)}"
            raise ValueError(_msg)
        if not all(isinstance(x, exp_type) for x in lt):
            _msg = f"{inspect_var(lt)} must contain {exp_type} elements."
            _msg += f" Provided: {[type(x).__name__ for x in lt]}"
            raise ValueError(_msg)
        if len(lt) == 0:
            _msg = f"{inspect_var(lt)} cannot be empty. "
            _msg += f"Provided: {lt}"
            raise ValueError(_msg)
        return True

    def _create_analyses(self) -> None:
        """*_create_analyses()* Creates sensitivity analyses for each coefficient.

        Sets up DimSensitivity objects for each coefficient to be analyzed.
        """
        self._analyses.clear()

        for i, coef in enumerate(self._coefficients):
            # Create sensitivity analysis
            analysis = DimSensitivity(
                _idx=i,
                _sym=f"SEN_{{{coef.sym}}}",
                _fwk=self._fwk,
                _cat=self._cat,
                name=f"Sensitivity of {coef.name}",
                description=f"Sensitivity analysis for coefficient {coef.sym}"
            )

            # Configure with coefficient
            analysis.set_coefficient(coef)

            # Add to list
            self._analyses.append(analysis)

    def _get_variable_value(self,
                            var_sym: str,
                            val_type: str = "avg") -> float:
        """*_get_variable_value()* Gets a value for a variable based on value type.

        Args:
            var_sym (str): Symbol of the variable.
            val_type (str, optional): Type of value to return (avg, min, max). Defaults to "avg".

        Returns:
            float: Variable value.

        Raises:
            ValueError: If the variable or value type is invalid.
        """
        # Check if the variable symbol exists in our variable map
        if var_sym not in self._variable_map:
            _msg = f"Variable '{var_sym}' not found in variable map."
            _msg += f" Available variables: {list(self._variable_map.keys())}"
            raise ValueError(_msg)

        # Get the Variable object from the map
        var = self._variable_map[var_sym]

        # CASE 1: Return average value
        if val_type == "avg":
            # First check if standardized average exists
            if var.std_avg is None:
                # If no standardized average, try regular average
                # If that's also None, use default value -1.0
                return var.avg if var.avg is not None else -1.0
            # Return standardized average if it exists
            return var.std_avg
            
        # CASE 2: Return minimum value
        elif val_type == "min":
            # First check if standardized minimum exists
            if var.std_min is None:
                # If no standardized minimum, try regular minimum
                # If that's also None, use default value -0.1
                return var.min if var.min is not None else -0.1
            # Return standardized minimum if it exists
            return var.std_min
            
        # CASE 3: Return maximum value
        elif val_type == "max":
            # First check if standardized maximum exists
            if var.std_max is None:
                # If no standardized maximum, try regular maximum
                # If that's also None, use default value -10.0
                return var.max if var.max is not None else -10.0
            # Return standardized maximum if it exists
            return var.std_max
            
        # CASE 4: Invalid value type
        else:
            # Build error message
            _msg = f"Invalid value type: {val_type}. "
            _msg += "Must be one of: avg, min, max."
            raise ValueError(_msg)

    def analyze_symbolic(self,
                         val_type: str = "avg") -> Dict[str, Dict[str, float]]:
        """*analyze_symbolic()* Performs symbolic sensitivity analysis.

        Analyzes each coefficient using symbolic differentiation at specified values.

        Args:
            val_type (str, optional): Type of value to use (avg, min, max). Defaults to "avg".

        Returns:
            Dict[str, Dict[str, float]]: Sensitivity results by coefficient and variable.
        """
        # Create analyses if not already done
        if not self._analyses:
            self._create_analyses()

        # Clear previous results
        self._results.clear()

        # Process each analysis
        for analysis in self._analyses:
            # Get variable values
            values = {}
            for var_sym in analysis.variables:
                values[var_sym] = self._get_variable_value(var_sym, val_type)

            # Perform analysis
            result = analysis.analyze_symbolically(values)

            # Store results
            coef_sym = analysis.pi_expr
            self._results[coef_sym] = result

        return self._results

    def analyze_numeric(self,
                        num_samples: int = 1000) -> Dict[str, Dict[str, Any]]:
        """*analyze_numeric()* Performs numerical sensitivity analysis.

        Analyzes each coefficient using Fourier Amplitude Sensitivity Test (FAST).

        Args:
            num_samples (int, optional): Number of samples to use. Defaults to 1000.

        Returns:
            Dict[str, Dict[str, Any]]: Sensitivity results by coefficient.
        """
        # Create analyses if not already done
        if not self._analyses:
            self._create_analyses()

        # Clear previous results
        self._results.clear()

        # Process each analysis
        for analysis in self._analyses:
            # Get variable bounds
            bounds = []
            for var_sym in analysis.variables:
                var = self._variable_map[var_sym]
                min_val = var.std_min if var.std_min is not None else (var.min if var.min is not None else 0.1)
                max_val = var.std_max if var.std_max is not None else (var.max if var.max is not None else 10.0)
                bounds.append([min_val, max_val])

            # Perform analysis
            result = analysis.analyze_numerically(bounds, num_samples)

            # Store results
            coef_sym = analysis.pi_expr
            self._results[coef_sym] = {
                'S1': dict(zip(analysis.variables, result['S1'])),
                'ST': dict(zip(analysis.variables, result['ST'])),
                'names': analysis.variables,
                'raw': result
            }

        return self._results

    def get_ranked_variables(self,
                             metric: str = "S1") -> Dict[str, List[tuple]]:
        """*get_ranked_variables()* Gets variables ranked by sensitivity.

        Args:
            metric (str, optional): Metric to use for ranking (S1, ST). Defaults to "S1".

        Returns:
            Dict[str, List[tuple]]: Variables ranked by sensitivity for each coefficient.

        Raises:
            ValueError: If results are not available or metric is invalid.
        """
        if not self._results:
            raise ValueError("No analysis results available. Run analyze_symbolic() or analyze_numeric() first.")

        rankings = {}

        for coef_sym, result in self._results.items():
            if metric not in result and 'raw' not in result:
                # Try symbolic results
                var_ranks = [(var, abs(sens)) for var, sens in result.items()]
                var_ranks.sort(key=lambda x: x[1], reverse=True)
                rankings[coef_sym] = var_ranks
            elif 'raw' in result:
                # Numeric results
                if metric not in ['S1', 'ST']:
                    raise ValueError(f"Invalid metric: {metric}. Must be one of: S1, ST.")

                var_ranks = [(var, result[metric][var]) for var in result['names']]
                var_ranks.sort(key=lambda x: x[1], reverse=True)
                rankings[coef_sym] = var_ranks
        return rankings

    def get_most_influential_variables(self,
                                       top_n: int = 3) -> Dict[str, List[str]]:
        """*get_most_influential_variables()* Gets most influential variables for each coefficient.
        Args:
            top_n (int, optional): Number of top variables to return. Defaults to 3.

        Returns:
            Dict[str, List[str]]: Top influential variables for each coefficient.
        """
        rankings = self.get_ranked_variables()

        top_vars = {}
        for coef_sym, ranked_vars in rankings.items():
            top_vars[coef_sym] = [var for var, _ in ranked_vars[:min(top_n, len(ranked_vars))]]

        return top_vars

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
    def variables(self) -> List[Variable]:
        """*variables* Get the list of variables.

        Returns:
            List[Variable]: List of variables.
        """
        return self._variables.copy()

    @variables.setter
    def variables(self, val: List[Variable]) -> None:
        """*variables* Set the list of variables.

        Args:
            val (List[Variable]): List of variables.

        Raises:
            ValueError: If list is invalid.
        """
        if self._validate_list(val, (Variable,)):
            self._variables = val
            self._setup_variable_map()

    @property
    def coefficients(self) -> List[Coefficient]:
        """*coefficients* Get the list of coefficients.

        Returns:
            List[Coefficient]: List of coefficients.
        """
        return self._coefficients.copy()

    @coefficients.setter
    def coefficients(self, val: List[Coefficient]) -> None:
        """*coefficients* Set the list of coefficients.

        Args:
            val (List[Coefficient]): List of coefficients.

        Raises:
            ValueError: If list is invalid.
        """
        if self._validate_list(val, (Coefficient,)):
            self._coefficients = val
            self._setup_coefficient_map()
            # Clear existing analyses
            self._analyses.clear()

    @property
    def analyses(self) -> List[DimSensitivity]:
        """*analyses* Get the list of sensitivity analyses.

        Returns:
            List[DimSensitivity]: List of sensitivity analyses.
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
        self._sym = "SENS HDL"
        self._fwk = "PHYSICAL"
        self.name = ""
        self.description = ""

        # Reset handler-specific attributes
        self._cat = "SYM"
        self._variables = []
        self._coefficients = []
        self._variable_map = {}
        self._coefficient_map = {}
        self._analyses = []
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
