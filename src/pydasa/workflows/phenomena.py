# -*- coding: utf-8 -*-
"""
Module phenomena.py
===========================================

Module for **AnalysisEngine** to orchestrate dimensional analysis workflows in *PyDASA*.

This module provides the **AnalysisEngine** class serves as the main entry point and workflow for *PyDASA's* dimensional analysis capabilities setting up the dimensional domain, solving the dimensional matrix, and coefficient generation.

Classes:
    **AnalysisEngine**: Main workflow class for dimensional analysis and coefficient generation.

*IMPORTANT:* Based on the theory from:

    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""
# Standard library imports
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Union, Tuple

# Import validation base classes
from pydasa.core.basic import Foundation

# Import related classes
from pydasa.elements.parameter import Variable
from pydasa.dimensional.buckingham import Coefficient
from pydasa.dimensional.model import Matrix
from pydasa.dimensional.framework import Schema

# Import utils
from pydasa.validations.error import inspect_var
from pydasa.serialization.parser import latex_to_python

# Import validation decorators
from pydasa.validations.decorators import validate_type
from pydasa.validations.decorators import validate_custom

# Import global configuration
from pydasa.core.setup import Framework, PYDASA_CFG

# custom type hinting
Variables = Union[Dict[str, Variable], Dict[str, Any]]
Coefficients = Union[Dict[str, Coefficient], Dict[str, Any]]
FDUs = Union[str, Dict[str, Any], Schema]


@dataclass
class AnalysisEngine(Foundation):
    """**AnalysisEngine** class for orchestrating dimensional analysis workflows in *PyDASA*.

    Main entry point that coordinates dimensional matrix solving and coefficient generation.
    Also known as DimProblem.

    Attributes:
        # Identification and Classification
        name (str): User-friendly name of the problem.
        description (str): Brief summary of the problem.
        _idx (int): Index/precedence of the problem.
        _sym (str): Symbol representation (LaTeX or alphanumeric).
        _alias (str): Python-compatible alias for use in code.
        _fwk (str): Framework context (PHYSICAL, COMPUTATION, SOFTWARE, CUSTOM).

        # Problem Components
        _variables (Dict[str, Variable]): All dimensional variables in the problem.
        _schema (Optional[Schema]): Dimensional framework schema for the problem.
        _model (Optional[Matrix]): Dimensional matrix for analysis.

        # Generated Results
        _coefficients (Dict[str, Coefficient]): Generated dimensionless coefficients.

        # Workflow State
        _is_solved (bool): Whether the dimensional matrix has been solved.
    """

    # Problem components
    # :attr: _variables
    _variables: Optional[Variables] = field(default_factory=dict)
    """Dictionary of all dimensional variables in the problem (can be Variable objects or dicts)."""

    # :attr: _schema
    _schema: Optional[FDUs] = Framework.PHYSICAL.value
    """Dimensional framework schema (manages FDUs). Can be str, dict, or Schema object."""

    # :attr: _model
    _model: Optional[Matrix] = None
    """Dimensional matrix for Buckingham Pi analysis."""

    # Generated results
    # :attr: _coefficients
    _coefficients: Optional[Coefficients] = field(default_factory=dict)
    """Dictionary of generated dimensionless coefficients (can be Coefficient objects or dicts)."""

    # Workflow state
    # :attr: _is_solved
    _is_solved: bool = False
    """Flag indicating if dimensional matrix has been solved."""

    def __post_init__(self) -> None:
        """*__post_init__()* Initializes the solver.

        Validates basic properties and sets up default values.
        """
        # Initialize from base class
        super().__post_init__()

        # Set default symbol if not specified
        if not self._sym:
            self._sym = f"Solver_{{{self._idx}}}" if self._idx >= 0 else "Solver_{-1}"

        if not self._alias:
            self._alias = latex_to_python(self._sym)

        # Set name and description if not already set
        if not self.name:
            self.name = f"Dimensional Analysis Engine {self._idx}"

        if not self.description:
            self.description = "Solves dimensional analysis using the Buckingham Pi-Theorem."

        # Convert default schema string to Schema object
        # if isinstance(self._schema, str):
            # self._schema = Schema(_fwk=self._schema)
        elif self._schema is not Framework.PHYSICAL.value:
            if isinstance(self._schema, str):
                self.schema = Schema(self._schema)

        # Initialize with default PHYSICAL framework if not already set
        if not hasattr(self, "_schema") or self._schema is None:
            self._schema = Schema(_fwk=Framework.PHYSICAL.value)

    # ========================================================================
    # Validation Methods
    # ========================================================================

    def _validate_dict(self,
                       dt: Dict[str, Any],
                       exp_type: Union[type, Tuple[type, ...]]) -> bool:
        """*_validate_dict()* Validates a dictionary with expected value types.

        Args:
            dt (Dict[str, Any]): Dictionary to validate.
            exp_type (Union[type, Tuple[type, ...]]): Expected type(s) for dictionary values.

        Raises:
            ValueError: If the object is not a dictionary.
            ValueError: If the dictionary is empty.
            ValueError: If the dictionary contains values of unexpected types.

        Returns:
            bool: True if the dictionary is valid.
        """
        # variable inspection
        var_name = inspect_var(dt)

        # Validate is dictionary
        if not isinstance(dt, dict):
            _msg = f"{var_name} must be a dictionary. "
            _msg += f"Provided: {type(dt).__name__}"
            raise ValueError(_msg)

        # Validate not empty
        if len(dt) == 0:
            _msg = f"{var_name} cannot be empty. "
            _msg += f"Provided: {dt}"
            raise ValueError(_msg)

        # Convert list to tuple for isinstance()
        type_check = exp_type if isinstance(exp_type, tuple) else (exp_type,) if not isinstance(exp_type, tuple) else exp_type

        # Validate value types
        if not all(isinstance(v, type_check) for v in dt.values()):
            # Format expected types for error message
            if isinstance(exp_type, tuple):
                type_names = " or ".join(t.__name__ for t in exp_type)
            else:
                type_names = exp_type.__name__

            actual_types = [type(v).__name__ for v in dt.values()]
            _msg = f"{var_name} must contain {type_names} values. "
            _msg += f"Provided: {actual_types}"
            raise ValueError(_msg)

        return True

    # ========================================================================
    # Property Getters and Setters
    # ========================================================================

    @property
    def variables(self) -> Optional[Dict[str, Variable]]:
        """*variables* Get the dictionary of variables.

        Returns:
            Dict[str, Variable]: Dictionary of variables.
        """
        return self._variables

    @variables.setter
    @validate_type(dict, Variable, allow_none=False)
    def variables(self, val: Variables) -> None:
        """*variables* Set the dictionary of variables.

        Args:
            val (Variables): Dictionary of variables (Variable objects or dicts).

        Raises:
            ValueError: If dictionary is invalid or contains invalid values.
        """
        # Convert dict values to Variable objects if needed
        converted = {}
        for key, value in val.items():
            # if value is already a Variable, keep it
            if isinstance(value, Variable):
                converted[key] = value
            # if value is a dict, convert to Variable
            elif isinstance(value, dict):
                # Convert dict to Variable
                converted[key] = Variable.from_dict(value)
            else:
                _msg = f"Input '{key}' must be type 'Variable' or 'dict'. "
                _msg += f"Provided: {type(value).__name__}"
                raise ValueError(_msg)

        self._variables = converted
        self._is_solved = False  # Reset solve state

    @property
    def schema(self) -> Schema:
        """*schema* Get the dimensional framework schema.

        Returns:
            Optional[Schema]: Dimensional framework schema.
        """
        return self._schema

    @schema.setter
    @validate_type(str, dict, Schema, allow_none=False)
    def schema(self, val: Optional[FDUs]) -> None:
        """*schema* Set the dimensional framework schema.

        Args:
            val (Optional[FDUs]): Dimensional framework schema (str, dict, or Schema object).

        Raises:
            ValueError: If string is not a valid framework name or dict is invalid.
        """
        # # if schema is none, setup the default PHYSICAL framework
        # if val is None:
        #     self._schema = Schema(_fwk=Framework.PHYSICAL.value)
        # if schema is a string, convert to Schema
        if isinstance(val, str):
            self._schema = Schema(_fwk=val.upper())
        # if schema is a dict, convert to Schema
        elif isinstance(val, dict):
            self._schema = Schema.from_dict(val)
        # if schema is already a Schema, keep it
        elif isinstance(val, Schema):
            self._schema = val
        else:
            _msg = "Input must be type 'str', 'dict', or 'Schema'. "
            _msg += f"Provided: {type(val).__name__}"
            raise ValueError(_msg)

    @property
    def matrix(self) -> Optional[Matrix]:
        """*matrix* Get the dimensional matrix.

        Returns:
            Optional[Matrix]: Dimensional matrix.
        """
        return self._model

    @matrix.setter
    @validate_type(Matrix, allow_none=True)
    def matrix(self, val: Optional[Matrix]) -> None:
        """*matrix* Set the dimensional matrix.

        Args:
            val (Optional[Matrix]): Dimensional matrix.
        """
        self._model = val
        if val is not None:
            self._is_solved = False  # Reset solve state

    @property
    def coefficients(self) -> Optional[Dict[str, Coefficient]]:
        """*coefficients* Get the generated coefficients.

        Returns:
            Optional[Dict[str, Coefficient]]: Dictionary of coefficients.
        """
        return self._coefficients

    @coefficients.setter
    @validate_type(dict, Coefficient, allow_none=False)
    def coefficients(self, val: Coefficients) -> None:
        """*coefficients* Set the generated coefficients.

        Args:
            val (Coefficients): Dictionary of coefficients (Coefficient objects or dicts).

        Raises:
            ValueError: If dictionary is invalid or contains invalid values.
        """
        # Convert dict values to Coefficient objects if needed
        converted = {}
        for key, value in val.items():
            # if value is already a Coefficient, keep it
            if isinstance(value, Coefficient):
                converted[key] = value
            # if value is a dict, convert to Coefficient
            elif isinstance(value, dict):
                # Convert dict to Coefficient
                converted[key] = Coefficient.from_dict(value)
            else:
                _msg = f"Input '{key}' must be type 'Coefficient' or 'dict'. "
                _msg += f"Provided: {type(value).__name__}"
                raise ValueError(_msg)

        self._coefficients = converted
        self._is_solved = False  # Reset solve state

    @property
    def is_solved(self) -> bool:
        """*is_solved* Check if dimensional matrix has been solved.

        Returns:
            bool: True if solved, False otherwise.
        """
        return self._is_solved

    # ========================================================================
    # Workflow Methods
    # ========================================================================

    def create_matrix(self, **kwargs) -> None:
        """*create_matrix()* Create and configure dimensional matrix.

        Creates a Matrix object from the current variables and optional parameters.

        Args:
            **kwargs: Optional keyword arguments to pass to Matrix constructor.

        Returns:
            Matrix: Configured dimensional matrix.

        Raises:
            ValueError: If variables are not set.
        """
        if not self._variables:
            raise ValueError("Variables must be set before creating matrix.")

        # Create matrix with variables
        self._model = Matrix(
            _idx=self.idx,
            _fwk=self._fwk,
            _schema=self._schema,
            _variables=self._variables,
            # **kwargs
        )

        self._is_solved = False     # Reset solve state
        # return self._model

    def solve(self) -> Dict[str, Coefficient]:
        """*solve()* Solve the dimensional matrix and generate coefficients.

        Performs dimensional analysis using the Buckingham Pi theorem to generate
        dimensionless coefficients.

        Returns:
            Dict[str, Coefficient]: Dictionary of generated coefficients.

        Raises:
            ValueError: If matrix is not created.
            RuntimeError: If solving fails.
        """
        if self._model is None:
            raise ValueError("Matrix must be created before solving. Call create_matrix() first.")

        try:
            # Solve the matrix (generate coefficients)
            # self._model.create_matrix()
            self._model.solve_matrix()
            # self._model.solve()

            # Extract generated coefficients from matrix
            self._coefficients = self._model.coefficients
            self._is_solved = True
            return self._coefficients

        except Exception as e:
            _msg = f"Failed to solve dimensional matrix: {str(e)}"
            raise RuntimeError(_msg) from e

    def run_analysis(self) -> Dict[str, Coefficient]:
        """*run_analysis()* Execute complete dimensional analysis workflow.

        Convenience method that runs the entire workflow: create matrix and solve.

        Args:
            **kwargs: Optional keyword arguments for Matrix constructor.

        Returns:
            Dict[str, Coefficient]: Dictionary of generated dimensionless coefficients.

        Raises:
            ValueError: If variables are not set.
        """
        # Step 1: Create matrix
        if isinstance(self._model, Matrix):
            self.create_matrix()

        # Step 2: Solve
        return self.solve()

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def reset(self) -> None:
        """*reset()* Reset the solver state.

        Clears all generated results, keeping only the input variables.
        """
        self._model = None
        self._coefficients.clear()
        self._is_solved = False

    def clear(self) -> None:
        """*clear()* Reset all attributes to default values.

        Resets all solver properties to their initial state, including variables.
        """
        self._variables.clear()
        self._schema = None
        self.reset()

    def to_dict(self) -> Dict[str, Any]:
        """*to_dict()* Convert solver state to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of solver state.
        """
        return {
            'name': self.name,
            'description': self.description,
            'idx': self._idx,
            'sym': self._sym,
            'alias': self._alias,
            'fwk': self._fwk,
            'variables': {k: v.to_dict() for k, v in self._variables.items()},
            'coefficients': {k: v.to_dict() for k, v in self._coefficients.items()},
            'is_solved': self._is_solved,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AnalysisEngine:
        """*from_dict()* Create a AnalysisEngine instance from a dictionary.

        Args:
            data (Dict[str, Any]): Dictionary containing the solver's state.

        Returns:
            AnalysisEngine: New instance of AnalysisEngine.
        """
        # Create instance with basic attributes
        instance = cls(
            _name=data.get('name', ''),
            description=data.get('description', ''),
            _idx=data.get('idx', -1),
            _sym=data.get('sym', ''),
            _alias=data.get('alias', ''),
            _fwk=data.get('fwk', ''),
        )

        # Set variables
        vars_data = data.get('variables', {})
        if vars_data:
            vars_dict = {k: Variable.from_dict(v) for k, v in vars_data.items()}
            instance.variables = vars_dict

        # Set coefficients
        coefs_data = data.get('coefficients', {})
        if coefs_data:
            coefs_dict = {k: Coefficient.from_dict(v) for k, v in coefs_data.items()}
            instance._coefficients = coefs_dict

        # Set state flags
        instance._is_solved = data.get('is_solved', False)

        return instance

    def __repr__(self) -> str:
        """*__repr__()* String representation of solver.

        Returns:
            str: String representation.
        """
        status = "solved" if self._is_solved else "not solved"
        coef_count = len(self._coefficients) if self._is_solved else 0
        
        return (f"AnalysisEngine(name={self.name!r}, "
                f"variables={len(self._variables)}, "
                f"coefficients={coef_count}, "
                f"status={status})")
