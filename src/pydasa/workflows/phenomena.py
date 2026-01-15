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
from typing import Dict, Optional, Any, Union, List

# Import validation base classes
from pydasa.core.basic import Foundation

# Import related classes
from pydasa.elements.parameter import Variable
from pydasa.dimensional.fundamental import Dimension
from pydasa.dimensional.buckingham import Coefficient
from pydasa.dimensional.model import Matrix
from pydasa.dimensional.vaschy import Schema

# Import utils
from pydasa.serialization.parser import latex_to_python

# Import validation decorators
from pydasa.validations.decorators import validate_type
from pydasa.validations.decorators import validate_emptiness
from pydasa.validations.decorators import validate_dict_types

# Import global configuration
from pydasa.core.setup import Frameworks   # , PYDASA_CFG

# custom type hinting
Variables = Union[Dict[str, Variable], Dict[str, Any]]
Coefficients = Union[Dict[str, Coefficient], Dict[str, Any]]
FDUs = Union[str, Dict[str, Any], List[Dict], Schema]


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
        _fwk (str): Frameworks context (PHYSICAL, COMPUTATION, SOFTWARE, CUSTOM).

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
    _variables: Dict[str, Variable] = field(default_factory=dict)
    """Dictionary of all dimensional variables in the problem."""

    # :attr: _schema
    _schema: Optional[Schema] = None
    """Dimensional framework schema. Input can be: None, List[Dict], Dict, or Schema object. After __post_init__, this will always be a Schema instance.
        - If None: Creates default Schema based on _fwk (e.g., PHYSICAL)
        - If List[Dict] or Dict: Creates CUSTOM Schema with provided FDU definitions
        - If Schema: Uses provided Schema object directly
    """

    # :attr: _model
    _model: Optional[Matrix] = None
    """Dimensional matrix for Buckingham Pi analysis."""

    # Generated results
    # :attr: _coefficients
    _coefficients: Dict[str, Coefficient] = field(default_factory=dict)
    """Dictionary of generated dimensionless coefficients."""

    # Workflow state
    # :attr: _is_solved
    _is_solved: bool = False
    """Flag indicating if dimensional matrix has been solved."""

    def __post_init__(self) -> None:
        """*__post_init__()* Post-initialization processing with validation and setup.

        Raises:
            ValueError: If framework is not supported.
            TypeError: If schema is of incorrect type.
            TypeError: If schema list items are not dictionaries.
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

        # Initialize schema based on provided input and framework
        if isinstance(self._schema, Schema):
            # Already a Schema object, use as-is
            pass

        elif self._schema is None:
            # No schema provided - create default based on framework
            if self.fwk == Frameworks.CUSTOM.value:
                _msg = "Custom framework requires '_schema' parameter with FDU definitions (List[Dict] or Dict)"
                raise ValueError(_msg)
            else:
                # Create schema for standard framework (PHYSICAL, COMPUTATION, SOFTWARE)
                self._schema = Schema(_fwk=self.fwk)

        elif isinstance(self._schema, list):
            # List of FDU definitions provided - create CUSTOM schema
            if not all(isinstance(item, dict) for item in self._schema):  # type: ignore
                _msg = "When '_schema' is a list, all items must be dictionaries with FDU definitions"
                raise TypeError(_msg)
            fdu_list = [Dimension(**d) if isinstance(d, dict) else d for d in self._schema]  # type: ignore
            self._schema = Schema(_fwk=Frameworks.CUSTOM.value,
                                  _fdu_lt=fdu_list)

        elif isinstance(self._schema, dict):
            # Single dict provided - could be from_dict or single FDU definition
            # Try from_dict first (has specific structure)
            if "fwk" in self._schema or "_fwk" in self._schema:
                self._schema = Schema.from_dict(self._schema)
            else:
                # Single FDU definition dict - treat as CUSTOM with one dimension
                fdu_list = [Dimension(**self._schema)]
                self._schema = Schema(_fwk=Frameworks.CUSTOM.value,
                                      _fdu_lt=fdu_list)

        else:
            _msg = "'_schema' must be None, List[Dict], Dict, or Schema object. "
            _msg += f"Provided: {type(self._schema).__name__}"
            raise TypeError(_msg)

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _convert_to_objects(self, data: dict, tgt_cls: type) -> dict:
        """*_convert_to_objects()* Converts dictionary values to instances of the target class if they are not already.

        Args:
            data (dict): Dictionary with values to convert.
            tgt_cls (type): Target class type for conversion.
        Returns:
            dict: Dictionary with values as instances of the target class.
        """
        # Convert dict values to Variable/Coefficient objects if needed
        ans = dict()
        for key, val in data.items():
            # if value is already the target class, keep it
            if isinstance(val, tgt_cls):
                ans[key] = val
            # otherwise, convert from dict
            else:
                ans[key] = tgt_cls.from_dict(val)
        # return proper dict
        return ans

    def _convert_to_schema(self, val: FDUs) -> Schema:
        """*_convert_to_schema()* Converts input to a Schema object.

        Args:
            val (FDUs): Dimensional framework schema. Can be str, dict, list of dicts, or Schema object.

        Raises:
            TypeError: If val is not a valid type.
            TypeError: If dict is invalid.

        Returns:
            Schema: Converted Schema object.
        """
        # if schema is already a Schema, return it
        if isinstance(val, Schema):
            return val

        # if schema is a string, convert to Schema
        elif isinstance(val, str):
            return Schema(_fwk=val.upper())

        # if schema is a list of dicts, create CUSTOM schema
        elif isinstance(val, list):
            if not all(isinstance(item, dict) for item in val):
                _msg = "When schema is a list, all items must be dictionaries with FDU definitions."
                raise TypeError(_msg)
            fdu_list = [
                Dimension(**d) if isinstance(d, dict) else d for d in val
            ]
            return Schema(_fwk=Frameworks.CUSTOM.value, _fdu_lt=fdu_list)

        # if schema is a dict, convert to Schema
        elif isinstance(val, dict):
            # Check if it's a complete Schema dict (has framework info)
            if "fwk" in val or "_fwk" in val:
                return Schema.from_dict(val)
            else:
                # Single FDU definition dict - treat as CUSTOM with one dimension
                fdu_list = [Dimension(**val)]
                return Schema(_fwk=Frameworks.CUSTOM.value, _fdu_lt=fdu_list)

        else:
            _msg = "Input must be type non-empty 'str', 'dict', 'list', or 'Schema'. "
            _msg += f"Provided: {type(val).__name__}"
            raise TypeError(_msg)

    # ========================================================================
    # Property Getters and Setters
    # ========================================================================

    @property
    def variables(self) -> Dict[str, Variable]:
        """*variables* Get the dictionary of variables.

        Returns:
            Dict[str, Variable]: Dictionary of variables.
        """
        return self._variables.copy()

    @variables.setter
    @validate_type(dict, Variable, allow_none=False)
    @validate_emptiness()
    @validate_dict_types(str, (Variable, dict))
    def variables(self, val: Variables) -> None:
        """*variables* Set the dictionary of variables.

        Args:
            val (Variables): Dictionary of variables (Variable objects or dicts).

        Raises:
            ValueError: If dictionary is invalid or contains invalid values.
        """
        # convert dict values to Variable objects if needed
        self._variables = self._convert_to_objects(val, Variable)
        self._is_solved = False  # Reset solve state

    @property
    def schema(self) -> Schema:
        """*schema* Get the dimensional framework schema.

        Returns:
            Schema: Dimensional framework schema. Always initialized after __post_init__.
        """
        if not isinstance(self._schema, Schema):
            raise RuntimeError("Schema not properly initialized. This should not happen.")
        return self._schema

    @schema.setter
    @validate_type(str, dict, list, Schema, allow_none=False)
    @validate_emptiness()
    def schema(self, val: FDUs) -> None:
        """*schema* Set the dimensional framework schema.

        Args:
            val (FDUs): Dimensional framework schema. Can be str, dict, list of dicts, or Schema object.

        Raises:
            TypeError: If val is not a valid type.
            TypeError: If dict is invalid.
        """
        # Convert input to Schema object using helper method
        self._schema = self._convert_to_schema(val)

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
    def coefficients(self) -> Dict[str, Coefficient]:
        """*coefficients* Get the generated coefficients.

        Returns:
            Dict[str, Coefficient]: Dictionary of coefficients.
        """
        return self._coefficients.copy()

    @coefficients.setter
    @validate_type(dict, Coefficient, allow_none=False)
    @validate_emptiness()
    @validate_dict_types(str, (Coefficient, dict))
    def coefficients(self, val: Coefficients) -> None:
        """*coefficients* Set the generated coefficients.

        Args:
            val (Coefficients): Dictionary of coefficients (Coefficient objects or dicts).

        Raises:
            ValueError: If dictionary is invalid or contains invalid values.
            ValueError: If dictionary is empty.
        """
        # Convert dict values to Coefficient objects if needed
        self._coefficients = self._convert_to_objects(val, Coefficient)
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
            TypeError: If schema is not set.
        """
        if not self._variables:
            raise ValueError("Variables must be set before creating matrix.")

        # After __post_init__, _schema is guaranteed to be a Schema instance
        if not isinstance(self._schema, Schema):
            raise TypeError("Schema must be set before creating matrix.")

        # Create matrix with variables
        self._model = Matrix(_idx=self.idx,
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
            self._model.create_matrix()
            self._model.solve_matrix()
            # self._model.solve()

            # Extract generated coefficients from matrix
            self._coefficients = self._model.coefficients
            self._is_solved = True
            return self._coefficients.copy()

        except Exception as e:
            _msg = f"Failed to solve dimensional matrix: {str(e)}"
            raise RuntimeError(_msg) from e

    def run_analysis(self) -> Dict[str, Any]:
        """*run_analysis()* Execute complete dimensional analysis workflow. Convenience method that runs the entire workflow: create matrix and solve.

        Returns:
            Dict[str, Any]: Dictionary of generated dimensionless coefficient in native python format

        Raises:
            ValueError: If variables are not set.
        """
        # Step 1: Create matrix if not already created
        if self._model is None:
            self.create_matrix()

        # Step 2: Solve and return coefficients
        # return self.solve()
        # Create + Solve matrix
        coefficients = self.solve()
        results = {k: v.to_dict() for k, v in coefficients.items()}
        return results

    def derive_coefficient(self,
                           expr: str,
                           symbol: str = "",
                           name: str = "",
                           description: str = "",
                           idx: int = -1) -> Coefficient:
        """*derive_coefficient()* Derive a new coefficient from existing ones.

        Creates a new dimensionless coefficient by algebraically combining existing
        Pi coefficients using the expression string.

        Args:
            expr (str): Expression defining the new coefficient using existing Pi symbols. (e.g., "\\Pi_{0}**(-1)" or "\\Pi_{1} * \\Pi_{3}")
            symbol (str, optional): Symbol representation (LaTeX or alphanumeric) for the derived coefficient. Defaults to "" to keep the original (e.g., Pi_{0}).
            name (str, optional): User-friendly name for the derived coefficient.
            description (str, optional): Description of the derived coefficient.
            idx (int, optional): Index/precedence of the derived coefficient. Defaults to -1.

        Returns:
            Coefficient: The newly derived dimensionless coefficient.

        Raises:
            ValueError: If matrix is not created or solved.
            RuntimeError: If derivation fails.

        Example:
            >>> # Derive Reynolds number from inverse of Pi_0
            >>> Re = engine.derive_coefficient(
            ...     expr="\\Pi_{0}**(-1)",
            ...     name="Reynolds Number",
            ...     description="Re = ρvD/μ"
            ... )
            >>> # Derive combined coefficient
            >>> pi_combined = engine.derive_coefficient(
            ...     expr="\\Pi_{1} * \\Pi_{3}",
            ...     name="Combined Coefficient"
            ... )
        """
        if self._model is None:
            raise ValueError("Matrix must be created before deriving coefficients. Call create_matrix() first.")

        if not self._is_solved:
            raise ValueError("Matrix must be solved before deriving coefficients. Call solve() first.")

        try:
            # Delegate to the Matrix's derive_coefficient method
            coef = self._model.derive_coefficient(expr,
                                                  symbol,
                                                  name,
                                                  description,
                                                  idx)
            return coef
        except Exception as e:
            _msg = f"Failed to derive coefficient: {str(e)}"
            raise RuntimeError(_msg) from e

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def reset(self) -> None:
        """*reset()* Reset the solver state.

        Resets all generated results, KEEPING only the input variables.
        """
        self._model = None
        self._coefficients.clear()
        self._is_solved = False

    def clear(self) -> None:
        """*clear()* Reset all attributes to default values.

        Clears all solver properties to their initial state, INCLUDING variables.
        """
        # Clear parent class attributes (idx, sym, alias, name, description)
        super().clear()

        # Clear AnalysisEngine-specific attributes
        self._variables.clear()
        self._schema = Schema(_fwk=Frameworks.PHYSICAL.value)
        self._model = None
        self._coefficients.clear()
        self._is_solved = False

    def to_dict(self) -> Dict[str, Any]:
        """*to_dict()* Convert solver state to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of solver state.
        """
        # After __post_init__, _schema is always a Schema instance
        schema_dict = self._schema.to_dict() if isinstance(self._schema, Schema) else None

        return {
            "name": self.name,
            "description": self.description,
            "idx": self._idx,
            "sym": self._sym,
            "alias": self._alias,
            "fwk": self._fwk,
            "schema": schema_dict,
            "variables": {
                k: v.to_dict() for k, v in self._variables.items()
            },
            "coefficients": {
                k: v.to_dict() for k, v in self._coefficients.items()
            },
            "is_solved": self._is_solved,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AnalysisEngine:
        """*from_dict()* Create a AnalysisEngine instance from a dictionary.

        Args:
            data (Dict[str, Any]): Dictionary containing the solver"s state.

        Returns:
            AnalysisEngine: New instance of AnalysisEngine.
        """
        # Get schema data if present
        schema_data = data.get("schema")
        _schema = Schema.from_dict(schema_data) if schema_data else None

        # Create instance with basic attributes
        instance = cls(
            _name=data.get("name", ""),
            description=data.get("description", ""),
            _idx=data.get("idx", -1),
            _sym=data.get("sym", ""),
            _alias=data.get("alias", ""),
            _fwk=data.get("fwk", ""),
            _schema=_schema,
        )

        # Set variables
        vars_data = data.get("variables", {})
        if vars_data:
            vars_dict = {k: Variable.from_dict(v) for k, v in vars_data.items()}
            instance.variables = vars_dict

        # Set coefficients
        coefs_data = data.get("coefficients", {})
        if coefs_data:
            coefs_dict = {k: Coefficient.from_dict(v) for k, v in coefs_data.items()}
            instance._coefficients = coefs_dict

        # Set state flags
        instance._is_solved = data.get("is_solved", False)

        return instance

    def __repr__(self) -> str:
        """*__repr__()* String representation of solver.

        Returns:
            str: String representation.
        """
        status = "solved" if self._is_solved else "not solved"
        coef_count = len(self._coefficients)

        return (f"AnalysisEngine(name={self.name!r}, "
                f"variables={len(self._variables)}, "
                f"coefficients={coef_count}, "
                f"status={status})")
