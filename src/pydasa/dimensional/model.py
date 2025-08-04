# -*- coding: utf-8 -*-
"""
Module for **DimMatrix** to perform Dimensional Analysis in *PyDASA*.

This module provides the DimMatrix class which implements matrix-based dimensional analysis following the Buckingham Pi theorem methodology.

*IMPORTANT:* Based on the theory from:
    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""

# native python modules
# import dataclass and typing
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Generic

# python third-party modules
import re
import numpy as np
import sympy as sp

# Import validation base classes
from src.pydasa.core.basic import Validation

# Import related classes
from src.pydasa.core.parameter import Variable
from src.pydasa.dimensional.framework import DimFramework
from src.pydasa.buckingham.vashchy import Coefficient

# Import utils
from src.pydasa.utils.default import T
# import global configuration
# import the 'cfg' module to allow global variable edition
from src.pydasa.utils import config as cfg

# global variables
MAX_OUT: int = 1
MAX_IN: int = 10


@dataclass
class DimMatrix(Validation, Generic[T]):
    """**DimMatrix** for Dimensional Analysis in *PyDASA*. Manages the dimensional matrix for performing analysis using the Buckingham Pi theorem methodology.

    Attributes:
        # Identification and Framework
        name (str): User-friendly name of the dimensional model.
        description (str): Brief summary of the dimensional model.
        _idx (int): Index/precedence of the dimensional model.
        _sym (str): Symbol representation (LaTeX or alphanumeric).
        _pyalias (str): Python-compatible alias for use in code.
        _fwk (str): Framework context (PHYSICAL, COMPUTATION, SOFTWARE, CUSTOM).

        # Framework Management
        _framework (DimFramework): Dimensional framework managing FDUs.

        # Variable and Parameter Management
        _variables (List[Variable]): List of all variables in the model.
        _relevant_lt (List[Variable]): List of relevant variables for analysis.

        # Matrix Representation
        _dim_mtx (Optional[np.ndarray]): Dimensional matrix.
        _dim_mtx_trans (Optional[np.ndarray]): Transposed dimensional matrix.
        _sym_mtx (Optional[sp.Matrix]): SymPy matrix for symbolic computation.
        _rref_mtx (Optional[np.ndarray]): Row-Reduced Echelon Form (RREF) matrix.
        _pivot_cols (List[int]): Pivot columns in the RREF matrix.

        # Analysis Results
        _coefficients (List[Coefficient]): List of dimensionless coefficients.

        # Model Statistics
        _n_var (int): Total number of variables.
        _n_relevant (int): Number of relevant variables.
        _n_in (int): Number of input variables.
        _n_out (int): Number of output variables.
        _n_ctrl (int): Number of control variables.
        _output (Optional[Variable]): Output variable for analysis.
    """

    # Identification attributes
    # Don't need them because of Validation Class
    # :attr: name
    name: str = "Dimensional Matrix"
    """User-friendly name of the dimensional matrix."""

    # :attr: description
    description: str = ""
    """Brief summary of the dimensional matrix."""

    # Framework management
    # :attr: _framework
    _framework: DimFramework = field(default_factory=DimFramework)
    """Dimensional framework managing FDUs."""

    # Variable management
    # :attr: _variables
    _variables: List[Variable] = field(default_factory=list)
    """List of all parameters/variables (*Variable*) in the model."""

    # :attr: _relevant_lt
    _relevant_lt: List[Variable] = field(default_factory=list)
    """List of relevant parameters/variables (*Variable*) for analysis."""

    # Matrix representation
    # :attr: _dim_mtx
    _dim_mtx: Optional[np.ndarray] = None
    """Dimensional matrix."""

    # :attr: _dim_mtx_trans
    _dim_mtx_trans: Optional[np.ndarray] = None
    """Transposed dimensional matrix."""

    # :attr: _sym_mtx
    _sym_mtx: Optional[sp.Matrix] = None
    """SymPy matrix for symbolic computation."""

    # :attr: _rref_mtx
    _rref_mtx: Optional[np.ndarray] = None
    """Row-Reduced Echelon Form (RREF) matrix."""

    # :attr: _pivot_cols
    _pivot_cols: List[int] = field(default_factory=list)
    """Pivot columns in the RREF matrix."""

    # Analysis results
    # :attr: _coefficients
    _coefficients: List[Coefficient] = field(default_factory=list)
    """List of dimensionless coefficients( *Coefficient*)."""

    # Model statistics
    # :attr: _n_var
    _n_var: int = 0
    """Total number of variables."""

    # :attr: _n_relevant
    _n_relevant: int = 0
    """Number of relevant variables."""

    # :attr: _n_in
    _n_in: int = 0
    """Number of input variables."""

    # :attr: _n_out
    _n_out: int = 0
    """Number of output variables."""

    # :attr: _n_ctrl
    _n_ctrl: int = 0
    """Number of control variables."""

    # :attr: _output
    _output: Optional[Variable] = None
    """Output variable for analysis."""

    # :attr: working_fdus
    working_fdus: List[str] = field(default_factory=list)
    """List of working FDUs used in the analysis."""

    def __post_init__(self) -> None:
        """*__post_init__()* Initializes the dimensional matrix. Validates variables, sets up the framework, identifies relevant variables, and prepares for dimensional analysis.
        """
        # Initialize base class
        super().__post_init__()

        # Update global configuration from framework
        self._framework.update_global_config()

        # Process variables if provided
        if self._variables:
            self._prepare_analysis()

    def _prepare_analysis(self) -> None:
        """*_prepare_analysis()* Prepares the model for dimensional analysis. Sets up relevant variables, computes model statistics, and identifies the output variable.
        """
        # Update variable statistics
        self._update_variable_stats()

        # Identify relevant variables
        self._relevant_lt = [v for v in self._variables if v.relevant]

        # Sort relevant variables by category
        self._relevant_lt = self._sort_by_category(self._relevant_lt)

        # Find output variable
        self._find_output_variable()

        # Extract working FDUs from relevant variables
        working_fdus = self._extract_fdus()

        # Update framework with working FDUs if using custom framework
        if self._fwk == "CUSTOM" and working_fdus:
            # check if working dims are in framewoerk
            if all(fdu in self._framework.fdu_symbols for fdu in working_fdus):
                self.working_fdus = working_fdus
            else:
                _msg = f"Invalid CUSTOM FDUs: {working_fdus}"
                _msg += f". Must have: {self._framework.fdu_symbols}."
                raise ValueError(_msg)
            # Update framework and global configuration
            # self._framework = custom_framework
            self._framework.update_global_config()

    def _update_variable_stats(self) -> None:
        """*_update_variable_stats()* Updates variable statistics. Computes the number of variables, inputs, outputs, and control variables. Validates the model constraints.

        Raises:
            ValueError: If the model has invalid variable counts.
        """
        # Count variables by category
        _vars = self._variables
        self._n_var = len(_vars)
        self._n_relevant = len([v for v in _vars if v.relevant])
        self._n_in = len([v for v in _vars if v.cat == "IN" and v.relevant])
        self._n_out = len([v for v in _vars if v.cat == "OUT" and v.relevant])
        self._n_ctrl = self._n_relevant - self._n_in - self._n_out

        # Validate output count
        if self._n_out > MAX_OUT:
            _msg = f"Invalid number of outputs: {self._n_out}. "
            _msg += f"Maximum allowed: {MAX_OUT}."
            raise ValueError(_msg)

        if self._n_out == 0:
            _msg = "No output variable defined. "
            _msg += "At least one output variable is required."
            raise ValueError(_msg)

        # Validate input count
        max_inputs = len(self._framework.fdu_symbols)
        if self._n_in > max_inputs:
            _msg = f"Too many input variables: {self._n_in}. "
            _msg += f"Maximum allowed: {max_inputs} (number of FDUs)."
            raise ValueError(_msg)

        if self._n_in == 0:
            _msg = "No input variables defined. "
            _msg += "At least one input variable is required."
            raise ValueError(_msg)

    def _sort_by_category(self, vars_lt: List[Variable]) -> List[Variable]:
        """*_sort_by_category()* Sorts variables by category.

        Args:
            vars_lt (List[Variable]): List of variables to sort.

        Returns:
            List[Variable]: Sorted list of variables (OUT, IN, CTRL).
        """
        # Define category precedence
        # category_order = ["OUT", "IN", "CTRL"]
        _cat_ord = list(cfg.PARAMS_CAT_DT.keys())

        # Sort variables by category
        sorted_vars = sorted(vars_lt, key=lambda v: _cat_ord.index(v.cat))
        # FIXME IA weird lambda function, check later!!!
        # sorted_vars = sorted(vars_lt,
        #                      key=lambda v: _cat_ord.index(v.cat) if v.cat in _cat_ord else len(_cat_ord))
        # Update indices
        for i, var in enumerate(sorted_vars):
            var._idx = i
        return sorted_vars

    def _find_output_variable(self) -> None:
        """*_find_output_variable()* Identifies the output variable. Sets the output variable for the analysis.
        """
        # TODO alt version
        # output_vars = [v for v in self._relevant_lt if v.cat == "OUT"]
        # if output_vars:
        #     self._output = output_vars[0]
        # get output variable in relevant list, none if not found
        # TODO check behaviour!!!!
        self._output = next((p for p in self._variables if p.cat == "OUT"), None)

    def _extract_fdus(self) -> List[str]:
        """*_extract_fdus()* Extracts FDUs from relevant variables.

        Returns:
            List[str]: List of FDU symbols used in relevant variables.
        """
        # same code but without regex
        # # Extract all FDU symbols from variable dimensions
        # working_fdus = cfg.WKNG_FDU_PREC_LT
        # all_fdus = []
        # for var in self._relevant_lt:
        #     if var._dim_col:
        #         for i, dim in enumerate(var._dim_col):
        #             if dim != 0 and i < len(working_fdus):
        #                 all_fdus.append(working_fdus[i])

        # # Remove duplicates and sort by precedence
        # unique_fdus = list(set(all_fdus))
        # ans = sorted(unique_fdus, key=lambda x: working_fdus.index(x))
        # return ans

        # the same but with regex
        match = [p.std_dims for p in self._relevant_lt]
        fdus = [d for d in re.findall(cfg.WKNG_FDU_SYM_RE, str(match))]
        fdus = list({fdus[i] for i in range(len(fdus))})
        # return list({m for p in relevant_lt for m in re.findall(cfg.WKNG_FDU_SYM_RE, p.std_dims)})
        return fdus

    def create_matrix(self) -> None:
        """*create_matrix()* Builds the dimensional matrix. Creates the dimensional matrix by arranging variable dimensions as columns in the matrix.
        """
        # Get number of FDUs and relevant variables
        n_fdu = len(self._framework.fdu_symbols)
        n_var = len(self._relevant_lt)

        # Create empty matrix
        self._dim_mtx = np.zeros((n_fdu, n_var), dtype=float)

        # Fill matrix with dimension columns
        for var in self._relevant_lt:
            # Ensure dimension column has correct length
            dim_col = var._dim_col
            if len(dim_col) < n_fdu:
                dim_col = dim_col + [0] * (n_fdu - len(dim_col))
            elif len(dim_col) > n_fdu:
                dim_col = dim_col[:n_fdu]

            # Set column in matrix
            self._dim_mtx[:, var._idx] = dim_col

        # Create transposed matrix for alternative operations
        self._dim_mtx_trans = self._dim_mtx.T

    def solve_matrix(self) -> None:
        """*solve_matrix()* Solves the dimensional matrix. Computes the Row-Reduced Echelon Form (RREF) of the matrix, identifies pivot columns, and generates dimensionless coefficients.
        """
        # Ensure matrix exists
        if self._dim_mtx is None:
            self.create_matrix()

        # Convert to SymPy matrix for symbolic computation
        self._sym_mtx = sp.Matrix(self._dim_mtx)

        # Compute RREF and pivot columns
        rref_result, pivot_cols = self._sym_mtx.rref()

        # Convert back to numpy for further processing
        self._rref_mtx = np.array(rref_result).astype(float)
        self._pivot_cols = list(pivot_cols)

        # Generate dimensionless coefficients
        self._generate_coefficients()

    def _generate_coefficients(self) -> None:
        """*_generate_coefficients()* Generates dimensionless coefficients. Creates Coefficient objects from the nullspace of the dimensional matrix.
        """
        # Compute nullspace vectors
        nullspace_vectors = self._sym_mtx.nullspace()

        # Clear existing coefficients
        self._coefficients.clear()

        # Extract variable symbols
        var_symbols = [var._sym for var in self._relevant_lt]

        # Create coefficient for each nullspace vector
        for i, vector in enumerate(nullspace_vectors):
            # Convert vector to numpy array
            vector_np = np.array(vector).flatten().astype(float)

            # Create parameter list for those with non-zero coefficients
            param_lt = []
            for j, val in enumerate(vector_np):
                if j < len(var_symbols) and abs(val) > 1e-10:
                    param_lt.append(var_symbols[j])

            # Create coefficient
            coef = Coefficient(
                _idx=i,
                _sym=f"\\Pi_{{{i}}}",
                _fwk=self._fwk,
                _cat="COMPUTED",
                _variables=var_symbols,
                _dim_col=vector_np.tolist(),
                _pivot_lt=self._pivot_cols,
                name=f"Pi-{i}",
                description=f"Dimensionless coefficient {i} from nullspace"
            )
            self._coefficients.append(coef)

    def analyze(self) -> None:
        """*analyze()* Performs complete dimensional analysis. Creates the dimensional matrix and solves it to generate dimensionless coefficients.
        """
        # Prepare for analysis
        self._prepare_analysis()

        # Create and solve the dimensional matrix
        self.create_matrix()
        self.solve_matrix()

    def clear(self) -> None:
        """*clear()* Resets the dimensional matrix and analysis results.
        """
        # Clear base class
        super().clear()

        # Clear variables and statistics
        self._variables.clear()
        self._relevant_lt.clear()
        self._n_var = 0
        self._n_relevant = 0
        self._n_in = 0
        self._n_out = 0
        self._n_ctrl = 0
        self._output = None

        # Clear matrix representations
        self._dim_mtx = None
        self._dim_mtx_trans = None
        self._sym_mtx = None
        self._rref_mtx = None
        self._pivot_cols.clear()

        # Clear analysis results
        self._coefficients.clear()

    # Property getters and setters

    @property
    def variables(self) -> List[Variable]:
        """*variables* Get the list of variables.

        Returns:
            List[Variable]: List of all variables.
        """
        return self._variables.copy()

    @variables.setter
    def variables(self, val: List[Variable]) -> None:
        """*variables* Set the list of variables.

        Args:
            val (List[Variable]): List of variables.

        Raises:
            ValueError: If the variable list is invalid.
        """
        if not val:
            raise ValueError("Variable list cannot be empty")

        if not all(isinstance(v, Variable) for v in val):
            raise ValueError("All elements must be Variable instances")

        # Set variables and update framework
        self._variables = val

        # Update relevant variables and prepare for analysis
        self._prepare_analysis()

    @property
    def framework(self) -> DimFramework:
        """*framework* Get the dimensional framework.

        Returns:
            DimFramework: Dimensional framework.
        """
        return self._framework

    @framework.setter
    def framework(self, val: DimFramework) -> None:
        """*framework* Set the dimensional framework.

        Args:
            val (DimFramework): Dimensional framework.

        Raises:
            ValueError: If the framework is invalid.
        """
        if not isinstance(val, DimFramework):
            raise ValueError("Framework must be a DimFramework instance")

        # Update framework and global configuration
        self._framework = val
        self._framework.update_global_config()

        # Prepare for analysis with new framework
        if self._variables:
            self._prepare_analysis()

    @property
    def relevant_lt(self) -> List[Variable]:
        """*relevant_lt* Get the list of relevant variables.

        Returns:
            List[Variable]: List of relevant variables.
        """
        return self._relevant_lt.copy()

    @relevant_lt.setter
    def relevant_lt(self, val: List[Variable]) -> None:
        """*relevant_lt* Set the list of relevant variables.

        Args:
            val (List[Variable]): List of relevant variables.

        Raises:
            ValueError: If the relevant variable list is invalid.
        """
        if not val:
            raise ValueError("Relevant variable list cannot be empty")

        if not all(isinstance(v, Variable) for v in val):
            raise ValueError("All elements must be Variable instances")

        # Set relevant variables and prepare for analysis
        self._relevant_lt = [p for p in val if p.relevant]
        self._prepare_analysis()

    @property
    def coefficients(self) -> List[Coefficient]:
        """*coefficients* Get the list of dimensionless coefficients.

        Returns:
            List[Coefficient]: List of dimensionless coefficients.
        """
        return self._coefficients.copy()

    @property
    def output(self) -> Optional[Variable]:
        """*output* Get the output variable.

        Returns:
            Optional[Variable]: Output variable.
        """
        return self._output

    @property
    def dim_mtx(self) -> Optional[np.ndarray]:
        """*dim_mtx* Get the dimensional matrix.

        Returns:
            Optional[np.ndarray]: Dimensional matrix.
        """
        return self._dim_mtx.copy() if self._dim_mtx is not None else None

    @property
    def rref_mtx(self) -> Optional[np.ndarray]:
        """*rref_mtx* Get the Row-Reduced Echelon Form matrix.

        Returns:
            Optional[np.ndarray]: RREF matrix.
        """
        return self._rref_mtx.copy() if self._rref_mtx is not None else None

    @property
    def pivot_cols(self) -> List[int]:
        """*pivot_cols* Get the pivot columns.

        Returns:
            List[int]: Pivot column indices.
        """
        return self._pivot_cols.copy()

    def to_dict(self) -> Dict[str, Any]:
        """*to_dict()* Convert dimensional model to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the model.
        """
        return {
            "name": self.name,
            "description": self.description,
            "idx": self._idx,
            "sym": self._sym,
            "fwk": self._fwk,
            "variables": [v.to_dict() for v in self._variables],
            "coefficients": [c.to_dict() for c in self._coefficients],
            "n_var": self._n_var,
            "n_relevant": self._n_relevant,
            "n_in": self._n_in,
            "n_out": self._n_out,
            "n_ctrl": self._n_ctrl
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DimMatrix:
        """*from_dict()* Create dimensional model from dictionary.

        Args:
            data (Dict[str, Any]): Dictionary representation of the model.

        Returns:
            DimMatrix: New dimensional model instance.
        """
        # TODO test this, needs depuration
        # Create variables from dictionary
        variables = []
        if "variables" in data:
            variables = [Variable.from_dict(v) for v in data["variables"]]

        # Remove keys that should not be passed to constructor
        model_data = data.copy()
        for key in ["variables", "coefficients", "n_var", "n_relevant", "n_in", "n_out", "n_ctrl"]:
            if key in model_data:
                del model_data[key]

        # Create model
        model = cls(**model_data)
        model._variables = variables

        # Prepare for analysis
        if variables:
            model._prepare_analysis()

        return model
