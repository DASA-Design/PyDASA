# -*- coding: utf-8 -*-
"""
Module model.py
============================================

Module for **Matrix** to perform Dimensional Analysis in *PyDASA*.

This module provides the Matrix class which implements matrix-based dimensional analysis following the Buckingham Pi theorem methodology.

Classes:
    **Matrix**: Represents a dimensional matrix for performing dimensional analysis, including methods for matrix creation, solving, and coefficient generation.

*IMPORTANT:* Based on the theory from:
    H. Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""

# native python modules
# forward references + postpone eval type hints
from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import List, Dict, Optional, Any, Union
import re

# numeric and symbolic computation
import numpy as np
import sympy as sp
from numpy.typing import NDArray

# basic-core and dimensional analysis imports
from pydasa.core.basic import Foundation
from pydasa.elements.parameter import Variable
from pydasa.dimensional.vaschy import Schema
from pydasa.dimensional.buckingham import Coefficient
# import global variables
from pydasa.core.setup import Frameworks
from pydasa.core.setup import VarCardinality
from pydasa.core.setup import PYDASA_CFG


# Global constants
MAX_OUT: int = 1
"""Maximum number of output variables allowed."""

MAX_IN: int = 10
"""Maximum number of input variables allowed."""


@dataclass
class Matrix(Foundation):
    """**Matrix** for Dimensional Analysis in *PyDASA*. Manages the dimensional matrix for performing analysis using the Buckingham Pi theorem methodology.

    Args:
        Foundation: Foundation class for validation of symbols and frameworks.

    Attributes:
        # Core Identification
        name (str): User-friendly name of the dimensional model.
        description (str): Brief summary of the dimensional model.
        _idx (int): Index/precedence of the dimensional model.
        _sym (str): Symbol representation (LaTeX or alphanumeric).
        _alias (str): Python-compatible alias for use in code.
        _fwk (str): Frameworks context (PHYSICAL, COMPUTATION, SOFTWARE, CUSTOM).

        # FDU Schema Management
        _schema (Schema): Dimensional framework managing FDUs.
        working_fdus (List[str]): Active FDUs used in current analysis.

        # Variable Management
        _variables (Dict[str, Variable]): All variables in the model.
        _relevant_lt (Dict[str, Variable]): Relevant variables for analysis.
        _output (Optional[Variable]): Output variable for analysis.

        # Variable Statistics
        _n_var (int): Total number of variables.
        _n_relevant (int): Number of relevant variables.
        _n_in (int): Number of input variables.
        _n_out (int): Number of output variables.
        _n_ctrl (int): Number of control variables.

        # Matrix Representations
        _dim_mtx (Optional[NDArray[np.float64]]): Dimensional matrix (FDUs × Variables).
        _dim_mtx_trans (Optional[NDArray[np.float64]]): Transposed dimensional matrix.
        _sym_mtx (Optional[sp.Matrix]): SymPy matrix for symbolic computation.
        _rref_mtx (Optional[NDArray[np.float64]]): Row-Reduced Echelon Form matrix.

        # Analysis Results
        _pivot_cols (List[int]): Pivot columns in the RREF matrix.
        _coefficients (Dict[str, Coefficient]): Dimensionless Pi coefficients.
    """

    # ========================================================================
    # Core Identification
    # ========================================================================
    # TODO may be I don't need it
    # :attr: _name
    _name: str = "Dimensional Matrix"
    """User-friendly name of the dimensional matrix."""

    # :attr: description
    description: str = ""
    """Brief summary of the dimensional matrix and its purpose."""

    # :attr: _idx
    _idx: int = -1
    """Index/precedence of the dimensional model."""

    # :attr: _sym
    _sym: str = ""
    """Symbol representation (LaTeX or alphanumeric)."""

    # :attr: _alias
    _alias: str = ""
    """Python-compatible alias for use in code."""

    # :attr: _fwk
    _fwk: str = Frameworks.PHYSICAL.value
    """Frameworks context (PHYSICAL, COMPUTATION, SOFTWARE, CUSTOM)."""

    # ========================================================================
    # Frameworks Management
    # ========================================================================

    # :attr: _schema
    _schema: Schema = field(default_factory=Schema)
    """Dimensional framework managing Fundamental Dimensional Units (FDUs)."""

    # :attr: working_fdus
    working_fdus: List[str] = field(default_factory=list)
    """List of active FDU symbols used in the current analysis."""

    # ========================================================================
    # Variable Management
    # ========================================================================

    # :attr: _variables
    _variables: Dict[str, Variable] = field(default_factory=dict)
    """Dictionary of all parameters/variables in the model.

    Keys are variable symbols (str), values are Variable instances.
    """

    # :attr: _relevant_lt
    _relevant_lt: Dict[str, Variable] = field(default_factory=dict)
    """Dictionary of relevant parameters/variables for dimensional analysis.

    Filtered subset of _variables where Variable.relevant == True. Keys are variable symbols (str), values are Variable instances.

    NOTE: called 'relevant list' by convention.
    """

    # :attr: _output
    _output: Optional[Variable] = None
    """The single output variable for the dimensional analysis.

    Must be a variable with cat == "OUT".
    """

    # ========================================================================
    # Variable Statistics
    # ========================================================================

    # :attr: _n_var
    _n_var: int = 0
    """Total number of variables in the model."""

    # :attr: _n_relevant
    _n_relevant: int = 0
    """Number of variables marked as relevant for analysis."""

    # :attr: _n_in
    _n_in: int = 0
    """Number of input variables (cat == "IN" and relevant == True)."""

    # :attr: _n_out
    _n_out: int = 0
    """Number of output variables (cat == "OUT" and relevant == True)."""

    # :attr: _n_ctrl
    _n_ctrl: int = 0
    """Number of control variables (cat == "CTRL" and relevant == True)."""

    # ========================================================================
    # Matrix Representations
    # ========================================================================

    # :attr: _dim_mtx
    # _dim_mtx: Optional[np.ndarray] = field(default_factory=lambda: np.array([]))
    _dim_mtx: Optional[NDArray[np.float64]] = None
    """Dimensional matrix as NumPy array.

    Shape: (n_fdus, n_relevant_vars)
    Each column represents a variable's dimensional formula.
    Each row represents an FDU's exponent across all variables.
    """

    # :attr: _dim_mtx_trans
    _dim_mtx_trans: Optional[NDArray[np.float64]] = None
    """Transposed dimensional matrix.

    Shape: (n_relevant_vars, n_fdus)
    Transpose of _dim_mtx for alternative operations.
    """

    # :attr: _sym_mtx
    _sym_mtx: Optional[sp.Matrix] = field(default_factory=lambda: sp.Matrix([]))
    """SymPy Matrix representation for symbolic computation.

    Used for RREF calculation and nullspace computation.
    Equivalent to _dim_mtx but in SymPy format.
    """

    # :attr: _rref_mtx
    _rref_mtx: Optional[NDArray[np.float64]] = None
    """Row-Reduced Echelon Form (RREF) of the dimensional matrix.

    Result of Gaussian elimination on _sym_mtx.
    Used to identify pivot columns and compute nullspace.
    """

    # :attr: _nullspace
    _nullspace: List[Union[np.ndarray, sp.Matrix]] = field(default_factory=list)
    """List of nullspace vectors of the dimensional matrix.

    Can be list of arrays or list of sympy vectors"""

    # ========================================================================
    # Analysis Results
    # ========================================================================

    # :attr: _pivot_cols
    _pivot_cols: List[int] = field(default_factory=list)
    """Indices of pivot columns in the RREF matrix.

    Identifies which variables are dependent (pivot) vs. independent (free).
    """

    # :attr: _coefficients
    _coefficients: Dict[str, Coefficient] = field(default_factory=dict)
    """Dictionary of dimensionless Pi coefficients.

    Keys are coefficient symbols (e.g., "\\Pi_{0}"), values are Coefficient instances.
    Generated from the nullspace of the dimensional matrix.
    """

    # ========================================================================
    # Initialization
    # ========================================================================

    def __post_init__(self) -> None:
        """*__post_init__()* Initialize the dimensional matrix.

        Validates variables, sets up the framework, identifies relevant variables, and prepares for dimensional analysis.
        """
        # Initialize base class
        super().__post_init__()

        # # Update global configuration from framework
        # if self._schema:
        #     self._schema.update_global_config()

        # Process variables if provided
        if self._variables:
            self._prepare_analysis()

        # Ensure proper types
        if not isinstance(self._dim_mtx, np.ndarray):
            self._dim_mtx = np.array([], dtype=float)

        if not isinstance(self._sym_mtx, sp.Matrix):
            self._sym_mtx = sp.Matrix([])

    # ========================================================================
    # Preparation Methods
    # ========================================================================

    def _prepare_analysis(self) -> None:
        """*_prepare_analysis()* Prepare the model for dimensional analysis.

        Sets up relevant variables, computes model statistics, identifies the output variable, and extracts working FDUs.

        Raises:
            ValueError: If variable configuration is invalid.
        """
        # Update variable statistics
        self._update_variable_stats()

        # Identify and sort relevant variables
        self._relevant_lt = {
            k: v for k, v in self._variables.items() if v.relevant
        }
        self._relevant_lt = self._sort_by_category(self._relevant_lt)

        # Find the output variable
        self._find_output_variable()

        # Extract working FDUs from relevant variables
        self.working_fdus = self._extract_fdus()

        # Handle CUSTOM framework
        if self._fwk == Frameworks.CUSTOM.value and self.working_fdus:
            _fwk = self._schema
            _w_fdus = self.working_fdus
            if not all(fdu in _fwk.fdu_symbols for fdu in _w_fdus):
                _msg = f"Invalid CUSTOM FDUs: {_w_fdus}. "
                _msg += f"Must be subset of: {_fwk.fdu_symbols}."
                raise ValueError(_msg)

    def _update_variable_stats(self) -> None:
        """*_update_variable_stats()* Update variable statistics.

        Computes the number of variables, inputs, outputs, and control variables. Validates the model constraints.

        Raises:
            ValueError: If model has invalid variable counts.
        """
        _vars = self._variables.values()

        # Count all variables
        self._n_var = len(_vars)
        self._n_relevant = len([v for v in _vars if v.relevant])

        # Count by category (only relevant ones)
        IN = VarCardinality.IN.value
        OUT = VarCardinality.OUT.value
        self._n_in = len([v for v in _vars if v.cat == IN and v.relevant])
        self._n_out = len([v for v in _vars if v.cat == OUT and v.relevant])
        self._n_ctrl = self._n_relevant - self._n_in - self._n_out

        # Validate output count
        if self._n_out == 0:
            _msg = "No output variable defined. At least one output variable"
            _msg += " (cat='OUT', relevant=True) is required."
            raise ValueError(_msg)

        if self._n_out > MAX_OUT:
            _msg = f"Invalid number of outputs: {self._n_out}. "
            _msg += f"Maximum allowed: {MAX_OUT}."
            raise ValueError(_msg)

        # Validate input count
        if self._n_in == 0:
            _msg = "No input variables defined. "
            _msg += "At least one input variable is required."
            raise ValueError(_msg)

        max_inputs = len(self._schema.fdu_symbols)
        if self._n_in > max_inputs:
            _msg = f"Too many input variables: {self._n_in}. "
            _msg += f"Maximum allowed: {max_inputs} (number of FDUs)."
            raise ValueError(_msg)

    def _sort_by_category(self,
                          vars_lt: Dict[str, Variable]) -> Dict[str, Variable]:
        """*_sort_by_category()* Sorts variables by category.

        Sorts variables in order: OUT → IN → CTRL. Updates variable indices to reflect sorted order.

        Args:
            vars_lt (Dict[str, Variable]): Dictionary of variables to sort.

        Returns:
            Dict[str, Variable]: Sorted dictionary of variables.
        """
        # Get category order from global config
        # cat_order = list(PYDASA_CFG.parameter_cardinality)
        cat_order = [c.value for c in PYDASA_CFG.parameter_cardinality]

        # Sort by category precedence
        sorted_items = sorted(vars_lt.items(),
                              key=lambda v: cat_order.index(v[1].cat))

        # # FIXME IA weird lambda function, check later!!!
        # sorted_items = sorted(vars_lt.items(),
        #                       key=lambda v: cat_order.index(v[1].cat) if v[1].cat in cat_order else len(cat_order))

        # Update indices and rebuild dictionary
        sorted_dict = {}
        for i, (k, v) in enumerate(sorted_items):
            v._idx = i
            sorted_dict[k] = v

        return sorted_dict

    def _find_output_variable(self) -> None:
        """*_find_output_variable()* Identifies the output variable.

        Finds the first variable with cat == "OUT" in the relevant list.
        """
        values = self._relevant_lt.values()
        self._output = next((v for v in values if v.cat == "OUT"), None)

    def _extract_fdus(self) -> List[str]:
        """*_extract_fdus()* Extracts FDUs from relevant variables.

        Scans all relevant variables' dimension strings to find which FDUs are actually used.

        Returns:
            List[str]: List of unique FDU symbols used, in precedence order.
        """
        # Collect all dimension strings
        var_dims = [v.std_dims for v in self._relevant_lt.values()]

        # Extract FDU symbols using regex
        fdus = [
            d for d in re.findall(self._schema.fdu_sym_regex, str(var_dims))
        ]

        # Remove duplicates while preserving order
        unique_fdus = list({fdus[i] for i in range(len(fdus))})

        return unique_fdus

    # ========================================================================
    # Matrix Operations
    # ========================================================================

    def create_matrix(self) -> None:
        """*create_matrix()* Builds the dimensional matrix.

        Creates the dimensional matrix by arranging variable dimensions as columns. Each row represents an FDU, each column a variable.

        Raises:
            ValueError: If no relevant variables exist.
            ValueError: If variables have invalid or missing dimensional columns.
        """
        if not self._relevant_lt:
            raise ValueError("No relevant variables to create matrix from.")

        # Validate that all variables have dimensional columns
        invalid_vars = []
        for var in self._relevant_lt.values():
            if not var._dim_col or len(var._dim_col) == 0:
                invalid_vars.append(f"{var._sym} (dims='{var._dims}')")

        if invalid_vars:
            _msg = "Variables with missing or empty dimensional columns detected:\n"
            _msg += "\n".join(f"  - {v}" for v in invalid_vars)
            _msg += "\n\nEnsure all relevant variables have valid '_dims' "
            _msg += "properties (e.g., 'L', 'M*L*T^-2', etc.) and not 'n.a.'"
            raise ValueError(_msg)

        # Get dimensions
        n_fdu = len(self._schema.fdu_symbols)
        n_var = len(self._relevant_lt)

        # Initialize empty matrix
        self._dim_mtx = np.zeros((n_fdu, n_var), dtype=float)

        # Fill matrix with dimension columns
        for var in self._relevant_lt.values():
            # Ensure dimension column has correct length
            dim_col = var._dim_col

            # Pad or truncate to match FDU count
            if len(dim_col) < n_fdu:
                dim_col = dim_col + [0] * (n_fdu - len(dim_col))
            elif len(dim_col) > n_fdu:
                dim_col = dim_col[:n_fdu]

            # Set column in matrix
            self._dim_mtx[:, var._idx] = dim_col

        # Create transposed version
        self._dim_mtx_trans = self._dim_mtx.T

    def solve_matrix(self) -> None:
        """*solve_matrix()* Solves the dimensional matrix.

        Computes the Row-Reduced Echelon Form (RREF) of the matrix, identifies pivot columns, and generates dimensionless coefficients from the nullspace.

        Raises:
            ValueError: If matrix hasn't been created yet.
        """
        # Ensure matrix exists
        if not isinstance(self._dim_mtx, np.ndarray) or self._dim_mtx.size == 0:
            self.create_matrix()

        # Convert to SymPy for symbolic computation
        self._sym_mtx = sp.Matrix(self._dim_mtx)

        # Compute RREF and pivot columns
        rref_result, pivot_cols = self._sym_mtx.rref()

        # Store results
        self._rref_mtx = np.array(rref_result).astype(float)
        self._pivot_cols = list(pivot_cols)

        # Generate coefficients from nullspace
        self._generate_coefficients()

    def _generate_coefficients(self) -> None:
        """*_generate_coefficients()* Generates dimensionless coefficients.

        Creates Coefficient objects from each nullspace vector of the dimensional matrix. Each nullspace vector represents a dimensionless group (Pi coefficient).
        """
        if self._sym_mtx is None:
            _msg = "Symbolic matrix not created. Call solve_matrix() first."
            raise ValueError(_msg)

        # Compute nullspace vectors
        self._nullspace = self._sym_mtx.nullspace()

        # Clear existing coefficients
        self._coefficients.clear()

        # Get variable symbols in order
        var_syms = [var for var in self._relevant_lt.keys()]

        # Create coefficient for each nullspace vector
        for i, vector in enumerate(self._nullspace):
            # Convert to numpy array
            vector_np = np.array(vector).flatten().astype(float)

            # Create variable dictionary for this coefficient
            # TODO is this reduntant? check later!!!
            coef_vars = {}
            for j, val in enumerate(vector_np):
                if j < len(var_syms) and isinstance(val, (int, float)):
                    coef_vars[var_syms[j]] = self._relevant_lt[var_syms[j]]

            # Create Pi coefficient
            pi_sym = f"\\Pi_{{{i}}}"
            coef = Coefficient(
                _idx=i,
                _sym=pi_sym,
                _alias=f"Pi_{i}",
                _fwk=self._fwk,
                _cat="COMPUTED",
                _variables=self._relevant_lt,
                _dim_col=vector_np.tolist(),
                _pivot_lt=self._pivot_cols,
                _name=f"Pi-{i}",
                description=f"Dimensionless coefficient {i} from nullspace"
            )

            self._coefficients[pi_sym] = coef

    # ========================================================================
    # Coefficient Derivation
    # ========================================================================

    def derive_coefficient(self,
                           expr: str,
                           name: str = "",
                           description: str = "",
                           idx: int = -1) -> Coefficient:
        """*derive_coefficient()* Creates a new coefficient derived from existing ones.

        Combines existing dimensionless coefficients using a mathematical expression. The new coefficient is marked as "DERIVED".

        Args:
            expr (str): Mathematical expression using existing coefficients.
                Examples: "\\Pi_{0} * \\Pi_{1}", "\\Pi_{0} / \\Pi_{2}^2"
            name (str, optional): Name for the derived coefficient. Defaults to "Derived-Pi-{idx}".
            description (str, optional): Description of the coefficient. Defaults to "Derived from: {expr}".
            idx (int, optional): Index for the coefficient. If -1, the next available index is used.

        Returns:
            Coefficient: The newly created derived coefficient.

        Raises:
            ValueError: If expression is invalid or references non-existent coefficients.
            ValueError: If expression creates dimensionally inconsistent result.

        Example:
            >>> # Create Reynolds number as ratio of two Pi groups
            >>> Re = model.derive_coefficient(
            ...     expr="\\Pi_{0} / \\Pi_{1}",
            ...     name="Reynolds Number",
            ...     description="Ratio of inertial to viscous forces"
            ... )
        """
        # Validate coefficients exist
        if not self._coefficients:
            _msg = "Cannot derive coefficients. No base coefficients exist yet."
            raise ValueError(_msg)

        # Extract coefficient symbols from expression
        coef_pattern = r"\\Pi_\{\d+\}"
        coef_symbols = re.findall(coef_pattern, expr)

        if not coef_symbols:
            _msg = f"Expression '{expr}' does not contain any valid "
            _msg += "coefficient references (format: \\Pi_{{n}})."
            raise ValueError(_msg)

        # Validate all referenced coefficients exist
        for sym in coef_symbols:
            if sym not in self._coefficients:
                _msg = f"Referenced coefficient {sym} does not exist."
                raise ValueError(_msg)

        # Determine next available index
        if idx == -1:
            existing_indices = [c._idx for c in self._coefficients.values()]
            idx = max(existing_indices) + 1 if existing_indices else 0

        # Generate defaults
        if name == "":
            name = f"Derived-Pi-{idx}"

        if description == "":
            description = f"Derived from: {expr}"

        # Get base coefficient for structure
        base_coef = self._coefficients[coef_symbols[0]]
        new_variables = base_coef._variables.copy()
        new_dim_col = list(base_coef._dim_col)

        # Validate all coefficients use same variables
        for sym in coef_symbols[1:]:
            coef = self._coefficients[sym]
            if set(coef._variables.keys()) != set(new_variables.keys()):
                _msg = f"Coefficient {sym} uses different variables. "
                _msg += "Cannot derive new coefficient."
                raise ValueError(_msg)

        # Parse expression for operations
        parts = re.split(r"(\*|/|\^)", expr)
        current_op = "*"

        for part in parts:
            part = part.strip()

            if part in ("*", "/", "^"):
                current_op = part
            elif re.match(coef_pattern, part):
                coef = self._coefficients[part]

                if current_op == "*":
                    # Multiplication: add exponents
                    new_dim_col = [a + b for a, b in zip(new_dim_col, coef._dim_col)]
                elif current_op == "/":
                    # Division: subtract exponents
                    new_dim_col = [a - b for a, b in zip(new_dim_col, coef._dim_col)]

        # Create derived coefficient
        new_sym = f"\\Pi_{{{idx}}}"
        derived_coef = Coefficient(
            _idx=idx,
            _sym=new_sym,
            _alias=f"Pi_{idx}",
            _fwk=self._fwk,
            _cat="DERIVED",
            _name=name,
            description=description,
            _variables=new_variables,
            _dim_col=new_dim_col,
            _pivot_lt=self._pivot_cols
        )

        # Add to coefficients dictionary
        self._coefficients[new_sym] = derived_coef

        return derived_coef

    # ========================================================================
    # High-Level Methods
    # ========================================================================

    def analyze(self) -> None:
        """*analyze()* Performs complete dimensional analysis

        Executes the full analysis workflow:
        1. Prepare analysis (validate variables, identify output)
        2. Create dimensional matrix
        3. Solve matrix (compute RREF and nullspace)
        4. Generate dimensionless coefficients

        This is the main entry point for dimensional analysis.
        """
        self._prepare_analysis()
        self.create_matrix()
        self.solve_matrix()

    def clear(self) -> None:
        """*clear()* Resets all dimensional matrix and analysis data.

        Clears all computed results while preserving the framework.

        NOTE: Numpy arrays don't have .clear() method, so we reassign. Lists have .clear() method.
        """
        # Clear variables
        # Reassign numpy arrays (no .clear() method)
        self._dim_mtx = np.array([], dtype=float)

        # Reassign sympy matrices
        self._sym_mtx = sp.Matrix([])

        # Clear lists (these have .clear() method)
        self._variables.clear()
        self._relevant_lt.clear()
        self._output = None

        # Reset scalars
        self._idx = -1
        self._sym = ""
        self._alias = ""
        self.name = ""
        self.description = ""

        # Reset statistics
        self._n_var = 0
        self._n_relevant = 0
        self._n_in = 0
        self._n_out = 0
        self._n_ctrl = 0

        # Clear matrices
        self._dim_mtx = None
        self._dim_mtx_trans = None
        self._sym_mtx = None
        self._rref_mtx = None
        self._pivot_cols.clear()

        # Clear results
        self._coefficients.clear()
        self.working_fdus.clear()

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def variables(self) -> Dict[str, Variable]:
        """*variables* Get the dictionary of variables.

        Returns:
            Dict[str, Variable]: Copy of variables dictionary.
        """
        return self._variables

    @variables.setter
    def variables(self, val: Dict[str, Variable]) -> None:
        """*variables* Set the dictionary of variables.

        Args:
            val (Dict[str, Variable]): Dictionary of variables.

        Raises:
            ValueError: If input is not a non-empty dictionary.
            ValueError: If any value is not a Variable instance.
        """
        if not val or not isinstance(val, dict):
            _msg = "Variables must be in non-empty dictionary. "
            _msg += f"Provided input: {type(val).__name__}"
            raise ValueError(_msg)

        if not all(isinstance(v, Variable) for v in val.values()):
            _msg = "All elements must be Variable instances"
            _msg += f", got: {[type(v).__name__ for v in val.values()]}"
            raise ValueError(_msg)

        self._variables = val

        # Update relevant variables and prepare for analysis
        self._prepare_analysis()

    @property
    def framework(self) -> Schema:
        """*framework* Get the dimensional framework.

        Returns:
            Schema: Current dimensional framework.
        """
        return self._schema

    @framework.setter
    def framework(self, val: Schema) -> None:
        """Set the dimensional framework.

        Args:
            val (Schema): New dimensional framework.

        Raises:
            ValueError: If input is not a Schema instance.
        """
        if not isinstance(val, Schema):
            _msg = "Schema must be a Schema instance. "
            _msg += f"Got: {type(val).__name__}"
            raise ValueError(_msg)

        # Update framework and global configuration
        self._schema = val

        # Prepare for analysis with new framework
        if self._variables:
            self._prepare_analysis()

    @property
    def relevant_lt(self) -> Dict[str, Variable]:
        """*relevant_lt* Get dictionary of relevant variables.

        Returns:
            Dict[str, Variable]: Dictionary of relevant variables.
        """
        return self._relevant_lt

    @relevant_lt.setter
    def relevant_lt(self, val: Dict[str, Variable]) -> None:
        """*relevant_lt* Set the dictionary of relevant variables, otherwise known as 'relevance list'.

        Args:
            val (Dict[str, Variable]): Dictionary of relevant variables.

        Raises:
            ValueError: If the relevant variable dictionary is invalid.
            ValueError: If any of the dictionary variables are invalid.
        """
        if not val or not isinstance(val, dict):
            raise ValueError("Variables must be in non-empty dictionary.")

        if not all(isinstance(v, Variable) for v in val.values()):
            raise ValueError("All elements must be Variable instances")

        # Set relevant variables and prepare for analysis
        # self._relevant_lt = [p for p in val if p.relevant]
        self._relevant_lt = {
            k: v for k, v in self._variables.items() if v.relevant
        }

        # Update relevant variables and prepare for analysis
        self._prepare_analysis()

    @property
    def coefficients(self) -> Dict[str, Coefficient]:
        """*coefficients* Get dictionary of dimensionless coefficients.

        Returns:
            Dict[str, Coefficient]: Dictionary of dimensionless coefficients.
        """
        return self._coefficients

    @property
    def output(self) -> Optional[Variable]:
        """*output* Get the output variable.

        Returns:
            Optional[Variable]: The output variable, or None if not set.
        """
        return self._output

    @property
    def dim_mtx(self) -> Optional[NDArray[np.float64]]:
        """*dim_mtx* Get the dimensional matrix.

        Returns:
            Optional[NDArray[np.float64]]: Dimensional matrix, or None.
        """
        return self._dim_mtx if self._dim_mtx is not None else None

    @property
    def rref_mtx(self) -> Optional[NDArray[np.float64]]:
        """*rref_mtx* Get the RREF matrix.

        Returns:
            Optional[NDArray[np.float64]]: RREF matrix, or None.
        """
        return self._rref_mtx if self._rref_mtx is not None else None

    @property
    def pivot_cols(self) -> List[int]:
        """*pivot_cols* Get pivot column indices.

        Returns:
            List[int]: Pivot column list.
        """
        return self._pivot_cols

    # ========================================================================
    # Serialization
    # ========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """*to_dict()* Convert model to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation of the model.
        """
        result = {}

        # Get all dataclass fields
        for f in fields(self):
            attr_name = f.name
            attr_value = getattr(self, attr_name)

            # Skip None values for optional fields
            if attr_value is None:
                continue

            # Convert based on type
            converted_value = self._convert_value(attr_value)

            # Remove leading underscore from private attributes
            clean_name = attr_name[1:] if attr_name.startswith("_") else attr_name
            result[clean_name] = converted_value

        return result

    def _convert_value(self, value: Any) -> Any:
        """Convert a value to JSON-serializable format.

        Args:
            value (Any): Value to convert.

        Returns:
            Any: Converted value.
        """
        # Handle None
        if value is None:
            return None

        # Handle numpy arrays
        if isinstance(value, np.ndarray):
            return value.tolist()

        # Handle sympy matrices
        if isinstance(value, sp.Matrix):
            return [[float(val) for val in row] for row in value.tolist()]

        # Handle objects with to_dict method
        if isinstance(value, (Schema, Variable, Coefficient)):
            return value.to_dict()

        # Handle dictionaries
        if isinstance(value, dict):
            if not value:  # Empty dict
                return {}

            # Check if all values have to_dict method
            first_val = next(iter(value.values()))
            if isinstance(first_val, (Variable, Coefficient)):
                return {k: v.to_dict() for k, v in value.items()}

            # Regular dict
            return value

        # Handle lists
        if isinstance(value, list):
            return [self._convert_value(item) for item in value]

        # Default: return as-is
        return value

    # def to_dict(self) -> Dict[str, Any]:
    #     """*to_dict()* Convert model to dictionary representation.

    #     Returns:
    #         Dict[str, Any]: Dictionary representation of the model.
    #     """
    #     result = {}

    #     # Get all dataclass fields
    #     for f in fields(self):
    #         attr_name = f.name
    #         attr_value = getattr(self, attr_name)

    #         # Skip numpy arrays (convert to list for JSON compatibility)
    #         if isinstance(attr_value, np.ndarray):
    #             attr_value = attr_value.tolist()

    #         # Skip sympy matrices (convert to list)
    #         if isinstance(attr_value, sp.Matrix):
    #             attr_value = [[float(val) for val in row] for row in attr_value.tolist()]

    #         # Handle Schema framework (convert to dict)
    #         if isinstance(attr_value, Schema):
    #             attr_value = attr_value.to_dict()

    #         # Handle Variable dictionaries (convert each Variable)
    #         if isinstance(attr_value, dict) and all(isinstance(v, Variable) for v in attr_value.values()):
    #             attr_value = {k: v.to_dict() for k, v in attr_value.items()}

    #         # Handle Coefficient dictionaries (convert each Coefficient)
    #         if isinstance(attr_value, dict) and all(isinstance(c, Coefficient) for c in attr_value.values()):
    #             attr_value = {k: c.to_dict() for k, c in attr_value.items()}

    #         # Handle Variable instance (output variable)
    #         if isinstance(attr_value, Variable):
    #             attr_value = attr_value.to_dict()

    #         # Skip None values for optional fields
    #         if attr_value is None:
    #             continue

    #         # Remove leading underscore from private attributes
    #         if attr_name.startswith("_"):
    #             clean_name = attr_name[1:]  # Remove first character
    #         else:
    #             clean_name = attr_name

    #         result[clean_name] = attr_value

    #     return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Matrix":
        """*from_dict()* Create model from dictionary representation.

        Args:
            data (Dict[str, Any]): Dictionary representation of the model.

        Returns:
            Matrix: New Matrix instance.
        """
        # Get all valid field names from the dataclass
        field_names = {f.name for f in fields(cls)}

        # Map keys without underscores to keys with underscores
        mapped_data = {}

        # Define conversion rules
        object_converters = {
            "schema": Schema,
            "variables": Variable,
            "relevant_lt": Variable,
            "output": Variable,
            "coefficients": Coefficient
        }

        array_fields = ["dim_mtx", "dim_mtx_trans", "rref_mtx"]
        matrix_fields = ["sym_mtx"]

        # Computed fields that should not be passed to constructor
        computed_fields = {
            "n_var", "n_relevant", "n_in", "n_out", "n_ctrl",
            "relevant_lt", "output", "coefficients",
            "dim_mtx", "dim_mtx_trans", "sym_mtx", "rref_mtx", "pivot_cols"
        }

        for key, value in data.items():
            # Skip computed fields
            clean_key = key[1:] if key.startswith("_") else key
            if clean_key in computed_fields:
                continue

            # Map key to field name
            field_key = None
            if key in field_names:
                field_key = key
            elif f"_{key}" in field_names:
                field_key = f"_{key}"
            elif key.startswith("_") and key[1:] in field_names:
                field_key = key[1:]

            if field_key is None:
                continue

            # Convert objects
            if clean_key in object_converters:
                converter = object_converters[clean_key]

                if isinstance(value, dict):
                    if clean_key in ["variables", "relevant_lt", "coefficients"]:
                        # Dictionary of objects
                        mapped_data[field_key] = {
                            k: converter.from_dict(v) if isinstance(v, dict) else v
                            for k, v in value.items()
                        }
                    else:
                        # Single object
                        mapped_data[field_key] = converter.from_dict(value)

            # Convert arrays
            elif clean_key in array_fields and isinstance(value, list):
                mapped_data[field_key] = np.array(value)

            # Convert matrices
            elif clean_key in matrix_fields and isinstance(value, list):
                mapped_data[field_key] = sp.Matrix(value)

            # Default: use as-is
            else:
                mapped_data[field_key] = value

        # Create model instance
        return cls(**mapped_data)
