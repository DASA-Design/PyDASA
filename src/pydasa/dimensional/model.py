# -*- coding: utf-8 -*-
"""
Module model.py
============================================

Module for **DimMatrix** to perform Dimensional Analysis in *PyDASA*.

This module provides the DimMatrix class which implements matrix-based dimensional analysis following the Buckingham Pi theorem methodology.

Classes:
    **DimMatrix**: Represents a dimensional matrix for performing dimensional analysis, including methods for matrix creation, solving, and coefficient generation.

*IMPORTANT:* Based on the theory from:
    H. Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""

# native python modules
from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import List, Dict, Optional, Any, Generic
import re

# python third-party modules
import numpy as np
import sympy as sp

# Import validation base classes
from pydasa.core.basic import Validation

# Import related classes
from pydasa.core.parameter import Variable
from pydasa.dimensional.framework import DimSchema
from pydasa.buckingham.vashchy import Coefficient

# Import utils
from pydasa.utils.default import T
from pydasa.utils import config as cfg

# Global constants
MAX_OUT: int = 1
"""Maximum number of output variables allowed."""

MAX_IN: int = 10
"""Maximum number of input variables allowed."""


@dataclass
class DimMatrix(Validation, Generic[T]):
    """**DimMatrix** for Dimensional Analysis in *PyDASA*. Manages the dimensional matrix for performing analysis using the Buckingham Pi theorem methodology.

    Attributes:
        # Core Identification
        name (str): User-friendly name of the dimensional model.
        description (str): Brief summary of the dimensional model.
        _idx (int): Index/precedence of the dimensional model.
        _sym (str): Symbol representation (LaTeX or alphanumeric).
        _alias (str): Python-compatible alias for use in code.
        _fwk (str): Framework context (PHYSICAL, COMPUTATION, SOFTWARE, CUSTOM).

        # Framework Management
        _framework (DimSchema): Dimensional framework managing FDUs.
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
        _dim_mtx (Optional[np.ndarray]): Dimensional matrix (FDUs × Variables).
        _dim_mtx_trans (Optional[np.ndarray]): Transposed dimensional matrix.
        _sym_mtx (Optional[sp.Matrix]): SymPy matrix for symbolic computation.
        _rref_mtx (Optional[np.ndarray]): Row-Reduced Echelon Form matrix.

        # Analysis Results
        _pivot_cols (List[int]): Pivot columns in the RREF matrix.
        _coefficients (Dict[str, Coefficient]): Dimensionless Pi coefficients.
    """

    # ========================================================================
    # Core Identification
    # ========================================================================
    # TODO may be I don't need it
    name: str = "Dimensional Matrix"
    """User-friendly name of the dimensional matrix."""

    description: str = ""
    """Brief summary of the dimensional matrix and its purpose."""

    _idx: int = -1
    """Index/precedence of the dimensional model."""

    _sym: str = ""
    """Symbol representation (LaTeX or alphanumeric)."""

    _alias: str = ""
    """Python-compatible alias for use in code."""

    _fwk: str = "PHYSICAL"
    """Framework context (PHYSICAL, COMPUTATION, SOFTWARE, CUSTOM)."""

    # ========================================================================
    # Framework Management
    # ========================================================================

    _framework: DimSchema = field(default_factory=DimSchema)
    """Dimensional framework managing Fundamental Dimensional Units (FDUs)."""

    working_fdus: List[str] = field(default_factory=list)
    """List of active FDU symbols used in the current analysis."""

    # ========================================================================
    # Variable Management
    # ========================================================================

    _variables: Dict[str, Variable] = field(default_factory=dict)
    """Dictionary of all parameters/variables in the model.

    Keys are variable symbols (str), values are Variable instances.
    """

    _relevant_lt: Dict[str, Variable] = field(default_factory=dict)
    """Dictionary of relevant parameters/variables for dimensional analysis.

    Filtered subset of _variables where Variable.relevant == True. Keys are variable symbols (str), values are Variable instances.

    NOTE: called 'relevant list' by convention.
    """

    _output: Optional[Variable] = None
    """The single output variable for the dimensional analysis.

    Must be a variable with cat == "OUT".
    """

    # ========================================================================
    # Variable Statistics
    # ========================================================================

    _n_var: int = 0
    """Total number of variables in the model."""

    _n_relevant: int = 0
    """Number of variables marked as relevant for analysis."""

    _n_in: int = 0
    """Number of input variables (cat == "IN" and relevant == True)."""

    _n_out: int = 0
    """Number of output variables (cat == "OUT" and relevant == True)."""

    _n_ctrl: int = 0
    """Number of control variables (cat == "CTRL" and relevant == True)."""

    # ========================================================================
    # Matrix Representations
    # ========================================================================

    _dim_mtx: Optional[np.ndarray] = None
    """Dimensional matrix as NumPy array.

    Shape: (n_fdus, n_relevant_vars)
    Each column represents a variable's dimensional formula.
    Each row represents an FDU's exponent across all variables.
    """

    _dim_mtx_trans: Optional[np.ndarray] = None
    """Transposed dimensional matrix.

    Shape: (n_relevant_vars, n_fdus)
    Transpose of _dim_mtx for alternative operations.
    """

    _sym_mtx: Optional[sp.Matrix] = None
    """SymPy Matrix representation for symbolic computation.

    Used for RREF calculation and nullspace computation.
    Equivalent to _dim_mtx but in SymPy format.
    """

    _rref_mtx: Optional[np.ndarray] = None
    """Row-Reduced Echelon Form (RREF) of the dimensional matrix.

    Result of Gaussian elimination on _sym_mtx.
    Used to identify pivot columns and compute nullspace.
    """

    # ========================================================================
    # Analysis Results
    # ========================================================================

    _pivot_cols: List[int] = field(default_factory=list)
    """Indices of pivot columns in the RREF matrix.

    Identifies which variables are dependent (pivot) vs. independent (free).
    """

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

        # Update global configuration from framework
        if self._framework:
            self._framework.update_global_config()

        # Process variables if provided
        if self._variables:
            self._prepare_analysis()

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
        working_fdus = self._extract_fdus()

        # Handle CUSTOM framework
        if self._fwk == "CUSTOM" and working_fdus:
            if not all(fdu in self._framework.fdu_symbols for fdu in working_fdus):
                _msg = f"Invalid CUSTOM FDUs: {working_fdus}. "
                _msg += f"Must be subset of: {self._framework.fdu_symbols}."
                raise ValueError(_msg)

            self.working_fdus = working_fdus
            # Update framework and global configuration
            self._framework.update_global_config()

    def _update_variable_stats(self) -> None:
        """*_update_variable_stats()* Update variable statistics.

        Computes the number of variables, inputs, outputs, and control variables. Validates the model constraints.

        Raises:
            ValueError: If model has invalid variable counts.
        """
        _vars = self._variables.values()

        # Count all variables
        self._n_var = len(_vars)
        self._n_relevant = sum(1 for v in _vars if v.relevant)

        # Count by category (only relevant ones)
        self._n_in = sum(1 for v in _vars if v.cat == "IN" and v.relevant)
        self._n_out = sum(1 for v in _vars if v.cat == "OUT" and v.relevant)
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

        max_inputs = len(self._framework.fdu_symbols)
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
        cat_order = list(cfg.PARAMS_CAT_DT.keys())

        # Sort by category precedence
        sorted_items = sorted(vars_lt.items(),
                              key=lambda v: cat_order.index(v[1].cat))
        # FIXME IA weird lambda function, check later!!!
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
        dim_strings = [v.std_dims for v in self._relevant_lt.values()]

        # Extract FDU symbols using regex
        fdus = [d for d in re.findall(cfg.WKNG_FDU_SYM_RE, str(dim_strings))]

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
        """
        if not self._relevant_lt:
            raise ValueError("No relevant variables to create matrix from.")

        # Get dimensions
        n_fdu = len(self._framework.fdu_symbols)
        n_var = len(self._relevant_lt)

        # Initialize empty matrix
        self._dim_mtx = np.zeros((n_fdu, n_var), dtype=float)

        # Fill matrix with dimension columns
        for var in self._relevant_lt.values():
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
        if self._dim_mtx is None:
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
        nullspace_vectors = self._sym_mtx.nullspace()

        # Clear existing coefficients
        self._coefficients.clear()

        # Get variable symbols in order
        var_syms = list(self._relevant_lt.keys())

        # Create coefficient for each nullspace vector
        for i, vector in enumerate(nullspace_vectors):
            # Convert to numpy array
            vector_np = np.array(vector).flatten().astype(float)

            # Create variable dictionary for this coefficient
            coef_vars = {}
            for j, val in enumerate(vector_np):
                if j < len(var_syms) and abs(val) > 1e-10:
                    coef_vars[var_syms[j]] = self._relevant_lt[var_syms[j]]

            # Create Pi coefficient
            pi_sym = f"\\Pi_{{{i}}}"
            coef = Coefficient(
                _idx=i,
                _sym=pi_sym,
                _alias=f"Pi_{i}",
                _fwk=self._fwk,
                _cat="COMPUTED",
                _variables=coef_vars,
                _dim_col=vector_np.tolist(),
                _pivot_lt=self._pivot_cols,
                name=f"Pi-{i}",
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
        if idx is -1:
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
            name=name,
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
        """*clear()* Resets the dimensional matrix and analysis results.

        Clears all computed results while preserving the framework.
        Variables are cleared by default.
        """
        # Clear base class
        super().clear()

        # Clear variables
        self._variables.clear()
        self._relevant_lt.clear()
        self._output = None

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
        self._prepare_analysis()

    @property
    def framework(self) -> DimSchema:
        """*framework* Get the dimensional framework.

        Returns:
            DimSchema: Current dimensional framework.
        """
        return self._framework

    @framework.setter
    def framework(self, val: DimSchema) -> None:
        """Set the dimensional framework.

        Args:
            val (DimSchema): New dimensional framework.

        Raises:
            ValueError: If input is not a DimSchema instance.
        """
        if not isinstance(val, DimSchema):
            _msg = "Framework must be a DimSchema instance. "
            _msg += f"Got: {type(val).__name__}"
            raise ValueError(_msg)

        # Update framework and global configuration
        self._framework = val
        self._framework.update_global_config()

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
        _vars = self._variables
        self._relevant_lt = {k: v for k, v in _vars.items() if v.relevant}

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
    def dim_mtx(self) -> Optional[np.ndarray]:
        """*dim_mtx* Get the dimensional matrix.

        Returns:
            Optional[np.ndarray]: Dimensional matrix, or None.
        """
        return self._dim_mtx if self._dim_mtx is not None else None

    @property
    def rref_mtx(self) -> Optional[np.ndarray]:
        """*rref_mtx* Get the RREF matrix.

        Returns:
            Optional[np.ndarray]: RREF matrix, or None.
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

            # Skip numpy arrays (convert to list for JSON compatibility)
            if isinstance(attr_value, np.ndarray):
                attr_value = attr_value.tolist()

            # Skip sympy matrices (convert to list)
            if isinstance(attr_value, sp.Matrix):
                attr_value = [[float(val) for val in row] for row in attr_value.tolist()]

            # Handle DimSchema framework (convert to dict)
            if isinstance(attr_value, DimSchema):
                attr_value = attr_value.to_dict()

            # Handle Variable dictionaries (convert each Variable)
            if isinstance(attr_value, dict) and all(isinstance(v, Variable) for v in attr_value.values()):
                attr_value = {k: v.to_dict() for k, v in attr_value.items()}

            # Handle Coefficient dictionaries (convert each Coefficient)
            if isinstance(attr_value, dict) and all(isinstance(c, Coefficient) for c in attr_value.values()):
                attr_value = {k: c.to_dict() for k, c in attr_value.items()}

            # Handle Variable instance (output variable)
            if isinstance(attr_value, Variable):
                attr_value = attr_value.to_dict()

            # Skip None values for optional fields
            if attr_value is None:
                continue

            # Remove leading underscore from private attributes
            if attr_name.startswith("_"):
                clean_name = attr_name[1:]  # Remove first character
            else:
                clean_name = attr_name

            result[clean_name] = attr_value

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DimMatrix:
        """*from_dict()* Create model from dictionary representation.

        Args:
            data (Dict[str, Any]): Dictionary representation of the model.

        Returns:
            DimMatrix: New DimMatrix instance.
        """
        # Get all valid field names from the dataclass
        field_names = {f.name for f in fields(cls)}

        # Map keys without underscores to keys with underscores
        mapped_data = {}

        for key, value in data.items():
            # Try the key as-is first (handles both _idx and name)
            if key in field_names:
                mapped_data[key] = value
            # Try adding underscore prefix (handles idx -> _idx)
            elif f"_{key}" in field_names:
                mapped_data[f"_{key}"] = value
            # Try removing underscore prefix (handles _name -> name if needed)
            elif key.startswith("_") and key[1:] in field_names:
                mapped_data[key[1:]] = value

        # Convert framework back from dict
        if "framework" in mapped_data or "_framework" in mapped_data:
            framework_data = mapped_data.get("framework") or mapped_data.get("_framework")
            if isinstance(framework_data, dict):
                mapped_data["_framework"] = DimSchema.from_dict(framework_data)

        # Convert Variable dictionaries back
        if "variables" in mapped_data or "_variables" in mapped_data:
            vars_data = mapped_data.get("variables") or mapped_data.get("_variables")
            if isinstance(vars_data, dict):
                mapped_data["_variables"] = {
                    k: Variable.from_dict(v) if isinstance(v, dict) else v
                    for k, v in vars_data.items()
                }

        # Convert relevant_lt back (usually reconstructed, but handle if present)
        if "relevant_lt" in mapped_data or "_relevant_lt" in mapped_data:
            rel_data = mapped_data.get("relevant_lt") or mapped_data.get("_relevant_lt")
            if isinstance(rel_data, dict):
                mapped_data["_relevant_lt"] = {
                    k: Variable.from_dict(v) if isinstance(v, dict) else v
                    for k, v in rel_data.items()
                }

        # Convert output variable back
        if "output" in mapped_data or "_output" in mapped_data:
            output_data = mapped_data.get("output") or mapped_data.get("_output")
            if isinstance(output_data, dict):
                mapped_data["_output"] = Variable.from_dict(output_data)

        # Convert Coefficient dictionaries back
        if "coefficients" in mapped_data or "_coefficients" in mapped_data:
            coef_data = mapped_data.get("coefficients") or mapped_data.get("_coefficients")
            if isinstance(coef_data, dict):
                mapped_data["_coefficients"] = {
                    k: Coefficient.from_dict(c) if isinstance(c, dict) else c
                    for k, c in coef_data.items()
                }

        # Convert lists back to numpy arrays
        for array_key in ["dim_mtx", "_dim_mtx", "dim_mtx_trans","_dim_mtx_trans",
                        "rref_mtx", "_rref_mtx"]:
            if array_key in mapped_data and isinstance(mapped_data[array_key], list):
                mapped_data[array_key] = np.array(mapped_data[array_key])

        # Convert lists back to sympy matrices
        if "sym_mtx" in mapped_data or "_sym_mtx" in mapped_data:
            sym_data = mapped_data.get("sym_mtx") or mapped_data.get("_sym_mtx")
            if isinstance(sym_data, list):
                mapped_data["_sym_mtx"] = sp.Matrix(sym_data)

        # Remove computed/derived fields that shouldn't be passed to constructor
        computed_fields = [
            "n_var", "_n_var",
            "n_relevant", "_n_relevant",
            "n_in", "_n_in",
            "n_out", "_n_out",
            "n_ctrl", "_n_ctrl",
            "relevant_lt", "_relevant_lt",  # Reconstructed from variables
            "output", "_output",  # Found during preparation
            "coefficients", "_coefficients",  # Generated during solve
            "dim_mtx", "_dim_mtx",  # Created during analysis
            "dim_mtx_trans", "_dim_mtx_trans",
            "sym_mtx", "_sym_mtx",
            "rref_mtx", "_rref_mtx",
            "pivot_cols", "_pivot_cols"
        ]
        
        for f in computed_fields:
            mapped_data.pop(f, None)

        # Create model instance
        model = cls(**mapped_data)

        # Variables trigger preparation which recreates derived fields
        # No need to manually set computed fields

        return model

    # def to_dict(self) -> Dict[str, Any]:
    #     """*to_dict* Convert model to dictionary representation.

    #     Returns:
    #         Dict[str, Any]: Dictionary containing model data.
    #     """
    #     return {
    #         "name": self.name,
    #         "description": self.description,
    #         "idx": self._idx,
    #         "sym": self._sym,
    #         "alias": self._alias,
    #         "fwk": self._fwk,
    #         "variables": {k: v.to_dict() for k, v in self._variables.items()},
    #         "coefficients": {k: c.to_dict() for k, c in self._coefficients.items()},
    #         "n_var": self._n_var,
    #         "n_relevant": self._n_relevant,
    #         "n_in": self._n_in,
    #         "n_out": self._n_out,
    #         "n_ctrl": self._n_ctrl
    #     }

    # @classmethod
    # def from_dict(cls, data: Dict[str, Any]) -> "DimMatrix":
    #     """*from_dict* Create model from dictionary representation.

    #     Args:
    #         data (Dict[str, Any]): Dictionary containing model data.

    #     Returns:
    #         DimMatrix: New DimMatrix instance.
    #     """
    #     # Extract variables
    #     variables = {}
    #     if "variables" in data:
    #         variables = {
    #             k: Variable.from_dict(v) for k, v in data["variables"].items()
    #         }

    #     # Remove keys not in constructor
    #     model_data = {
    #         k: v for k, v in data.items()
    #         if k not in ["variables", "coefficients", "n_var", "n_relevant",
    #                     "n_in", "n_out", "n_ctrl"]
    #     }

    #     # Create model
    #     model = cls(**model_data)

    #     # Set variables (triggers preparation)
    #     if variables:
    #         model._variables = variables
    #         model._prepare_analysis()

    #     return model
