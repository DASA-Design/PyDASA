# -*- coding: utf-8 -*-
"""
Module for solving the Dimensional Model in *PyDASA*.

1. **DimensionalModel**:
    - Manages the structure and data of the dimensional system.
    - Handles parameters, Fundamental Dimensions (FDUs), and their relationships.
    - Provides access to metadata and validates framework, symbols, and attributes.

2. **DimensionalAnalyzer**:
   - Extends DimensionalModel to perform dimensional analysis.
   - Builds and solves the dimensional matrix.
   - Generates dimensionless coefficients (e.g., Pi groups).
   - Exports results for further analysis (e.g., Python, JSON, LaTeX).

*IMPORTANT:* Based on the theory from:

    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""

# native python modules
# import dataclass and typing
from typing import List, Optional, Generic, Union
from dataclasses import dataclass, field
import inspect
import re

# python third-party modules
import sympy as sp
# from sympy import Matrix, matrix2numpy
import numpy as np

# custom modules
# Dimensional Analysisis modules
from Src.PyDASA.Units.fdu import FDU
from Src.PyDASA.Units.params import Parameter
from Src.PyDASA.Pi.coef import PiCoefficient

# data structures modules
from Src.PyDASA.DStructs.Tables.scht import SCHashTable

# FDU regex manager
from Src.PyDASA.Utils.cstm import RegexManager

# generic error handling and type checking
from Src.PyDASA.Utils.err import error_handler as _error
from Src.PyDASA.Utils.dflt import T

# import the 'cfg' module with to allow global variable edition
from Src.PyDASA.Utils import cfg

# Generalizing input type for class typing
FDUElm = Union[FDU, dict]

# checking custom modules
assert _error
assert T

# global variables
MAX_OUT: int = 1
MAX_IN: int = 10


@dataclass
class DimensionalModel(Generic[T]):
    """*DimensionalModel* Represents a Dimensional Model in *PyDASA*.

    Args:
        Generic (T): Generic type for a Python data structure.

    Returns:
        DimensionalModel: An object with the following attributes:
            - _idx (int): Index of the *DimensionalModel*.
            - _sym (str): Symbol of the *DimensionalModel*.
            - _fwk (str): Framework of the *DimensionalModel* using the *FDU_FWK_DT* map. By default, it is set to `PHYSICAL`.
            - _fdu_ht (SCHashTable): Custom FDU hash table.
            - _fdu_regex (RegexManager): FDU regex manager.
            - _param_lt (List[Parameter]): List of *Parameter* objects.
            - _relevance_lt (List[Parameter]): List of relevant *Parameter* objects.
            - name (str): Name of the *DimensionalModel*.
            - description (str): Summary of the *DimensionalModel*.
            - io_fdu (Optional[List[dict]]): Optional list of user-defined FDUs (Fundamental Dimensions).
    """
    # Private attributes with validation logic
    # :attr: _idx
    _idx: int = -1
    """
    Unique identifier/index of the *DimensionalModel*.
    """

    # Symbol of the FDU
    # :attr: _sym
    _sym: str = ""
    """
    Symbol of the *DimensionalModel*. It must be alphanumeric (preferably a single character + Latin or Greek letter). Useful for user-friendly representation of the instance.
    """

    # Working framework of the FDU
    # :attr: _fwk
    _fwk: str = "PHYSICAL"
    """
    Framework of the *DimensionalModel*, can be one of the following: `PHYSICAL`, `COMPUTATION`, `DIGITAL` or `CUSTOM`. By default, it is set to `PHYSICAL`.
    """

    # FDUs hash table
    # :attr: _fdu_ht
    _fdu_ht: SCHashTable = field(default_factory=SCHashTable)
    """
    Custom Hash Table for the *DimensionalModel* FDUs. It is used to store the FDUs and their properties.
    """

    # FDU regex manager
    # :attr: _fdu_regex
    _fdu_regex: RegexManager = field(default_factory=RegexManager)
    """
    FDU Regex Manager of the *DimensionalModel*. It is used to validate the user-defined FDUs and their properties if necessary.
    """

    # List of parameters
    # :attr: _param_lt
    _param_lt: List[Parameter] = field(default_factory=list)
    """
    List of *Parameter* objects. It is used to store the parameters of the system and to create the relevance list.
    """

    # List of relevant parameters
    # :attr: _relevance_lt
    _relevance_lt: List[Parameter] = field(default_factory=list)
    """
    List of relevant *Parameter* objects. It is a subset of the `_param_lt` list with the 'relevance' attribute set to `True`. It is the basis for the Dimensional Matrix.
    """

    # Public attributes
    # :attr: name
    name: str = ""
    """
    User-friendly name of the *DimensionalModel*.
    """

    # :attr: description
    description: str = ""
    """
    Small summary of the *DimensionalModel*.
    """

    # list of user defined FDUs
    # :attr: io_fdu
    io_fdu: Optional[List[FDUElm]] = None
    """
    User-defined FDUs. The list can be Python dictionaries or *FDU* objects. It defines the FDUs used in the *DimensionalModel*.
    """

    def __post_init__(self) -> None:
        """*__post_init__()* initializes the *DimensionalModel* object. It sets up the FDU hash table, FDU precedence list, and FDU regex manager.
        """
        if self._fdu_ht.empty:
            self._filter_fdu_map()

        # if self._fdu_regex.fdu_prec_lt is None:
        self._setup_fdu_precedence()
        self._setup_fdu_regex()

    def _filter_fdu_map(self) -> None:
        """*_filter_fdu_map()* Sets up the hash table for FDUs based on the framework.

        Raises:
            ValueError: error if the framework is invalid.
        """
        if self._fwk in cfg.FDU_FWK_DT and self._fwk != "CUSTOM":
            self._configure_default_fdus()
        elif self._fwk == "CUSTOM" and self.io_fdu:
            self._configure_custom_fdus()
        else:
            _msg = f"Invalid framework: {self._fwk}. "
            _msg += "Must be one of the following: "
            _msg += f"{', '.join(cfg.FDU_FWK_DT.keys())}."
            raise ValueError(_msg)

    def _configure_default_fdus(self) -> None:
        """*_configure_default_fdus()* Configures default FDUs in the hash table.

        Raises:
            ValueError: error if the framework is invalid.
        """
        # map for easy access to the FDUs
        _frk_dt = {
            "PHYSICAL": cfg.PHY_FDU_PREC_DT,
            "COMPUTATION": cfg.COMPU_FDU_PREC_DT,
            "DIGITAL": cfg.DIGI_FDU_PREC_DT,
        }
        # get the framework dictionary
        fdu_data = _frk_dt.get(self._fwk, {})
        # if the framework is not valid, raise an error
        if not fdu_data:
            _msg = f"Invalid framework: {self._fwk}. "
            _msg += "Must be one of the following: "
            _msg += f"{', '.join(cfg.FDU_FWK_DT.keys())}."
            raise ValueError(_msg)
        # otherwise, configure the FDUs
        for i, (_sym, _desc) in enumerate(fdu_data.items()):
            _fdu = FDU(_sym=_sym,
                       _fwk=self._fwk,
                       _idx=i,
                       name=_desc,
                       description=_desc)
            self._fdu_ht.insert(_fdu.sym, _fdu)

    def _configure_custom_fdus(self) -> None:
        """*configure_custom_fdus()* Configures custom FDUs provided by the user.

        NOTE: this method uses the optional `io_fdu` attribute to determine the FDUs.
        """
        # TODO maybe I can improve this for more genericity
        for i, fdu_data in enumerate(self.io_fdu):
            # standarize all frameworks to be the same
            fdu_data["_fwk"] = self._fwk
            fdu_data["_idx"] = i
            fdu = FDU(**fdu_data)
            self._fdu_ht.insert(fdu.sym, fdu)

    def _setup_fdu_precedence(self) -> None:
        """*_setup_fdu_precedence()* Sets up the FDU precedence list with the FDUs in the hash table.

        Raises:
            ValueError: error if the FDU hash table is empty.
        """
        # TODO dont like this global, maybe improve this method
        global MAX_IN
        if self._fdu_ht.empty:
            _msg = "FDU hash table is empty. "
            _msg += f"Current Size: {self._fdu_ht.size()}."
            _msg += "Please configure the FDU hash table before setting up the precedence list."
            raise ValueError(_msg)
        cfg.DFLT_FDU_PREC_LT = list(self._fdu_ht.keys())
        MAX_IN = len(cfg.DFLT_FDU_PREC_LT)

    def _setup_fdu_regex(self) -> None:
        """*_setup_fdu_regex()* Sets up the FDU regex manager with the FDU precedence list.
        """
        _PREC_LT = cfg.DFLT_FDU_PREC_LT
        self._fdu_regex = RegexManager(_fwk=self._fwk,
                                       _fdu_prec_lt=_PREC_LT)
        self._fdu_regex.update_global_regex()

    @property
    def idx(self) -> int:
        """*idx* Get the *DimensionalModel* index in the program.

        Returns:
            int: ID of the *DimensionalModel*.
        """
        return self._idx

    @idx.setter
    def idx(self, value: int) -> None:
        """*idx* Sets the *DimensionalModel* index in the program. It must be an integer.

        Args:
            value (int): Index of the *DimensionalModel*.

        Raises:
            ValueError: error if the Index is not an integer.
        """
        if not isinstance(value, int):
            _msg = "Index must be an integer, "
            _msg += f"Provided type: {type(value)}"
            raise ValueError(_msg)
        self._idx = value

    @property
    def sym(self) -> str:
        """*sym* Get the symbol of the *DimensionalModel*.

        Returns:
            str: Symbol of the *DimensionalModel*.
        """
        return self._sym

    @sym.setter
    def sym(self, value: str) -> None:
        """*sym* Sets the symbol of the *DimensionalModel*. It must be alphanumeric (preferably a single character, a Latin or Greek letter).

        Args:
            value (str): Symbol of the *DimensionalModel*.

        Raises:
            ValueError: error if the symbol is not alphanumeric.
        """
        if not value.isalnum():
            _msg = "Symbol must be alphanumeric. "
            _msg += f"Provided: {value}"
            _msg += "Preferably a Latin or Greek letter."
            raise ValueError(_msg)
        self._sym = value

    @property
    def fwk(self) -> str:
        """*fwk* Gets the working framework of the *DimensionalModel*.

        Returns:
            str: Working framework of the *DimensionalModel*.
        """
        return self._fwk

    @fwk.setter
    def fwk(self, value: str) -> None:
        """*fwk* Sets the working framework of the *DimensionalModel*.

        Args:
            value (str): Worjing Framework of the *DimensionalModel*. It must be a supported FDU framework

        Raises:
            ValueError: error if value is not a valid framework.
        """
        if value not in cfg.FDU_FWK_DT.keys():
            _msg = f"Invalid framework: {value}. "
            _msg += "Must be one of the following: "
            _msg += f"{', '.join(cfg.FDU_FWK_DT.keys())}."
            raise ValueError(_msg)
        self._fwk = value

    @property
    def param_lt(self) -> List[Parameter]:
        """*param_lt* property to get the list of parameters in the *DimensionalModel*.

        Returns:
            List[Parameter]: List of *Parameter* objects.
        """
        return self._param_lt

    @param_lt.setter
    def param_lt(self, value: List[Parameter]) -> None:
        """*param_lt* property to set the list of parameters in the *DimensionalModel*. It must be a list of *Parameter* objects.

        Args:
            value (List[Parameter]): List of *Parameter* objects.

        Raises:
            ValueError: If the parameter list is empty.
        """
        if not value:
            _msg = "Parameter list cannot be empty. "
            _msg += f"Provided: {value}"
            raise ValueError(_msg)
        if not all(isinstance(p, Parameter) for p in value):
            _msg = "Parameter list must be a list of Parameter objects. "
            _msg += f"Provided: {value}"
            raise ValueError(_msg)
        value = self._adjust_param_lt(value)
        self._param_lt = value

    def _adjust_param_lt(self, param_lt: List[Parameter]) -> List[Parameter]:
        """*_adjust_param_lt()* adjusts the parameter list to set the framework and index of the parameters.

        Args:
            param_lt (List[Parameter]): List of *Parameter* objects.

        Returns:
            List[Parameter]: Adjusted list of *Parameter* objects.
        """
        # validate the parameter list
        if self._validate_param_lt(param_lt):
            # check if there is a precedence on the parameters
            if all(p.idx == -1 for p in param_lt):
                # set the index of the parameters
                for i, param in enumerate(param_lt):
                    param.idx = i
                    # fix the framework of the parameters
                    param.fwk = self._fwk
        return param_lt

    def _validate_param_lt(self, param_lt: List[Parameter]) -> bool:
        """*_validate_param_lt()* validates the parameter list. It checks if the number of inputs, outputs, and control parameters are valid.

        Args:
            param_lt (List[Parameter]): List of *Parameter* objects.

        Raises:
            ValueError: error if there is more than one output *Parameter*.
            ValueError: error if there is no output *Parameter*.
            ValueError: error if there are more inputs *Parameter* than FDUs.
            ValueError: error if there are no input *Parameter*.

        Returns:
            bool: True if the parameter list is valid, False otherwise.
        """
        # count the relevant inputs, outputs, and control parameters
        self._n_param = len(param_lt)
        self._n_in = len([p for p in param_lt if p.cat == "INPUT"])
        self._n_out = len([p for p in param_lt if p.cat == "OUTPUT"])
        self._n_relv = len([p for p in param_lt if p.relevant is True])
        self._n_ctrl = self._n_relv - self._n_in - self._n_out

        # check if the number of outputs is valid, ONLY ONE OUTPUT!
        if self._n_out > MAX_OUT:
            _msg = f"Number of outputs is invalid: {self._n_out}. "
            _msg += f"Maximum allowed: {MAX_OUT}."
            raise ValueError(_msg)
        if self._n_out == 0:
            _msg = "No output parameter defined. "
            _msg += "At least one output parameter is required."
            raise ValueError(_msg)
        # check if the number of inputs is valid
        if self._n_in > MAX_IN:
            _msg = f"Number of inputs is invalid: {self._n_in}. "
            _msg += f"Maximum allowed: {MAX_IN}."
            raise ValueError(_msg)
        if self._n_in == 0:
            _msg = "No input parameter defined. "
            _msg += "At least one input parameter is required."
            raise ValueError(_msg)
        return True

    @property
    def relevance_lt(self) -> List[Parameter]:
        """*relevance_lt* property to get the list of relevant parameters in the *DimensionalModel*. It is a subset of the `_param_lt` list with the 'relevant' attribute set to `True`.

        Returns:
            List[Parameter]: List of relevant *Parameter* objects.
        """
        return self._relevance_lt

    @relevance_lt.setter
    def relevance_lt(self, param_lt: List[Parameter]) -> None:
        """*set_relevance_lt()* sets the relevance list of the *DimensionalModel*. It is a subset of the `_param_lt` list with the 'relevance' attribute set to `True`.

        Args:
            param_lt (List[Parameter]): List of *Parameter* objects.
        """
        self._relevance_lt = [p for p in param_lt if p.relevant]

    def _error_handler(self, err: Exception) -> None:
        """*_error_handler()* to process the context (package/class), function name (method), and the error (exception) that was raised to format a detailed error message and traceback.

        Args:
            err (Exception): Python raised exception.
        """
        _context = self.__class__.__name__
        _function_name = inspect.currentframe().f_code.co_name
        _error(_context, _function_name, err)

    def __str__(self) -> str:
        """*__str__()* returns a string representation of the *DimensionalModel* object.

        Returns:
            str: String representation of the *DimensionalModel* object.
        """
        _attr_lt = []
        for attr, value in vars(self).items():
            # Skip private attributes starting with "__"
            # if attr.startswith("__"):
            #    continue
            # Format attribute name and value
            _attr_name = attr.lstrip("_")
            _attr_lt.append(f"{_attr_name}={repr(value)}")
        # Format the string representation of the ArrayList class and its attributes
        _str = f"{self.__class__.__name__}({', '.join(_attr_lt)})"
        return _str

    def __repr__(self) -> str:
        """*__repr__()* returns a string representation of the *DimensionalModel* object.

        Returns:
            str: String representation of the *DimensionalModel* object.
        """
        return self.__str__()


@dataclass
class DimensionalAnalyzer(DimensionalModel[T]):
    """**DimensionalAnalyzer** extends *DimensionalModel* to perform the dimensional analysis.

    This class provides methods to:
    - Sort parameters and FDUs for analysis.
    - Build and solve the dimensional matrix.
    - Generate dimensionless coefficients (e.g., Pi groups).

    Attributes:
        output (Optional[Parameter]): The output parameter of the analysis.
        _wrk_fdu_lt (List[str]): List of working Fundamental Dimensions (FDUs).
        _dim_mtx (Optional[np.ndarray]): Dimensional matrix.
        _dim_mtx_trans (Optional[np.ndarray]): Transposed dimensional matrix.

    # TODO complete docstring

    Args:
        Generic (T): Generic type for a Python data structure.
        DimensionalModel (DimensionalModel[T]): Inherits from the *DimensionalModel* class.        
    """

    # Private attributes with validation logic

    output: Optional[Parameter] = None
    _wrk_fdu_lt: List[str] = field(default_factory=list)
    _dim_mtx: Optional[np.ndarray] = None
    _dim_mtx_trans: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """__post_init__ _summary_
        """
        super().__post_init__()
        if self._validate_param_lt(self.param_lt):
            self.output = self._setup_output()
            self._wrk_fdu_lt = self._extract_fdu(self.relevance_lt)
            self._wrk_fdu_lt = self._sort_fdu(self._wrk_fdu_lt)
            self._fdu_ht = self._filter_fdu_map(self._wrk_fdu_lt)
            self._relevance_lt = self._sort_by_category(self.relevance_lt)

    def _setup_output(self) -> Optional[Parameter]:
        """*_setup_da_output()* Finds and sets the output parameter.

        Returns:
            Optional[Parameter]: The output parameter of the analysis. None if not found.
        """
        # use next for better performance and readability
        return next((p for p in self._param_lt if p.cat == "OUTPUT"), None)

    def _extract_fdu(self, relevance_lt: List[Parameter]) -> List[str]:
        """*_extract_fdu()* Extracts working FDUs from relevant parameters.

        Args:
            relevance_lt (List[Parameter]): List of relevant parameters.

        Returns:
            List[str]: List of working FDUs.
        """
        match = [p.std_dims for p in relevance_lt]
        fdus = [d for d in re.findall(cfg.WKNG_FDU_SYM_REGEX, str(match))]
        fdus = list({fdus[i] for i in range(len(fdus))})
        # return list({m for p in relevance_lt for m in re.findall(cfg.WKNG_FDU_SYM_REGEX, p.std_dims)})
        return fdus

    def _sort_fdu(self, fdu_lt: List[str]) -> List[str]:
        """*_sort_fdu()* Sorts FDUs based on framework precedence.

        Args:
            fdu_lt (List[str]): List of working FDUs.

        Returns:
            List[str]: Sorted list of working FDUs.
        """
        return sorted(fdu_lt, key=lambda x: cfg.WKNG_FDU_PREC_LT.index(x))

    def _filter_fdu_map(self, fdu_lt: List[str]) -> SCHashTable:
        """*_filter_fdu_map()* Filters the FDU hash table to include only working FDUs.

        Args:
            fdu_lt (List[str]): List of working FDUs.

        Returns:
            SCHashTable: Filtered FDU hash table.
        """
        new_fdu_ht = SCHashTable()
        for key in fdu_lt:
            if key in self._fdu_ht.keys():
                new_fdu_ht.insert(key, self._fdu_ht.delete(key))
        return new_fdu_ht

    def _sort_by_category(self, param_lt: List[Parameter]) -> List[Parameter]:
        """*_sort_by_category()* Sorts parameters by category based on precedence in `cfg` module.

        Args:
            param_lt (List[Parameter]): List of parameters.

        Returns:
            List[Parameter]: Sorted list of parameters.
        """
        _PAR_CAT = list(cfg.PARAMS_CAT_DT.keys())
        sorted_params = sorted(param_lt, key=lambda p: _PAR_CAT.index(p.cat))
        for i, param in enumerate(sorted_params):
            param.idx = i
        return sorted_params

    def _setup_matrix(self, n_fdu: int, n_relv: int) -> np.ndarray:
        """*_setup_matrix()* Creates a zero-filled dimensional matrix.

        Args:
            n_fdu (int): Number of active FDUs.
            n_relv (int): Number of relevant parameters.

        Returns:
            np.ndarray: Zero filled matrix with (n_fdu, n_relv) shape.
        """
        return np.zeros((n_fdu, n_relv), dtype=float)

    def _fill_matrix(self,
                     relevance_lt: List[Parameter],
                     mtx: np.ndarray) -> np.ndarray:
        """*_fill_matrix()* Fills the dimensional matrix with parameter exponents.

        Args:
            relevance_lt (List[Parameter]): List of relevant parameters.
            mtx (np.ndarray): Dimensional matrix.

        Returns:
            np.ndarray: Filled dimensional matrix.
        """
        for relv in relevance_lt:
            mtx[:, relv.idx] = np.array(relv.dim_col, dtype=float)
        return mtx

    def create_matrix(self) -> None:
        """*create_matrix()* uilds the dimensional matrix and its transpose.
        """
        self._dim_mtx = self._setup_matrix(self._n_in, self._n_relv)
        self._dim_mtx = self._fill_matrix(self.relevance_lt, self._dim_mtx)
        self._dim_mtx_trans = self._dim_mtx.T

    def _diagonalize_matrix(self,
                            mtx: np.ndarray) -> tuple[np.ndarray, List[int]]:
        """*_diagonalize_matrix()* Computes the Row-Reduced Echelon Form (RREF) of the matrix.

        Args:
            mtx (np.ndarray): Dimensional matrix.

        Returns:
            tuple[np.ndarray, List[int]]: RREF matrix and pivot columns.
        """
        # Convert the NumPy matrix to a SymPy Matrix
        self._sym_mtx = sp.Matrix(mtx)
        # Compute the Row-Reduced Echelon Form (RREF) and pivot columns
        rref_mtx, pivot_cols = self._sym_mtx.rref()
        # return the casted RREF matrix and the pivot columns
        return sp.matrix2numpy(rref_mtx, dtype=float), pivot_cols

    def generate_pi_coefficients(self,
                                 relevance_lt: List[Parameter],
                                 mtx: sp.Matrix) -> List[PiCoefficient]:
        """*generate_pi_coefficients()* Generates dimensionless Pi coefficients using the null space of the matrix.

        Args:
            relevance_lt (List[Parameter]): List of relevant parameters.
            mtx (sp.Matrix): Dimensional matrix.

        Returns:
            List[PiCoefficient]: List of Pi coefficients.
        """
        nullspace_vectors = mtx.nullspace()
        symbols = [param.sym for param in relevance_lt]
        coefficients = []
        for i, vector in enumerate(nullspace_vectors):
            vector_np = sp.matrix2numpy(vector, dtype=float).T
            # Extract symbols of the parameters involved in the coefficient
            piparam = [symbols[j] for j, c in enumerate(vector_np[0]) if c != 0]
            coefficients.append(PiCoefficient(
                _idx=i,
                _sym=f"\\Pi{{{i}}}",
                _fwk=self._fwk,
                _param_lt=symbols,
                _pivot_lt=self._pivot_cols,
                _dim_col=vector_np,
                name=f"Pi-Coefficient No. {i}",
                description=f"Pi-Group {i} computed with {', '.join(piparam)}"
            ))

        # OLD CODE VERSION
        # nullspace_vectors = mtx.nullspace()
        # symbol_lt = [param.sym for param in relevance_lt]
        # coefficients = [
        #     PiCoefficient(
        #         _idx=i,
        #         _sym=f"\\Pi{{{i}}}",
        #         _fwk=self._fwk,
        #         _param_lt=symbol_lt,
        #         _dim_col=sp.matrix2numpy(vector, dtype=float).T,
        #         _pivot_lt=self._pivot_cols,
        #         name=f"Pi-Coefficient No. {i}",
        #         description=f"Pi-Group {i} computed with {', '.join(symbol_lt)}"
        #     ) for i, vector in enumerate(nullspace_vectors)
        # ]
        return coefficients

    def solve_matrix(self) -> None:
        """*_solve_dim_matrix()* Solves the dimensional matrix and computes the Pi coefficients.
        """
        ans = self._diagonalize_matrix(self._dim_mtx)
        self._rref_mtx, self._pivot_cols = ans
        self._pi_coef_lt = self.generate_pi_coefficients(self.relevance_lt,
                                                         self._sym_mtx)

    @property
    def pi_coef_lt(self) -> List[PiCoefficient]:
        """*pi_coef_lt* property to get the list of Pi-Coefficients in the *DimensionalAnalyzer*.

        Returns:
            List[PiCoefficient]: List of Pi-Coefficients.
        """
        return self._pi_coef_lt

    @pi_coef_lt.setter
    def pi_coef_lt(self, value: List[PiCoefficient]) -> None:
        """*pi_coef_lt* property to set the list of Pi-Coefficients in the *DimensionalAnalyzer*. It must be a list of *PiCoefficient* objects.

        Args:
            value (List[PiCoefficient]): List of *PiCoefficient* objects.
        """
        if not all(isinstance(p, PiCoefficient) for p in value):
            _msg = "Coefficient list must be a list of PiCoefficient objects. "
            _msg += f"Provided: {value}"
            raise ValueError(_msg)
        self._pi_coef_lt = value
