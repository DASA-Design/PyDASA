# -*- coding: utf-8 -*-
"""
Module for solving the Dimensional Model in *PyDASA*. Defines the *DimensionalModel* for managing dimensional data and the *DimensionalAnalyzer* for performing dimensional analysis.

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
# import re
# import modules for defining Dimensionless Coefficients (DN) type
from typing import List, Tuple
from typing import Optional, Generic, Union
# import dataclass for class attributes and validations
from dataclasses import dataclass, field
import inspect

# custom modules
# Dimensional Analysisis modules
from Src.PyDASA.Units.fdu import FDU
from Src.PyDASA.Units.params import Parameter
from Src.PyDASA.Pi.coef import PiCoefficient

# data structures modules
from Src.PyDASA.DataStructs.Tables.scht import SCHashTable
from Src.PyDASA.DataStructs.Tables.htme import MapEntry

# FDU regex manager
from Src.PyDASA.Utils.cstm import RegexManager

# generic error handling and type checking
from Src.PyDASA.Utils.err import error_handler as error
from Src.PyDASA.Utils.dflt import T

# importing PyDASA's regex for managing FDUs
# using the 'as' allows shared variable edition
from Src.PyDASA.Utils import cfg as config

# Generalizing input type for class typing
FDUElm = Union[FDU, dict, str]
Params = Union[Parameter, dict, str]
FDUEnt = MapEntry[Tuple[str, FDU]]

# checking custom modules
assert error
assert config
assert T


@dataclass
class DimensionalModel(Generic[T]):
    """*DimensionalModel* class creates *Dimensional Model* in *PyDASA*. Dimensional Models are used to define the dimensions of physical or digital quantities, the parameters, and the relevance list of the system

    # TODO complete docstring

    Args:
        Generic (T): Generic type for a Python data structure.
    """
    # Private attributes with validation logic
    # :attr: _idx
    _idx: int = -1
    """
    Index of the *DimensionalModel*. It is the unique integer for the column's order in the dimensional matrix.
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
    Framework the *DimensionalModel* follows in accordance with the FDU. It can be one of the following: `PHYSICAL`, `COMPUTATION`, `DIGITAL` or `CUSTOM`. Parameters and FDUs must be in the same framework.
    """

    # FDU regex manager
    # :attr: _fdu_regex
    _fdu_regex: RegexManager = field(default_factory=lambda: RegexManager)
    """
    FDU regex manager. It is used to validate and configure the FDU's symbols, working regex, and precedence list.
    """

    # FDUs hash table
    # :attr: _fdu_ht
    _fdu_ht: SCHashTable[FDUEnt] = field(default_factory=SCHashTable[FDUEnt])
    """
    FDU hash table. Custom hash table for storing FDUs. It is a dictionary-like structure that allows for fast lookups and insertions.
    """

    # List of parameters
    # :attr: _param_lt
    _param_lt: List[Parameter] = field(default_factory=list[Parameter])
    """
    List of *Parameter* objects. It is used to store the parameters of the system and is the base for the relevance list.
    """

    # List of relevant parameters
    # :attr: _relevance_lt
    _relevance_lt: List[Parameter] = field(default_factory=list[Parameter])
    """
    List of relevant *Parameter* objects. It is used to store the parameters use to solve the Dimensional Matrix. It is a subset of the `_param_lt` list with the 'relevance' attribute set to `True`.
    """

    # Public attributes
    # :attr: name
    name: str = ""
    """
    Name of the *DimensionalModel*. User-friendly name of the dimensional model.
    """

    # :attr: description
    description: str = ""
    """
    Description of the *DimensionalModel*. It is a small summary of the dimensional model.
    """

    # list of user defined FDUs
    # :attr: io_fdu
    io_fdu: Optional[List[dict]] = None
    """
    Optional list of user-defined FDUs (Fundamental Dimensions). This list stores FDUs as dictionaries or strings, including their symbols and frameworks.

    Purpose:
        - Used to populate the `_fdu_ht` hash table.
        - If empty, the `_fdu_ht` hash table will use the default FDUs from the `config` module.
        - If not empty, it will use the user-defined FDUs.
    """

    def __post_init__(self) -> None:
        """__post_init__ _summary_
        """
        # TODO implement this method
        # set up the hash table for the FDUs
        self._setup_fdu_ht()
        # set up the FDU precedence list
        self._setup_fdu_precedence()
        # set up the FDU regex manager
        self._setup_fdu_regex()

        # TODO maybe I dont need the if, because of hwo RegexManager is implemented

    def _setup_fdu_ht(self) -> None:
        """*_setup_fdu_ht()* sets up the hash table for the FDUs. It uses the `_fwk` attribute to determine the framework of the FDUs.

        Raises:
            ValueError: error if the framework is invalid.
        """
        if self._fwk in config.FDU_FWK_DT and self._fwk != "CUSTOM":
            self._configure_default_fdus()
        elif self._fwk == "CUSTOM" and self.io_fdu:
            self._configure_custom_fdus()
        else:
            _msg = f"Invalid framework: {self._fwk}. "
            _msg += "Must be one of the following: "
            _msg += f"{', '.join(config.FDU_FWK_DT.keys())}."
            raise ValueError(_msg)

    def _configure_default_fdus(self) -> None:
        """*_configure_default_fdus()* configures the default FDUs in the hash table. It uses the `_fwk` attribute to determine the framework of the FDUs.

        Raises:
            ValueError: error if the framework is invalid.
        """
        # map for easy access to the FDUs
        _frk_dt = {
            "PHYSICAL": config.PHY_FDU_PREC_DT,
            "COMPUTATION": config.COMPU_FDU_PREC_DT,
            "DIGITAL": config.DIGI_FDU_PREC_DT,
        }
        # get the framework dictionary
        _cfg_dt = _frk_dt.get(self._fwk, {})
        # if the framework is not valid, raise an error
        if not _cfg_dt:
            _msg = f"Invalid framework: {self._fwk}. "
            _msg += "Framework must be one of the following: "
            _msg += f"{', '.join(config.FDU_FWK_DT.keys())}."
            raise ValueError(_msg)
        # otherwise, configure the FDUs
        for i, (_sym, _desc) in enumerate(_cfg_dt.items()):
            t_fdu = FDU(_sym=_sym,
                        _fwk=self._fwk,
                        _idx=i,
                        name=_desc,
                        description=_desc)
            self._fdu_ht.insert(t_fdu.sym, t_fdu)
            i += 1

    def _configure_custom_fdus(self) -> None:
        """*configure_custom_fdus()* configures the custom FDUs in the hash table. It uses the `io_fdu` attribute to determine the framework of the FDUs.
        """
        # TODO maybe I can improve this for more genericity
        for i, t_fdu in enumerate(self.io_fdu):
            # standarize all frameworks to be the same
            t_fdu["_fwk"] = self._fwk
            t_fdu["_idx"] = i
            fdu = FDU(**t_fdu)
            self._fdu_ht.insert(fdu.sym, fdu)
            i += 1

    def _setup_fdu_precedence(self) -> None:
        """_setup_fdu_precedence _summary_

        Raises:
            ValueError: _description_
        """
        # TODO maybe I can improve this for more genericity
        if self._fdu_ht.empty:
            _msg = "FDU hash table is empty. "
            _msg += f"Current Size: {self._fdu_ht.size()}."
            _msg += "Please configure the FDU hash table before setting up the precedence list."
            raise ValueError(_msg)
        _fdu_keys = self._fdu_ht.keys()
        _fdu_prec_lt = []
        for fdu in _fdu_keys:
            _fdu_prec_lt.append(fdu)
        config.DFLT_FDU_PREC_LT = _fdu_prec_lt

    def _setup_fdu_regex(self) -> None:
        """_setup_fdu_regex _summary_
        """
        _PREC_LT = config.DFLT_FDU_PREC_LT.copy()
        self._fdu_regex = RegexManager(_fwk=self._fwk,
                                       _fdu_prec_lt=_PREC_LT)
        self._fdu_regex.update_global_regex()

    @property
    def idx(self) -> int:
        """*idx* property gets the *DimensionalModel* index in the program.

        Returns:
            int: ID of the *DimensionalModel*.
        """
        return self._idx

    @idx.setter
    def idx(self, value: str) -> None:
        """*idx* property to set the *DimensionalModel* index in the dimensional matrix. It must be alphanumeric.

        Args:
            value (str): ID of the *DimensionalModel*.

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
        """*sym* property to get the symbol of the *DimensionalModel*. It must be alphanumeric (preferably a single character, a Latin or Greek letter).

        Returns:
            str: Symbol of the *DimensionalModel*. i.e.: V, d, D, m, Q, \\rho, etc.
        """
        return self._sym

    @sym.setter
    def sym(self, value: str) -> None:
        """*sym* property to set the symbol of *DimensionalModel*. It must be alphanumeric (preferably a single character, a Latin or Greek letter).

        Args:
            value (str): Symbol of the *DimensionalModel*. . i.e.: V, d, D, m, Q, \\rho, etc.

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
        """*fwk* property to get the framework of the *DimensionalModel*.

        Returns:
            str: Framework of the *DimensionalModel*. It can be one of the following: `PHYSICAL`, `COMPUTATION`, `DIGITAL` or `CUSTOM`.
        """
        return self._fwk

    @fwk.setter
    def fwk(self, value: str) -> None:
        """*fwk* property of the framework of the *DimensionalModel*. It must be one of the following: `PHYSICAL`, `COMPUTATION`, `DIGITAL` or `CUSTOM`.

        Args:
            value (str): Framework of the *DimensionalModel*. Must be the same as the FDU framework.

        Raises:
            ValueError: If the framework is not one of the allowed values.
        """
        if value not in config.FDU_FWK_DT.keys():
            _msg = f"Invalid framework: {value}. "
            _msg += "Framework must be one of the following: "
            _msg += f"{', '.join(config.FDU_FWK_DT.keys())}."
            raise ValueError(_msg)
        self._fwk = value

    def _error_handler(self, err: Exception) -> None:
        """*_error_handler()* to process the context (package/class), function name (method), and the error (exception) that was raised to format a detailed error message and traceback.

        Args:
            err (Exception): Python raised exception.
        """
        _context = self.__class__.__name__
        _function_name = inspect.currentframe().f_code.co_name
        error(_context, _function_name, err)

    def __str__(self) -> str:
        """*__str__()* returns a string representation of the *DimensionalModel* object.

        Returns:
            str: String representation of the *DimensionalModel* object.
        """
        _attr_lt = []
        for attr, value in vars(self).items():
            # Skip private attributes starting with "__"
            if attr.startswith("__"):
                continue
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
    """DimensionalAnalyzer _summary_

    # TODO complete docstring

    Args:
        Generic (T): Generic type for a Python data structure.
    """
    pass
