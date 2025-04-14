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
from typing import Optional, List, Generic, Union
# import dataclass for class attributes and validations
from dataclasses import dataclass, field

# custom modules
# Dimensional Analysisis modules
from Src.PyDASA.Units.fdu import FDU
from Src.PyDASA.Units.params import Parameter
from Src.PyDASA.Pi.coef import PiCoefficient

# data structures modules
from Src.PyDASA.DataStructs.Tables.scht import SeparateChainingTable
from Src.PyDASA.DataStructs.Tables.htme import MapEntry

# FDU regex manager
from Src.PyDASA.Utils.cstm import RegexManager

# generic error handling and type checking
from Src.PyDASA.Utils.err import error_handler as error
from Src.PyDASA.Utils.dflt import T

# importing PyDASA's regex for managing FDUs
# using the 'as' allows shared variable edition
from Src.PyDASA.Utils import cfg as config

# Generalizing input type for the classes
FDUs = Union[FDU, dict, str]
Params = Union[Parameter, dict, str]

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
    _fdu_ht = field(default_factory=SeparateChainingTable[MapEntry[str, FDU]])
    """
    DFUs hash table. Custom hash table for storing FDUs. It is a dictionary-like structure that allows for fast lookups and insertions.
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
    # :attr: io_fdu_lt
    io_fdu_lt: Optional[List[FDUs]] = None
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
        # TODO complete docstring
        pass

    def _config_fdu_precedence(self, lt: List[dict]) -> None:
        """_config_fdu_precedence _summary_

        Args:
            lt (List[str]): _description_
        """
        # TODO complete docstring
        self._fdu_regex._fdu_prec_lt = lt

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




@dataclass
class DimensionalAnalyzer(DimensionalModel[T]):
    """DimensionalAnalyzer _summary_

    # TODO complete docstring

    Args:
        Generic (T): Generic type for a Python data structure.
    """
    pass
