# -*- coding: utf-8 -*-
"""
Module for representing the **FDU** or **Fundamental Dimensional Unit** for Dimensional Analysis in *PyDASA*.

*IMPORTANT:* Based on the theory from:

    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""

# native python modules
# import dataclass for defining the node class
from dataclasses import dataclass
from typing import Generic
import re

# custom modules
# generic error handling and type checking
from Src.PyDASA.Utils.err import error_handler as error
from Src.PyDASA.Utils.dflt import T

# import global variables
from Src.PyDASA.Utils.cfg import FDU_FWK_DT, LATEX_REGEX

# checking custom modules
assert error
assert T


@dataclass
class FDU(Generic[T]):
    """**FDU** class for processing the data of a **Fundamental Dimensional Unit** in *PyDASA*.

    FDUs are fundamental building blocks of dimensional analysis and are used to define the dimensions of physical or digital quantities.

    Args:
        Generic (T): Generic type for a Python data structure.

    Returns:
        FDU: A FDU object with the following attributes:
            - _idx (int): The Index of the FDU.
            - _sym (str): The symbol of the FDU.
            - _fwk (str): The framework of the FDU. It can be one supported frameworks.
            - name (str): User-friendly name of the FDU.
            - description (str): Brief summary of the FDU.
    """

    # Private attributes with validation logic
    # :attr: _idx
    _idx: int = -1
    """
    Unique identifier/index of the *FDU*. It is  the order of precedence in the rows of the Dimensional Matrix.
    """

    # :attr: _sym
    _sym: str = ""
    """
    Symbol of the *FDU*. It is a LaTeX or an alphanumeric string (preferably a single Latin or Greek letter). It is used for user-friendly representation of the instance.
    """

    # :attr: _fwk
    _fwk: str = "PHYSICAL"
    """
    Framework of the *FDU* in the Dimensional Matrix. Can be: `PHYSICAL`, `COMPUTATION`, `DIGITAL` or `CUSTOM`.
    """

    # Public attributes
    # :attr: name
    name: str = ""
    """
    User-friendly name of the FDU.
    """

    # :attr: description
    description: str = ""
    """
    Small summary of the FDU.
    """

    @property
    def idx(self) -> int:
        """*idx* Get the FDU precedence in the Dimensional Matrix.

        Returns:
            int: Precedence of the FDU.
        """
        return self._idx

    @idx.setter
    def idx(self, val: int) -> None:
        """*idx* Set the FDU precedence in the Dimensional Matrix. It must be a non-negative integer.

        Args:
            val (int): Precedence of the FDU.

        Raises:
            ValueError: If the Index is not a non-negative integer.
        """
        if not isinstance(val, int) or val < 0:
            _msg = "Precedence must be a non-negative integer. "
            _msg += f"Provided: {val}"
            raise ValueError(_msg)
        self._idx = val

    @property
    def sym(self) -> str:
        """*sym* Get the FDU symbol.

        Returns:
            str: Symbol of the FDU.
        """
        return self._sym

    @sym.setter
    def sym(self, val: str) -> None:
        """*sym* Set the FDU symbol. It must be a single alphanumeric character.

        Args:
            val (str): Symbol of the FDU.

        Raises:
            ValueError: If the symbol is not alphanumeric.
        """

        # Regular expression to match valid LaTeX strings or alphanumeric strings
        if not (val.isalnum() or re.match(LATEX_REGEX, val)) or len(val) != 1:
            _msg = "Symbol must be alphanumeric or a valid LaTeX string. "
            _msg += f"Provided: '{val}' "
            _msg += "Examples: 'V', 'd', '\\Pi_{0}', '\\rho'."
            raise ValueError(_msg)
        self._sym = val

    @property
    def fwk(self) -> str:
        """*fwk* Get the FDU framework.

        Returns:
            str: Working framework of the FDU.
        """
        return self._fwk

    @fwk.setter
    def fwk(self, val: str) -> None:
        """*fwk* Set the FDU framework. It must be one of the allowed values. The allowed values are: `PHYSICAL`, `COMPUTATION`, `DIGITAL` or `CUSTOM`.

        Args:
            val (str): Framework of the FDU.

        Raises:
            ValueError: If the framework is not one of the allowed values.
        """
        if val not in FDU_FWK_DT.keys():
            _msg = f"Invalid framework: {val}. "
            _msg += "Framework must be one of the following: "
            _msg += f"{', '.join(FDU_FWK_DT.keys())}."
            raise ValueError(_msg)
        self._fwk = val

    def __str__(self) -> str:
        """*__str__()* returns a string representation of the FDU object.

        Returns:
            str: String representation of the FDU object.
        """
        _attr_lt = []
        for attr, val in vars(self).items():
            # Skip private attributes starting with "__"
            if attr.startswith("__"):
                continue
            # Format attribute name and value
            _attr_name = attr.lstrip("_")
            _attr_lt.append(f"{_attr_name}={repr(val)}")
        # Format the string representation of the ArrayList class and its attributes
        _str = f"{self.__class__.__name__}({', '.join(_attr_lt)})"
        return _str

    def __repr__(self) -> str:
        """*__repr__()* returns a string representation of the FDU object.

        Returns:
            str: String representation of the FDU object.
        """
        return self.__str__()
