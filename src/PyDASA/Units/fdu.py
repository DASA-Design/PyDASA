# -*- coding: utf-8 -*-
"""
Module for representing the **FDU** or **Fundamental Dimensional Unit** for Dimensional Analysis in *PyDASA*.

*IMPORTANT:* Based on the theory from:

    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""

# native python modules
# import dataclass for defining the node class
from dataclasses import dataclass
# import modules for defining the MapEntry type
from typing import Generic

# custom modules
# generic error handling and type checking
from Src.PyDASA.Utils.err import error_handler as error
from Src.PyDASA.Utils.dflt import T

# import global variables
from Src.PyDASA.Utils.cfg import FDU_FWK_DT

# checking custom modules
assert error
assert T


@dataclass
class FDU(Generic[T]):
    """**FDU** class for creating a **Fundamental Dimensional Unit** in *PyDASA*. FDUs are the basic building blocks of dimensional analysis and are used to define the dimensions of physical and digital quantities.

    Args:
        Generic (T): Generic type for a Python data structure.

    Returns:
        FDU: A FDU object with the following attributes:
            - `_prec`: The ID of the FDU.
            - `_sym`: The symbol of the FDU.
            - `_fwk`: The framework of the FDU. It can be one of the following: `PHYSICAL`, `DIGITAL`, or `CUSTOM`.
            - `name`: The name of the FDU.
            - `description`: The description of the FDU.
    """

    # Private attributes with validation logic
    # Unique FDU symbol in the system
    # :attr: _sym
    _sym: str = ""
    """
    Unique FDU's symbol. It must be a single alphanumeric character (preferably a single Latin or Greek letter).
    """

    # :attr: _fwk
    _fwk: str = "PHYSICAL"
    """
    Framework of the FDU. It can be one of the following: `PHYSICAL`, `DIGITAL`, or `CUSTOM`. Useful for identifying the framework of the FDU.
    """

    # Precedence of the FDU in the dimensional matrix
    # :attr: _prec
    _prec: int = -1
    """
    ID of the FDU. It must be a unique alphanumeric. Useful for identifying the FDU in the system and dimensional matrix construction.
    """

    # Public attributes
    # Name of the FDU
    # :attr: name
    name: str = ""
    """
    Name of the FDU. User-friendly name of the FDU.
    """

    # Description of the FDU
    # :attr: description
    description: str = ""
    """
    Description of the FDU. It is a string with a small summary of the FDU.
    """

    @property
    def sym(self) -> str:
        """*sym* property to get the symbol of the FDU.

        Returns:
            str: Symbol of the FDU.
        """
        return self._sym

    @sym.setter
    def sym(self, value: str) -> None:
        """*sym* property to set the symbol of the FDU. It must be alphanumeric (preferably a single character + Latin or Greek letter).

        Args:
            value (str): Symbol of the FDU.

        Raises:
            ValueError: If the symbol is not alphanumeric.
        """
        if not value.isalnum() or len(value) != 1:
            _msg = "Symbol must be a single alphanumeric character. "
            _msg += f"Provided: {value}"
            _msg += "Preferably a Latin or Greek letter."
            raise ValueError(_msg)
        self._sym = value

    @property
    def prec(self) -> int:
        """*prec* property to get the row order of the FDU in the dimensional matrix.

        Returns:
            int: Precedence of the FDU.
        """
        return self._prec

    @prec.setter
    def prec(self, value: int) -> None:
        """*prec* property to order the FDU in the rows of the dimensional matrix.

        Args:
            value (int): Precedence of the FDU. Must be a non-negative integer.

        Raises:
            ValueError: If the ID is not alphanumeric.
        """
        if not isinstance(value, int) or value < 0:
            _msg = "Precedence must be a non-negative integer. "
            _msg += f"Provided: {value}"
            raise ValueError(_msg)
        if value != int(value):
            _msg = "Precedence must be an integer. "
            _msg += f"Provided: {value}"
            raise ValueError(_msg)
        self._prec = value

    @property
    def framework(self) -> str:
        """*framework* property to get the framework of the FDU.

        Returns:
            str: Framework of the FDU. It can be one of the following: `PHYSICAL`, `DIGITAL`, or `CUSTOM`.
        """
        return self._fwk

    @framework.setter
    def framework(self, value: str) -> None:
        """*framework* property of the allowed framework of the FDU.

        Args:
            value (str): Framework of the FDU. It can be one of the following: `PHYSICAL`, `DIGITAL`, or `CUSTOM`.

        Raises:
            ValueError: If the framework is not one of the allowed values.
        """
        if value not in FDU_FWK_DT.keys():
            _msg = f"Invalid framework: {value}. "
            _msg += "Framework must be one of the following: "
            _msg += f"{', '.join(FDU_FWK_DT.keys())}."
            raise ValueError(_msg)
        self._fwk = value

    def __str__(self) -> str:
        """*__str__()* returns a string representation of the FDU object.

        Returns:
            str: String representation of the FDU object.
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
        """*__repr__()* returns a string representation of the FDU object.

        Returns:
            str: String representation of the FDU object.
        """
        return self.__str__()
