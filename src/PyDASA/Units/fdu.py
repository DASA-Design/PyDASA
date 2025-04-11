# -*- coding: utf-8 -*-
"""
Module to represent the **FDU** or **Fundamental Dimensional Unit** data structure for Dimensional Analysis in *PyDASA*.

*IMPORTANT:* This code and its specifications for Python are based on the theory and subject developed by the following authors/books:

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
    """**FDU** class for creating a **Fundamental Dimensional Unit** in *PyDASA*. Fundamental for the process of Dimensional Analysis and creating Dimensionless Coefficients.

    Args:
        Generic (T): Generic type for a Python data structure.

    Returns:
        FDU: A FDU object with the following attributes:
            - `_prec`: The ID of the FDU.
            - `_symbol`: The symbol of the FDU.
            - `_framework`: The framework of the FDU. It can be one of the following: `PHYSICAL`, `DIGITAL`, or `CUSTOM`.
            - `name`: The name of the FDU.
            - `description`: The description of the FDU.
    """

    # Private attributes with validation logic
    # Unique FDU symbol in the system
    # :attr: _symbol
    _symbol: str = ""
    """
    Unique FDU's symbol. It must be a single alphanumeric character (preferably a single Latin or Greek letter).
    """

    # :attr: _framework
    _framework: str = "PHYSICAL"
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
    def symbol(self) -> str:
        """*symbol* property to get the symbol of the FDU.

        Returns:
            str: Symbol of the FDU.
        """
        return self._symbol

    @symbol.setter
    def symbol(self, value: str) -> None:
        """*symbol* property to set the symbol of the FDU. It must be alphanumeric (preferably a single character + Latin or Greek letter).

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
        self._symbol = value

    @property
    def precedence(self) -> int:
        """*precedence* property to get the row order of the FDU in the dimensional matrix.

        Returns:
            int: Precedence of the FDU.
        """
        return self._prec

    @precedence.setter
    def precedence(self, value: int) -> None:
        """*precedence* property to order the FDU in the rows of the dimensional matrix.

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
        return self._framework

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
        self._framework = value

    def __str__(self) -> str:
        """*__str__* returns a string representation of the FDU object.

        Returns:
            str: String representation of the FDU object.
        """
        _str = f"{self.__class__.__name__}("
        _str += f"symbol='{self._symbol}', "
        _str += f"precedence='{self._prec}', "
        _str += f"framework='{self._framework}', "
        _str += f"name='{self.name}', "
        _str += f"description='{self.description}')"
        return _str
