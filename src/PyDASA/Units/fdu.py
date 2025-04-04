# -*- coding: utf-8 -*-
"""
Module to represent the **FDU** or **Fundamental Dimensional Unit** data structure for Dimensional Analysis in *PyDASA*.

*IMPORTANT:* This code and its specifications for Python are based on the theory and subject developed by the following authors/books:

    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""

# native python modules
# import dataclass for defining the node class
from dataclasses import dataclass, field
# import modules for defining the MapEntry type
from typing import Generic

# custom modules
# generic error handling and type checking
from Src.PyDASA.Utils.err import error_handler as error
from Src.PyDASA.Utils.dflt import T

# checking custom modules
assert error
assert T

# Set of supported Fundamental Dimensional Units (FDU)
# :data: FDU_FWK_DT
FDU_FWK_DT = {
    "PHYSICAL": "Traditional physical units",
    "DIGITAL": "Software Architecture units",
    "CUSTOM": "Custom units",
}
"""
Dictionary with the supported Fundamental Dimensional Units (FDU) in *PyDASA*.
"""


@dataclass
class FDU(Generic[T]):
    """**FDU** class for creating a **Fundamental Dimensional Unit** in *PyDASA*. Fundamental for the process of Dimensional Analysis and creating Dimensionless Coefficients.

    Args:
        Generic (T): Generic type for a Python data structure.

    Returns:
        FDU: A FDU object with the following attributes:
            - `_id`: The ID of the FDU.
            - `_symbol`: The symbol of the FDU.
            - `_framework`: The framework of the FDU. It can be one of the following: `PHYSICAL`, `DIGITAL`, or `CUSTOM`.
            - `name`: The name of the FDU.
            - `description`: The description of the FDU.
    """
    # Private attributes with validation logic
    # :attr: _id
    _id: str = field(init=False, repr=False)
    """
    ID of the FDU. It must be alphanumeric. Useful for identifying the FDU in the system and dimensional matrix construction.
    """

    # :attr: _symbol
    _symbol: str = field(init=False, repr=False)
    """
    Symbol of the FDU. It must be alphanumeric (preferably a single character + Latin or Greek letter). Useful for user-friendly representation of the FDU.
    """

    # :attr: _framework
    _framework: str = field(init=False, repr=False)
    """
    Framework of the FDU. It can be one of the following: `PHYSICAL`, `DIGITAL`, or `CUSTOM`. Useful for identifying the framework of the FDU.
    """

    # Public attributes
    # :attr: name
    name: str
    """
    Name of the FDU. User-friendly name of the FDU.
    """
    # :attr: description
    description: str
    """
    Description of the FDU. It is a string with a small summary of the FDU.
    """

    @property
    def id(self) -> str:
        """*id* property to get the ID of the FDU.

        Returns:
            str: ID of the FDU.
        """
        return self._id

    @id.setter
    def id(self, value: str) -> None:
        """*id* property to set the ID of the FDU. It must be alphanumeric.

        Args:
            value (str): ID of the FDU.

        Raises:
            ValueError: If the ID is not alphanumeric.
        """
        if not value.isalnum():
            raise ValueError("ID must be alphanumeric.")
        self._id = value

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
        if not value.isalnum():
            _msg = "Symbol must be alphanumeric. "
            _msg += f"Provided: {value}"
            _msg += "Preferably a Latin or Greek letter."
            raise ValueError(_msg)
        self._symbol = value

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
        _fdu = f"FDU(id={self._id}, "
        _fdu += f"symbol={self._symbol}, "
        _fdu += f"framework={self._framework}, "
        _fdu += f"name={self.name}, "
        _fdu += f"description={self.description})"
        return _fdu
