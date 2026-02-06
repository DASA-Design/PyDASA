# -*- coding: utf-8 -*-
"""
Module basic.py
===========================================

This module provides base classes with common validation logic used across different classes to minimize code duplication in **PyDASA**.

**IMPORTANT**
    - Based on the theory from H. Görtler, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""
# native python modules
# forward references + postpone eval type hints
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import inspect

# indicate it is an abstract base class
from abc import ABC

# import global variables
from pydasa.core.setup import Frameworks
from pydasa.core.setup import PYDASA_CFG
from pydasa.validations.patterns import LATEX_RE

# import validation decorators
from pydasa.validations.decorators import validate_type
from pydasa.validations.decorators import validate_emptiness
from pydasa.validations.decorators import validate_choices
from pydasa.validations.decorators import validate_pattern
from pydasa.validations.decorators import validate_index


@dataclass
class SymBasis(ABC):
    """Abstract Class to manage the entity's symbolic, sorting, and dimensional domain/framework functionalities in **PyDASA**.

    Inherits from:
        - **ABC**: Python Abstract Base Class to indicate that this class is not meant to be instantiated directly.
    """

    # :attr: _sym
    _sym: str = ""
    """Symbol representation (LaTeX or alphanumeric)."""

    # :attr: _fwk
    _fwk: str = Frameworks.PHYSICAL.value
    """Frameworks context (PHYSICAL, COMPUTATION, SOFTWARE, CUSTOM)."""

    # :attr: _alias
    _alias: str = ""
    """Python-compatible alias for symbol, used in executable code. e.g.: `\\rho_{1}` -> `rho_1`."""

    def __post_init__(self) -> None:
        """Post initialization processing and validation for symbolic attributes.
        """
        # super().__post_init__()
        # Validate the symbol and framework
        if not self._sym:
            self._sym = self._sym.strip()
        if not self._fwk:
            self._fwk = self._fwk.strip()
        if not self._alias:
            self._alias = self._alias.strip()

    @property
    def sym(self) -> str:
        """Get the entity's symbol.

        Returns:
            str: Symbol value.
        """
        return self._sym

    @sym.setter
    @validate_type(str)
    @validate_emptiness()
    @validate_pattern(LATEX_RE, allow_alnum=True)
    def sym(self, val: str) -> None:
        """Set the entity's symbol with validation.

        Args:
            val (str): Symbol value.

        Raises:
            ValueError: If symbol format is invalid.
        """
        self._sym = val

    @property
    def fwk(self) -> str:
        """Get the entity's framework.

        Returns:
            str: Frameworks value.
        """
        return self._fwk

    @fwk.setter
    @validate_type(str)
    @validate_choices(PYDASA_CFG.frameworks)
    def fwk(self, val: str) -> None:
        """Set the entity's framework with validation.

        Args:
            val (str): Frameworks value.

        Raises:
            ValueError: If framework is not supported.
        """
        self._fwk = val

    @property
    def alias(self) -> Optional[str]:
        """Get the Python variable synonym.

        Returns:
            Optional[str]: Python variable name. e.g.: `\\rho_{1}` -> `rho_1`.
        """
        return self._alias

    @alias.setter
    @validate_type(str)
    @validate_emptiness()
    def alias(self, val: str) -> None:
        """Set the Python variable synonym.

        Args:
            val (str): Python variable name. e.g.: `\\rho_{1}` -> `rho_1`.

        Raises:
            ValueError: If variable name is empty.
        """
        self._alias = val

    def clear(self) -> None:
        """Reset symbol, alias, and framework attributes to default values.
        """
        # Reset symbol attributes
        self._sym = ""
        self._alias = ""
        self._fwk = Frameworks.PHYSICAL.value


@dataclass
class IdxBasis(SymBasis):
    """Basic class to manage index/precedence functionalities in **PyDASA** entities.

    Inherits from:
        - **SymBasis**: Inherits symbol and framework validation.
    """

    # :attr: _idx
    _idx: int = -1
    """Unique identifier/index for ordering in dimensional matrix."""

    def __post_init__(self) -> None:
        """Post initialization processing and validation for index and precedence.
        """
        super().__post_init__()
        if self._idx != -1:
            self.idx = self._idx

    @property
    def idx(self) -> int:
        """Get the index/precedence value.

        Returns:
            int: Index value.
        """
        return self._idx

    @idx.setter
    @validate_type(int, allow_none=False)
    @validate_index()
    def idx(self, val: int) -> None:
        """Set the index/precedence value.

        Args:
            val (int): Index value (must be non-negative).

        Raises:
            ValueError: If index is not a non-negative integer.
        """
        self._idx = val

    def clear(self) -> None:
        """Reset index and inherited symbol attributes to default values.
        """
        # Reset parent class attributes
        super().clear()

        # Reset index attribute
        self._idx = -1


@dataclass
class Foundation(IdxBasis):
    """Basic class to manage common attributes and validation logic for dimensional analysis entities in **PyDASA**.

    Provides common validation logic and attributes shared by dimensional framework/domain, Variable, and Coeffcient classes.

    Inherits from:
        - **IdxBasis**: Inherits index, symbol, and framework validation.
    """

    # :attr: _name
    _name: str = ""
    """User-friendly name of the entity."""

    # :attr: description
    description: str = ""
    """Brief summary or description of the entity."""

    def __post_init__(self) -> None:
        """Post initialization and processing for name and description attributes.
        """
        if self.description:
            self.description = self.description.strip()

    @property
    def name(self) -> str:
        """Get the entity's name.

        Returns:
            str: Name value.
        """
        return self._name

    @name.setter
    @validate_type(str, allow_none=False)
    def name(self, val: str) -> None:
        """Set the entity's name with validation.

        Args:
            val (str): Name value.

        Raises:
            ValueError: If name is not a non-empty string.
        """
        self._name = val.strip()

    def clear(self) -> None:
        """Reset name, description, and inherited attributes to default values.
        """
        # Reset parent class attributes
        super().clear()

        # Reset name and description attributes
        self.name = ""
        self.description = ""

    def __str__(self) -> str:
        """Return string representation with all non-private attributes for detailed inspection and logging.

        Returns:
            str: Detailed string representation.
        """
        _attr_lt = []
        for attr, value in vars(self).items():
            # Skip private attributes starting with "__"
            if attr.startswith("__"):
                continue
            # Format callable attributes
            if callable(value):
                value = f"{value.__name__}{inspect.signature(value)}"
            # Format attribute name and value
            _attr_name = attr.lstrip("_")
            _attr_lt.append(f"{_attr_name}={repr(value)}")
        # Format the string with the class name and the attributes
        _str = f"{self.__class__.__name__}({', '.join(_attr_lt)})"
        return _str

    def __repr__(self) -> str:
        """Return the entity's str representation.

        Returns:
            str: String representation.
        """
        return self.__str__()
