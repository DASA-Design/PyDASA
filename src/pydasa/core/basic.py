# -*- coding: utf-8 -*-
"""
Module basic.py
===========================================

This module provides base classes with common validation logic used across FDU, Parameter, and Variable classes to eliminate code duplication.

Classes:
    **Foundation**: enforces common validation logic.
    **IdxBasis**: enforces index/precedence validation logic.
    **SymBasis**: enforces symbol and framework validation logic.

*IMPORTANT:* Based on the theory from:

    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""
# native python modules
# forward references + postpone eval type hints
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

# indicate it is an abstract base class
from abc import ABC

# import global variables
from pydasa.core.setup import Framework
from pydasa.core.setup import PYDASA_CFG
from pydasa.utils.patterns import LATEX_RE

# import validation decorators
from pydasa.validations.decorators import (
    validate_type,
    validate_emptiness,
    validate_choices,
    validate_pattern,
    validate_index,
)


@dataclass
class SymBasis(ABC):
    """**SymBasis** Foundation class for entities with symbols and framework functionalities.

    Attributes:
        _sym (str): Symbol representation.
        _alias (str): Python-compatible alias for use in code.
        _fwk (str): Framework context.
    """

    # :attr: _sym
    _sym: str = ""
    """Symbol representation (LaTeX or alphanumeric)."""

    # :attr: _fwk
    _fwk: str = Framework.PHYSICAL.value
    """Framework context (PHYSICAL, COMPUTATION, SOFTWARE, CUSTOM)."""

    # :attr: _alias
    _alias: str = ""
    """Python-compatible alias for symbol, used in executable code. e.g.: `\\rho_{1}` -> `rho_1`."""

    def __post_init__(self) -> None:
        """*__post_init__()* Post-initialization processing with symbol and framework validation.
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
        """*sym* Get the symbol.

        Returns:
            str: Symbol value.
        """
        return self._sym

    @sym.setter
    @validate_type(str)
    @validate_emptiness()
    @validate_pattern(LATEX_RE, allow_alnum=True)
    def sym(self, val: str) -> None:
        """*sym* Set the symbol with validation.

        Args:
            val (str): Symbol value.

        Raises:
            ValueError: If symbol format is invalid.
        """
        self._sym = val

    @property
    def fwk(self) -> str:
        """*fwk* Get the framework.

        Returns:
            str: Framework value.
        """
        return self._fwk

    @fwk.setter
    @validate_type(str)
    @validate_choices(PYDASA_CFG.frameworks)
    def fwk(self, val: str) -> None:
        """*fwk* Set the framework with validation.

        Args:
            val (str): Framework value.

        Raises:
            ValueError: If framework is not supported.
        """
        self._fwk = val

    @property
    def alias(self) -> Optional[str]:
        """*alias* Get the Python variable synonym.

        Returns:
            Optional[str]: Python variable name. e.g.: `\\rho_{1}` -> `rho_1`.
        """
        return self._alias

    @alias.setter
    @validate_type(str)
    @validate_emptiness()
    def alias(self, val: str) -> None:
        """*alias* Set the Python variable synonym.

        Args:
            val (str): Python variable name. e.g.: `\\rho_{1}` -> `rho_1`.

        Raises:
            ValueError: If variable name is empty.
        """
        self._alias = val

    def clear(self) -> None:
        """*clear()* Reset symbol and framework attributes to default values.

        Resets the entity's symbol-related properties to their initial state.
        """
        # Reset symbol attributes
        self._sym = ""
        self._alias = ""
        self._fwk = Framework.PHYSICAL.value


@dataclass
class IdxBasis(SymBasis):
    """**IdxBasis** Foundation class for entities with index/precedence functionality.

    Attributes:
        _idx (int): Index/precedence value
    """

    # :attr: _idx
    _idx: int = -1
    """Unique identifier/index for ordering in dimensional matrix."""

    def __post_init__(self) -> None:
        """*__post_init__()* Post-initialization processing with symbol and framework validation.
        """
        super().__post_init__()
        if self._idx != -1:
            self.idx = self._idx

    @property
    def idx(self) -> int:
        """*idx* Get the index/precedence value.

        Returns:
            int: Index value.
        """
        return self._idx

    @idx.setter
    @validate_type(int, allow_none=False)
    @validate_index()
    def idx(self, val: int) -> None:
        """*idx* Set the index/precedence value.

        Args:
            val (int): Index value (must be non-negative).

        Raises:
            ValueError: If index is not a non-negative integer.
        """
        self._idx = val

    def clear(self) -> None:
        """*clear()* Reset index and inherited attributes to default values.

        Resets the entity's index and symbol-related properties to their initial state.
        """
        # Reset parent class attributes
        super().clear()

        # Reset index attribute
        self._idx = -1


@dataclass
class Foundation(IdxBasis):
    """**Foundation** Foundation class for all dimensional analysis entities.

    Provides common validation logic and attributes shared by FDU, Variable, and Coeffcient classes.

    Attributes:
        name (str): User-friendly name
        description (str): Brief summary or description
    """

    # :attr: _name
    _name: str = ""
    """User-friendly name of the entity."""

    # :attr: description
    description: str = ""
    """Brief summary or description of the entity."""

    def __post_init__(self) -> None:
        """*__post_init__()* Post-initialization processing with description capitalization.
        """
        if self.description:
            self.description = self.description.strip()

    @property
    def name(self) -> str:
        """*name* Get the name.

        Returns:
            str: Name value.
        """
        return self._name

    @name.setter
    @validate_type(str, allow_none=False)
    def name(self, val: str) -> None:
        """*name* Set the name with validation.

        Args:
            val (str): Name value.

        Raises:
            ValueError: If name is not a non-empty string.
        """
        self._name = val.strip()

    def clear(self) -> None:
        """*clear()* Reset all attributes to default values.

        Resets the entity's properties to their initial state.
        """
        # Reset parent class attributes
        super().clear()

        # Reset name and description attributes
        self.name = ""
        self.description = ""

    def __str__(self) -> str:
        """*__str__()* String representation showing all non-private attributes.

        Returns:
            str: Formatted string representation.
        """
        attr_list = []
        for attr, val in vars(self).items():
            if attr.startswith("__"):
                continue
            attr_name = attr.lstrip("_")
            attr_list.append(f"{attr_name}={repr(val)}")
        return f"{self.__class__.__name__}({', '.join(attr_list)})"

    def __repr__(self) -> str:
        """*__repr__()* Detailed string representation.

        Returns:
            str: String representation.
        """
        return self.__str__()
