"""
Module fundamental.py
===========================================

This module provides base classes with common validation logic used across
FDU, Parameter, and Variable classes to eliminate code duplication.

Classes:
    **Validation**: enforces common validation logic.
    **IdxValidation**: enforces index/precedence validation logic.
    **SymValidation**: enforces symbol and framework validation logic.

*IMPORTANT:* Based on the theory from:

    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""
# native python modules
# forward references + postpone eval type hints
from __future__ import annotations
from dataclasses import dataclass
# TODO do I need the Generic, T stuf???
# from typing import Generic
# from new.pydasa.utils.default import T

# indicate it is an abstract base class
from abc import ABC
import re

# import global variables
from new.pydasa.utils.config import FDU_FWK_DT, LATEX_RE


@dataclass
class SymValidation(ABC):
    """**SymValidation** Base class for entities with symbols and framework functionalities.

    Attributes:
        _sym (str): Symbol representation.
        _pyalias (str): Python-compatible alias for use in code.
        _fwk (str): Framework context.
    """

    # :attr: _sym
    _sym: str = ""
    """Symbol representation (LaTeX or alphanumeric)."""

    # :attr: _pyalias
    _pyalias: str = ""
    """Python-compatible alias for symbol, used in executable code."""

    # :attr: _fwk
    _fwk: str = "PHYSICAL"
    """Framework context (PHYSICAL, COMPUTATION, SOFTWARE, CUSTOM)."""

    def __post_init__(self) -> None:
        """Post-initialization processing with symbol and framework validation."""
        # super().__post_init__()
        # Validate the symbol and framework
        if not self._sym:
            self._sym = self._sym.strip()
        if not self._fwk:
            self._fwk = self._fwk.strip()
        if not self._pyalias:
            self._pyalias = self._pyalias.strip()

    @property
    def sym(self) -> str:
        """*sym* Get the symbol.

        Returns:
            str: Symbol value.
        """
        return self._sym

    @sym.setter
    def sym(self, val: str) -> None:
        """*sym* Set the symbol with validation.

        Args:
            val (str): Symbol value.

        Raises:
            ValueError: If symbol format is invalid.
        """
        self._validate_sym(val)
        self._sym = val

    @property
    def fwk(self) -> str:
        """*fwk* Get the framework.

        Returns:
            str: Framework value.
        """
        return self._fwk

    @fwk.setter
    def fwk(self, val: str) -> None:
        """*fwk* Set the framework with validation.

        Args:
            val (str): Framework value.

        Raises:
            ValueError: If framework is not supported.
        """
        self._validate_fwk(val)
        self._fwk = val

    @property
    def pyalias(self) -> str:
        """*pyalias* Get the Python alias for the variable symbol.

        Returns:
            str: Python-compatible alias for use in executable code.
        """
        return self._pyalias

    @pyalias.setter
    def pyalias(self, val: str) -> None:
        """*pyalias* Set the Python alias with validation.

        Args:
            val (str): Python alias value.

        Raises:
            ValueError: If Python alias is not a valid Python identifier.
        """
        if not val.isidentifier():
            _msg = f"Python alias must be a valid Python identifier. Provided: {val}"
            raise ValueError(_msg)
        self._pyalias = val

    def _validate_sym(self, val: str) -> None:
        """*_validate_sym()* Validate symbol format.

        Args:
            val (str): Symbol to validate.

        Raises:
            ValueError: If symbol format is invalid.
        """
        if not isinstance(val, str) or not val.strip():
            _msg = f"Symbol must be a non-empty string. Provided: {val}"
            raise ValueError(_msg)

        # Accept valid LaTeX or alphanumeric symbols
        is_latex = re.match(LATEX_RE, val)
        is_alnum = val.isalnum()

        # Optionally restrict length for non-LaTeX symbols
        # TODO check this, might be wrong!!!
        if not (is_alnum or is_latex):
            msg = (
                "Symbol must be alphanumeric or a valid LaTeX string. "
                f"Provided: '{val}'. "
                "Examples: 'V', 'd', '\\Pi_{0}', '\\rho'."
            )
            raise ValueError(msg)

    # Add this to your Validation or SymValidation class
    def _validate_fwk(self, value: str) -> None:
        """*_validate_fwk()* Validates the framework identifier.

        Args:
            value (str): Framework identifier to validate.

        Raises:
            ValueError: If the framework identifier is invalid.
        """
        # from src.pydasa.utils.config import FDU_FWK_DT
        if value not in FDU_FWK_DT:
            msg = f"Invalid framework: {value}. "
            msg += "Framework must be one of the following: "
            msg += f"{', '.join(FDU_FWK_DT.keys())}."
            raise ValueError(msg)


@dataclass
class IdxValidation(SymValidation):
    """**IdxValidation** Base class for entities with index/precedence functionality.

    Attributes:
        _idx (int): Index/precedence value
    """

    # :attr: _idx
    _idx: int = -1
    """Unique identifier/index for ordering in dimensional matrix."""

    def __post_init__(self) -> None:
        """Post-initialization processing with index validation."""
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
    def idx(self, val: int) -> None:
        """*idx* Set the index/precedence value.

        Args:
            val (int): Index value (must be non-negative).

        Raises:
            ValueError: If index is not a non-negative integer.
        """

        self._validate_idx(val)
        self._idx = val

    def _validate_idx(self, val: int) -> None:
        """*idx* Validate index/precedence value.

        Args:
            val (int): Index value to validate.

        Raises:
            ValueError: If index is not a non-negative integer.
        """
        if not isinstance(val, int) or val < 0:
            _msg = f"Index must be a non-negative integer. Provided: {val}"
            raise ValueError(_msg)


@dataclass
class Validation(IdxValidation):
    """**Validation** Base class for all dimensional analysis entities.

    Provides common validation logic and attributes shared by FDU, Variable, and Coeffcient classes.

    Attributes:
        name (str): User-friendly name
        description (str): Brief summary or description
    """

    # :attr: name
    name: str = ""
    """User-friendly name of the entity."""

    # :attr: description
    description: str = ""
    """Brief summary or description of the entity."""

    def __post_init__(self) -> None:
        """Post-initialization processing with description capitalization."""
        if self.description:
            self.description = self.description.capitalize()

    def _validate_name(self, name: str) -> None:
        """Validate the name format.

        Args:
            name (str): Name to validate.

        Raises:
            ValueError: If name is not a non-empty string.
        """
        if not isinstance(name, str) or not name.strip():
            _msg = f"Name must be a non-empty string. Provided: {name}"
            raise ValueError(_msg)
        self.name = name.strip()

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
