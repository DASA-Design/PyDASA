# -*- coding: utf-8 -*-
"""
Module framework.py
===========================================

Module for **DimSchema** to manage Fundamental Dimensional Units (FDUs) for Dimensional Analysis in *PyDASA*.

This module provides the DimSchema class which manages dimensional frameworks, FDU precedence, and regex patterns for dimensional expression validation.

Classes:
    **DimSchema**: Manages dimensional frameworks and FDUs, providing methods for validation,

*IMPORTANT:* Based on the theory from:

    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""

from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import List, Dict, Optional, Generic, Any, Sequence, Mapping, cast

# Import validation base classes
from pydasa.core.basic import Validation

# Import Dimension class
from pydasa.core.fundamental import Dimension

# Import generic type
from pydasa.utils.default import T

# Import global variables
from pydasa.utils.config import (
    FDU_FWK_DT,
    PHY_FDU_PREC_DT,
    COMPU_FDU_PREC_DT,
    SOFT_FDU_PREC_DT,
    DFLT_POW_RE
)

# Import the 'cfg' module to allow global variable editing
from pydasa.utils import config as cfg


@dataclass
class DimSchema(Validation, Generic[T]):
    """**DimSchema** Manages dimensional frameworks and FDUs for *PyDASA*.

    Maintains a collection of Dimensions with their precedence, provides regex patterns
    for dimensional expressions, and manages the dimensional framework context.

    Attributes:
        _fdu_lt (List[Dimension]): List of Fundamental Dimensional Units in precedence order.
        _fdu_map (Dict[str, Dimension]): Dictionary mapping FDU symbols to Dimension objects.
        _fdu_regex (str): Regex pattern for matching dimensional expressions (e.g., 'M/L*T^-2' to 'M^1*L^-1*T^-2').
        _fdu_pow_regex (str): Regex pattern for matching dimensions with exponents. (e.g., 'M*L^-1*T^-2' to 'M^(1)*L^(-1)*T^(-2)').
        _fdu_no_pow_regex (str): Regex pattern for matching dimensions without exponents. (e.g., 'M*L*T' to 'M^(1)*L^(1)*T^(1)').
        _fdu_sym_regex (str): Regex pattern for matching FDUs in symbolic expressions. (e.g., 'M^(1)*L^(-1)*T^(-2)' to 'L**(-1)*M**(1)*T**(-2)').
    """

    # FDUs storage
    # FDU precedence list, linked to WKNG_FDU_PREC_LT.
    # :attr: _fdu_lt
    _fdu_lt: List[Dimension] = field(default_factory=list[Dimension])
    """List of Fundamental Dimensional Units in precedence order."""

    # FDU framework
    # :attr: _fwk
    _fdu_map: Dict[str, Dimension] = field(default_factory=dict)
    """Dictionary mapping FDU symbols to Dimension objects."""

    # :attr: _fdu_symbols
    _fdu_symbols: List[str] = field(default_factory=list)
    """List of FDU symbols in the framework."""

    # Regex patterns
    # FDUs matching regex pattern, linked to WKNG_FDU_RE.
    # :attr: _fdu_regex
    _fdu_regex: str = ""
    """Regex pattern for matching dimensional expressions."""

    # FDU power regex pattern, linked to WKNG_POW_RE.
    # :attr: _fdu_pow_regex
    _fdu_pow_regex: str = DFLT_POW_RE
    """Regex pattern for matching dimensions with exponents."""

    # FDU no power regex pattern, linked to WKNG_NO_POW_RE.
    # :attr: _fdu_no_pow_regex
    _fdu_no_pow_regex: str = ""
    """Regex pattern for matching dimensions without exponents

    NOTE: This pattern doesn't change under any circumstances.
    """

    # FDU symbolic regex pattern, linked to WKNG_FDU_SYM_RE.
    # :attr: _fdu_sym_regex
    _fdu_sym_regex: str = ""
    """Regex pattern for matching FDUs in symbolic expressions."""

    def __post_init__(self) -> None:
        """*__post_init__()* Initializes the framework and sets up regex patterns.
        """
        # Initialize base class
        super().__post_init__()

        # Initialize FDUs based on framework
        self._setup_fdus()

        # Initialize indices, map, and symbol precedence
        self._validate_fdu_precedence()
        self._update_fdu_map()
        self._update_fdu_symbols()

        # Generate regex patterns
        self._setup_regex()

    def _setup_fdus(self) -> None:
        """*_setup_fdus()* Initializes FDUs based on the selected framework.

        Creates and adds standard FDUs for the selected framework (PHYSICAL,
        COMPUTATION, SOFTWARE) or validates custom FDUs.

        Raises:
            ValueError: If the FDU framework is not properly defined.
        """
        # if the framework is supported, configure the default
        if self.fwk in FDU_FWK_DT and self.fwk != "CUSTOM":
            self.fdu_lt = self._setup_fdu_framework()

        # if the framework is user-defined, use the provided list[dict]
        elif self.fwk == "CUSTOM":
            if not self._fdu_lt:
                raise ValueError("Custom framework requires '_fdu_lt' to define FDUs")

            # Check if _fdu_lt contains Dimension objects (already created)
            if all(isinstance(val, Dimension) for val in self._fdu_lt):
                # Already Dimension objects, just assign them
                self.fdu_lt = self._fdu_lt
            # Check if _fdu_lt contains dicts (need to be converted)
            elif all(isinstance(val, dict) for val in self._fdu_lt):
                # Convert dicts to Dimension objects
                raw = cast(Sequence[Mapping[str, Any]], self._fdu_lt)
                self.fdu_lt = self._setup_custom_framework(raw)
            else:
                _msg = "'fdu_lt' elements must be type Dimension() or dict()"
                raise ValueError(_msg)

        # otherwise, raise an error
        else:
            _msg = f"Invalid Framework: {self.fwk}. "
            _msg += f"Valid options: {', '.join(FDU_FWK_DT.keys())}."
            raise ValueError(_msg)

    def _setup_custom_framework(self,
                                fdus: Sequence[Mapping[str, Any]]) -> List[Dimension]:
        """*_setup_custom_framework()* Initializes a custom framework with the provided FDUs.

        Args:
            fdus (List[Dict]): List of dictionaries representing custom FDUs.

        Returns:
            List[Dimension]: List of Dimension objects created from the provided FDUs.
        """
        # detecting custom framework
        ans = []
        if self.fwk == "CUSTOM":
            # Create custom FDU set
            for idx, data in enumerate(fdus):
                data = dict(data)
                fdu = Dimension(
                    _idx=idx,
                    _sym=data.get("_sym", ""),
                    _fwk=self._fwk,
                    _unit=data.get("_unit", ""),
                    name=data.get("name", ""),
                    description=data.get("description", ""))

                ans.append(fdu)
        return ans

    def _setup_fdu_framework(self) -> List[Dimension]:
        """*_setup_fdu_framework()* Returns the default FDU precedence list for the specified framework.

        Returns:
            List[str]: Default FDUs precedence list based on the framework map.
        """
        # map for easy access to the FDUs
        _frk_map = {
            "PHYSICAL": PHY_FDU_PREC_DT,
            "COMPUTATION": COMPU_FDU_PREC_DT,
            "SOFTWARE": SOFT_FDU_PREC_DT,
        }
        ans = []
        # select FDU framework
        if self.fwk in _frk_map:
            # Create standard FDU set
            for idx, (sym, data) in enumerate(_frk_map[self.fwk].items()):
                fdu = Dimension(
                    _idx=idx,
                    _sym=sym,
                    _fwk=self._fwk,
                    _unit=data.get("_unit", ""),
                    name=data.get("name", ""),
                    description=data.get("description", ""))

                ans.append(fdu)
            # _prec_lt = list(_frk_map[self.fwk].keys())
        return ans

    def _validate_fdu_precedence(self) -> None:
        """*_validate_fdu_precedence()* Ensures FDUs have valid and unique precedence values.

        Raises:
            ValueError: If FDU precedence values are duplicated.
        """
        # trick to do nothing if FDU set is null
        if not self._fdu_lt:
            return

        # Check for duplicate precedence values
        indices = [fdu.idx for fdu in self._fdu_lt]
        if len(indices) != len(set(indices)):
            raise ValueError("Duplicate precedence values in FDUs.")

        # Sort FDUs by idx precedence
        self._fdu_lt.sort(key=lambda fdu: fdu.idx)

    def _update_fdu_map(self) -> None:
        """*_update_fdu_map()* Updates the FDU symbol to object mapping.
        """
        self._fdu_map.clear()
        for fdu in self._fdu_lt:
            self._fdu_map[fdu.sym] = fdu

    def _update_fdu_symbols(self) -> None:
        """*_update_fdu_symbols()* Updates the list of FDU symbols in precedence order."""
        self._fdu_symbols = [fdu.sym for fdu in self._fdu_lt]

    def _setup_regex(self) -> None:
        """*_setup_regex()* Sets up regex patterns for dimensional validation. Generates regex patterns for:
            - validating dimensional expressions.
            - parsing exponents.
            - completing expressions with exponent.
            - handling symbolic expressions.
        """
        # trick to do nothing if FDU set is null
        if not self._fdu_lt:
            return None

        # Get FDU symbols in precedence order
        # fdu_symbols = [fdu.sym for fdu in self._fdu_lt]
        _fdu_chars = ''.join(self.fdu_symbols)

        # Generate main regex for dimensional expressions
        self._fdu_regex = rf"^[{_fdu_chars}](\^-?\d+)?(\*[{_fdu_chars}](?:\^-?\d+)?)*$"

        # Use default regex for exponents
        self._fdu_pow_regex = DFLT_POW_RE

        # Generate regex for dimensions without exponents
        self._fdu_no_pow_regex = rf"[{_fdu_chars}](?!\^)"

        # Generate regex for dimensions in symbolic expressions
        self._fdu_sym_regex = rf"[{_fdu_chars}]"

    def update_global_config(self) -> None:
        """*update_global_config()* Updates global config variables with current framework settings.

        Makes the current framework's settings available globally for all PyDASA components.
        """
        # Get FDU symbols in precedence order
        # fdu_symbols = [fdu.sym for fdu in self._fdu_lt]

        # Update global configuration
        cfg.WKNG_FDU_PREC_LT = self.fdu_symbols
        cfg.WKNG_FDU_RE = self._fdu_regex
        cfg.WKNG_POW_RE = self._fdu_pow_regex
        cfg.WKNG_NO_POW_RE = self._fdu_no_pow_regex
        cfg.WKNG_FDU_SYM_RE = self._fdu_sym_regex

    # propierties getters and setters

    @property
    def fdu_lt(self) -> List[Dimension]:
        """*fdu_lt* Get the list of FDUs in precedence order.

        Returns:
            List[Dimension]: List of FDUs.
        """
        return self._fdu_lt.copy()

    @fdu_lt.setter
    def fdu_lt(self, val: List[Dimension]) -> None:
        """*fdu_lt* Set the FDUs in precedence order.

        Args:
            val (List[Dimension]): List of FDUs.

        Raises:
            ValueError: If the FDUs list is empty or invalid.
        """
        if not val or not all(isinstance(i, Dimension) for i in val):
            _msg = "FDUs list must be a non-empty list of Dimensions. "
            _msg += f"Provided: {val}"
            raise ValueError(_msg)
        self._fdu_lt = val

    @property
    def fdu_symbols(self) -> List[str]:
        """*fdu_symbols* Get the list of FDU symbols in precedence order.

        Returns:
            List[str]: List of FDU symbols.
        """
        return self._fdu_symbols.copy()

    @property
    def size(self) -> int:
        """*size* Get the number of FDUs in the framework.

        Returns:
            int: Number of FDUs.
        """
        return len(self._fdu_lt)

    @property
    def fdu_regex(self) -> str:
        """*fdu_regex* Get the FDU regex pattern.

        Returns:
            str: Regex pattern for validating dimensional expressions.
        """
        return self._fdu_regex

    @fdu_regex.setter
    def fdu_regex(self, val: str) -> None:
        """*fdu_regex* Set the FDUs regex pattern.

        Args:
            val (str): FDUs regex pattern.

        Raises:
            ValueError: If the FDUs regex pattern is empty or not a string.
        """
        if not val or not isinstance(val, str):
            _msg = "FDUs regex pattern must be a non-empty string. "
            _msg += f"Provided: {val}"
            raise ValueError(_msg)
        self._fdu_regex = val

    @property
    def fdu_pow_regex(self) -> str:
        """*fdu_pow_regex* Get the FDU powered regex pattern.

        Returns:
            str: Regex pattern for matching dimensions with exponents.
        """
        return self._fdu_pow_regex

    @fdu_pow_regex.setter
    def fdu_pow_regex(self, val: str) -> None:
        """*fdu_pow_regex* Set the FDUs pow-regex pattern.

        Args:
            val (str): FDUs pow-regex pattern for matching dimensions with exponent.

        Raises:
            ValueError: If the FDUs pow-regex pattern is empty or not a string.
        """
        if not val or not isinstance(val, str):
            _msg = f"Invalid FDU pow-regex pattern: {val}. "
            _msg += "must be a non-empty string."
            raise ValueError(_msg)
        self._fdu_pow_regex = val

    @property
    def fdu_no_pow_regex(self) -> str:
        """*fdu_no_pow_regex* Get the FDU no-power regex pattern.

        Returns:
            str: Regex pattern for matching dimensions without exponents.
        """
        return self._fdu_no_pow_regex

    @fdu_no_pow_regex.setter
    def fdu_no_pow_regex(self, val: str) -> None:
        """*fdu_no_pow_regex* Set the FDUs no-pow-regex pattern.

        Args:
            val (str): FDUs no-pow-regex pattern for matching dimensions without exponent.

        Raises:
            ValueError: If the FDUs no-pow-regex pattern is empty or not a string.
        """
        if not val or not isinstance(val, str):
            _msg = f"Invalid FDU no-pow-regex pattern: {val}. "
            _msg += "must be a non-empty string."
            raise ValueError(_msg)
        self._fdu_no_pow_regex = val

    @property
    def fdu_sym_regex(self) -> str:
        """*fdu_sym_regex* Get the FDU symbol regex pattern.

        Returns:
            str: Regex pattern for matching FDUs in symbolic expressions.
        """
        return self._fdu_sym_regex

    @fdu_sym_regex.setter
    def fdu_sym_regex(self, val: str) -> None:
        """*fdu_sym_regex* Set the FDUs sym-regex pattern.

        Args:
            val (str): FDUs sym-regex pattern for matching dimensions in symbolic expressions.

        Raises:
            ValueError: If the FDUs sym-regex pattern is empty or not a string.
        """
        if not val or not isinstance(val, str):
            _msg = f"Invalid FDUs sym-regex pattern: {val}. "
            _msg += "must be a non-empty string."
            raise ValueError(_msg)
        self._fdu_sym_regex = val

    def get_fdu(self, symbol: str) -> Optional[Dimension]:
        """*get_fdu()* Get an FDU by its symbol.

        Args:
            symbol (str): FDU symbol.

        Returns:
            Optional[Dimension]: FDU object if found, None otherwise.
        """
        return self._fdu_map.get(symbol)

    def has_fdu(self, symbol: str) -> bool:
        """*has_fdu()* Check if an FDU with the given symbol exists.

        Args:
            symbol (str): FDU symbol.

        Returns:
            bool: True if the FDU exists, False otherwise.
        """
        return symbol in self._fdu_map

    def add_fdu(self, fdu: Dimension) -> None:
        """*add_fdu()* Add an FDU to the framework.

        Args:
            fdu (Dimension): FDU to add.

        Raises:
            ValueError: If an FDU with the same symbol already exists.
            ValueError: If the FDU framework does not match the current framework.
        """
        if self.has_fdu(fdu.sym):
            raise ValueError(f"FDU with symbol '{fdu.sym}' already exists.")

        # Set framework
        if fdu.fwk != self._fwk:
            _msg = "FDU framework mismatch: "
            _msg += f"Expected '{self._fwk}', got '{fdu.fwk}'"
            raise ValueError(_msg)

        # Add FDU
        self._fdu_lt.append(fdu)

        # Update indices, map, and symbol precedence
        self._validate_fdu_precedence()
        self._update_fdu_map()
        self._update_fdu_symbols()

        # Update regex patterns
        self._setup_regex()

    def remove_fdu(self, sym: str) -> Dimension:
        """*remove_fdu()* Remove an FDU from the framework.

        Args:
            sym (str): Symbol of the FDU to remove.

        Returns:
            Dimension: removed FDU object.
        """
        if not self.has_fdu(sym):
            raise ValueError(f"FDU with symbol '{sym}' does not exist.")

        # Remove FDU
        # find index with the symbol
        if sym in self._fdu_map:
            # direct retrieve the FDU to avoid Optional return of dict.get
            fdu_obj = self._fdu_map[sym]
            # Remove by precedence index and capture the removed Dimension
            idx = fdu_obj.idx
            ans = self._fdu_lt.pop(idx)
        else:
            # Should not happen because of the earlier has_fdu check, but keep safe
            raise ValueError(f"FDU with symbol '{sym}' does not exist.")

        # Update indices, map, and symbol precedence
        self._validate_fdu_precedence()
        self._update_fdu_map()
        self._update_fdu_symbols()

        # Update regex patterns
        self._setup_regex()

        return ans

    def reset(self) -> None:

        self._fdu_lt.clear()
        self._fdu_map.clear()
        self._fdu_symbols.clear()
        self._fdu_regex = ""
        self._fdu_pow_regex = DFLT_POW_RE
        self._fdu_no_pow_regex = ""
        self._fdu_sym_regex = ""

    def to_dict(self) -> Dict[str, Any]:
        """*to_dict()* Convert framework to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation of the framework.
        """
        result = {}

        # Get all dataclass fields
        for f in fields(self):
            attr_name = f.name
            attr_value = getattr(self, attr_name)

            # Handle Dimension list (convert each Dimension)
            if isinstance(attr_value, list) and all(isinstance(d, Dimension) for d in attr_value):
                attr_value = [d.to_dict() for d in attr_value]

            # Handle Dimension dictionary (convert each Dimension)
            if isinstance(attr_value, dict) and all(isinstance(d, Dimension) for d in attr_value.values()):
                attr_value = {k: d.to_dict() for k, d in attr_value.items()}

            # Skip None values for optional fields
            if attr_value is None:
                continue

            # Remove leading underscore from private attributes
            if attr_name.startswith("_"):
                clean_name = attr_name[1:]  # Remove first character
            else:
                clean_name = attr_name

            result[clean_name] = attr_value

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DimSchema:
        """*from_dict()* Create framework from dictionary representation.

        Args:
            data (Dict[str, Any]): Dictionary representation of the framework.

        Returns:
            DimScheme: New DimScheme instance.
        """
        # Get all valid field names from the dataclass
        field_names = {f.name for f in fields(cls)}

        # Map keys without underscores to keys with underscores
        mapped_data = {}

        for key, value in data.items():
            # Try the key as-is first (handles both _fwk and name)
            if key in field_names:
                mapped_data[key] = value
            # Try adding underscore prefix (handles fwk -> _fwk)
            elif f"_{key}" in field_names:
                mapped_data[f"_{key}"] = value
            # Try removing underscore prefix (handles _name -> name if needed)
            elif key.startswith("_") and key[1:] in field_names:
                mapped_data[key[1:]] = value

        # Convert Dimension list back
        if "fdu_lt" in mapped_data or "_fdu_lt" in mapped_data:
            fdu_data = mapped_data.get("fdu_lt") or mapped_data.get("_fdu_lt")
            if isinstance(fdu_data, list):
                mapped_data["_fdu_lt"] = [
                    Dimension.from_dict(d) if isinstance(d, dict) else d
                    for d in fdu_data
                ]

        # Convert Dimension map back
        if "fdu_map" in mapped_data or "_fdu_map" in mapped_data:
            map_data = mapped_data.get(
                "fdu_map") or mapped_data.get("_fdu_map")
            if isinstance(map_data, dict):
                mapped_data["_fdu_map"] = {
                    k: Dimension.from_dict(d) if isinstance(d, dict) else d
                    for k, d in map_data.items()
                }

        # Remove computed/derived fields that shouldn't be passed to constructor
        computed_fields = [
            "fdu_map", "_fdu_map",  # Reconstructed from fdu_lt
            "fdu_symbols", "_fdu_symbols",  # Reconstructed from fdu_lt
            "size"  # Computed property
        ]

        for field_name in computed_fields:
            mapped_data.pop(field_name, None)

        # Create framework instance
        framework = cls(**mapped_data)

        return framework

    # def validate_dimensional_expression(self, expression: str) -> bool:
    # TODO old code, to be removed in the future
    #     """*validate_dimensional_expression()* Check if a dimensional expression is valid.

    #     Args:
    #         expression (str): Dimensional expression to validate.

    #     Returns:
    #         bool: True if the expression is valid, False otherwise.
    #     """
    #     import re

    #     if not expression or not self._fdu_regex:
    #         return False

    #     return bool(re.match(self._fdu_regex, expression))

    # def parse_dimensional_expression(self, expression: str) -> Dict[str, int]:
    #     """*parse_dimensional_expression()* Parse a dimensional expression into a dictionary.

    #     Args:
    #         expression (str): Dimensional expression to parse.

    #     Returns:
    #         Dict[str, int]: Dictionary mapping FDU symbols to exponents.

    #     Raises:
    #         ValueError: If the expression is invalid.
    #     """
    #     if not self.validate_dimensional_expression(expression):
    #         raise ValueError(f"Invalid dimensional expression: {expression}")

    #     # Initialize result with zeros for all FDUs
    #     result = {fdu.sym: 0 for fdu in self._fdu_lt}

    #     # Split by multiplication operator
    #     terms = expression.split('*')

    #     # Process each term
    #     for term in terms:
    #         # Extract symbol and exponent
    #         if '^' in term:
    #             symbol, exponent_str = term.split('^')
    #             exponent = int(exponent_str)
    #         else:
    #             symbol = term
    #             exponent = 1

    #         # Update result
    #         result[symbol] = exponent

    #     return result

    # def format_dimensional_expression(self, dimensions: Dict[str, int]) -> str:
    #     """*format_dimensional_expression()* Format a dictionary of dimensions into an expression.

    #     Args:
    #         dimensions (Dict[str, int]): Dictionary mapping FDU symbols to exponents.

    #     Returns:
    #         str: Formatted dimensional expression.
    #     """
    #     # Sort by FDU precedence
    #     terms = []

    #     # Process dimensions in precedence order
    #     for fdu in self._fdu_lt:
    #         exponent = dimensions.get(fdu.sym, 0)
    #         if exponent != 0:
    #             if exponent == 1:
    #                 terms.append(fdu.sym)
    #             else:
    #                 terms.append(f"{fdu.sym}^{exponent}")

    #     # Join terms with multiplication operator
    #     return '*'.join(terms) if terms else "1"
