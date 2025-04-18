# -*- coding: utf-8 -*-
"""
Module class **RegexManager** manages the regular expressions (regex) for validating the Fundamental Dimensional Unit (FDU) in *PyDASA*.

*IMPORTANT:* Based on the theory from:

    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""

# native python modules
# import dataclass for defining the node class
from typing import List, Generic, Optional
from dataclasses import dataclass

# custom modules
# generic error handling and type checking
from Src.PyDASA.Util.dflt import T

# import PyDASA global variables (framework, precedence list, and regex)
from Src.PyDASA.Util.cfg import (
    FDU_FWK_DT,
    PHY_FDU_PREC_DT,
    COMPU_FDU_PREC_DT,
    DIGI_FDU_PREC_DT,
    DFLT_POW_REGEX
)

# import the 'cfg' module with to allow global variable edition
from Src.PyDASA.Util import cfg

# checking custom modules
assert T


@dataclass
class RegexManager(Generic[T]):
    """*RegexManager* Manages regex patterns for Fundamental Dimensional Units (FDUs) in *PyDASA*.

    Args:
        Generic (T): Generic type for a Python data structure.

    Returns:
        RegexManager: An object with the following attributes:
            - _fwk (str): Supported framework of the FDU using the *FDU_FWK_DT* map. By default, it is set to `PHYSICAL`.
            - _fdu_prec_lt(Optional[List[str]]): FDU precedence list.
            - _fdu_regex (Optional[str]): Regex pattern for matching FDUs. (e.g., 'M/L*T^-2' to 'M^1*L^-1*T^-2').
            - _fdu_pow_regex (Optional[str]): Regex pattern for matching FDUs with exponents. (e.g., 'M*L^-1*T^-2' to 'M^(1)*L^(-1)*T^(-2)').
            - _fdu_no_pow_regex (Optional[str]): Regex pattern for matching FDUs without exponents. (e.g., 'M*L*T' to 'M^(1)*L^(1)*T^(1)').
            - _fdu_sym_regex (Optional[str]): Regex pattern for matching FDUs in Sympy processor. (e.g., 'M^(1)*L^(-1)*T^(-2)' to 'L**(-1)*M**(1)*T**(-2)').
    """

    # private attributes
    # FDU framework, linked to FDU_FWK_DT
    # :attr: _fwk
    _fwk: str = "PHYSICAL"
    """
    Framework of the FDU, can be one of the following: `PHYSICAL`, `COMPUTATION`, `DIGITAL` or `CUSTOM`. By default, it is set to `PHYSICAL`.
    """

    # FDU precedence list for regex, linked to WKNG_FDU_PREC_LT.
    # :attr: _fdu_prec_lt
    _fdu_prec_lt: Optional[List[str]] = None
    """
    FDU precedence list, define the order of the FDU regex pattern and the columns in the dimensional matrix. By default, it is set to `None`.
    """

    # FDUs matching regex pattern, linked to WKNG_FDU_REGEX.
    # :attr: _fdu_regex
    _fdu_regex: Optional[str] = None
    """
    Main FDU Regex pattern for matching FDUs in *PyDASA*. (e.g., 'M/L*T^-2' to 'M^1*L^-1*T^-2').
    """

    # FDUs matching regex pattern for dimensions with exponents, linked to WKNG_POW_REGEX.
    # :attr: _fdu_pow_regex
    _fdu_pow_regex: Optional[str] = None
    """
    Regex pattern for matching FDUs with exponents. (e.g., 'M*L^-1*T^-2' to 'M^(1)*L^(-1)*T^(-2)').
    """

    # FDUs matching regex pattern for dimensions without exponents, linked to WKNG_NO_POW_REGEX.
    # :attr: _fdu_no_pow_regex
    _fdu_no_pow_regex: Optional[str] = DFLT_POW_REGEX
    """
    Regex pattern for matching FDUs without exponents. (e.g., 'M*L*T' to 'M^(1)*L^(1)*T^(1)').

    NOTE: This pattern doesn't change under any circumstances.
    """

    # FDUs matching regex pattern for dimensions in Sympy symbolic processor, linked to WKNG_FDU_SYM_REGEX.
    # :attr: _fdu_sym_regex
    _fdu_sym_regex: Optional[str] = None
    """
    Regex pattern for matching FDUs in Sympy processor. (e.g., 'M^(1)*L^(-1)*T^(-2)' to 'L**(-1)*M**(1)*T**(-2)').
    """

    # public attributes

    def __post_init__(self) -> None:
        """*__post_init__()* Initializes the *RegexManager* by setting up dimensional precedence and regex patterns.
        """
        self.setup_precedence()
        self.setup_regex()

    def setup_precedence(self) -> None:
        """*_setup_precedence()* initializes the FDU framework and sets up the FDUs precedence list.

        Raises:
            ValueError: error if the framework is invalid.
        """
        # if the framework is supported, configure the default
        if self.fwk in FDU_FWK_DT and self.fwk != "CUSTOM":
            self.fdu_prec_lt = self._get_default_precedence()
        # if the framework is user-defined, use the provided list
        elif self.fwk == "CUSTOM" and self.fdu_prec_lt:
            self.fdu_prec_lt = self._fdu_prec_lt
        # otherwise, raise an error
        else:
            _msg = f"Invalid Framework: {self.fwk}. "
            _msg += "Must be one of the following: "
            _msg += f"{', '.join(FDU_FWK_DT.keys())}."
            raise ValueError(_msg)

    def _get_default_precedence(self) -> List[str]:
        """*_get_default_precedence()* Returns the default FDU precedence list for the specified framework.

        Raises:
            ValueError: error if the framework is invalid.

        Returns:
            List[str]: Default FDUs precedence list based on the framework map.
        """
        # map for easy access to the FDUs
        _frk_map = {
            "PHYSICAL": PHY_FDU_PREC_DT,
            "COMPUTATION": COMPU_FDU_PREC_DT,
            "DIGITAL": DIGI_FDU_PREC_DT,
        }
        if self.fwk in _frk_map:
            _prec_lt = list(_frk_map[self.fwk].keys())
            return _prec_lt
        # otherwise, raise an error
        _msg = f"Invalid Framework: {self.fwk}. "
        _msg += "Must be one of the following: "
        _msg += f"{', '.join(FDU_FWK_DT.keys())}."
        raise ValueError(_msg)

    def setup_regex(self) -> None:
        """*_setup_wkng_regex()* Sets up regex patterns for FDUs based on the precedence list.

        Raises:
            ValueError: error if not all elements in the precedence list are strings.
        """
        # check for errors on the FDU precedence list
        if not all(isinstance(i, str) for i in self.fdu_prec_lt):
            _msg = "FDUs precedence list must be a non-empty list of strings. "
            _msg += f"Provided: {self.fdu_prec_lt}"
            raise ValueError(_msg)
        # Otherwise, compile the regex patterns for FDUs

        # check for valid custom FDU precedence list
        self.fdu_prec_lt = self._fdu_prec_lt
        _fdu_chars = ''.join(self.fdu_prec_lt)

        # compile FDU regex patterns and check for errors
        self._fdu_regex = rf"^[{_fdu_chars}](\^-?\d+)?(\*[{''.join(self.fdu_prec_lt)}](?:\^-?\d+)?)*$"
        self.fdu_regex = self._fdu_regex

        # compile FDU regex patterns with exponent and check for errors
        self._fdu_pow_regex = DFLT_POW_REGEX
        self.fdu_pow_regex = self._fdu_pow_regex

        # compile FDU regex patterns without exponent and check for errors
        self._fdu_no_pow_regex = rf"[{_fdu_chars}](?!\^)"
        self.fdu_no_pow_regex = self._fdu_no_pow_regex

        # compile FDU regex patterns in Sympy Lib and check for errors
        self._fdu_sym_regex = rf"[{_fdu_chars}]"
        self.fdu_sym_regex = self._fdu_sym_regex

    def update_global_regex(self) -> None:
        """*update_global_regex()* Updates global config variables with the custom regex patterns.
        """
        cfg.WKNG_FDU_PREC_LT = self._fdu_prec_lt
        cfg.WKNG_FDU_REGEX = self._fdu_regex
        cfg.WKNG_POW_REGEX = self._fdu_pow_regex
        cfg.WKNG_NO_POW_REGEX = self._fdu_no_pow_regex
        cfg.WKNG_FDU_SYM_REGEX = self._fdu_sym_regex

    @property
    def fwk(self) -> str:
        """*fwk* Get the FDU framework.

        Returns:
            str: Working framework of the FDU.
        """
        return self._fwk

    @fwk.setter
    def fwk(self, value: str) -> None:
        """*fwk* Set the FDU framework. It must be one of the allowed values. The allowed values are: `PHYSICAL`, `COMPUTATION`, `DIGITAL` or `CUSTOM`.

        Args:
            value (str): Framework of the FDU.

        Raises:
            ValueError: If the framework is not one of the allowed values.
        """
        if value not in FDU_FWK_DT.keys():
            _msg = f"Invalid Framework: {value}. "
            _msg += "Must be one of the following: "
            _msg += f"{', '.join(FDU_FWK_DT.keys())}."
            raise ValueError(_msg)
        self._fwk = value

    @property
    def fdu_prec_lt(self) -> List[str]:
        """*fdu_prec_lt* Get the FDUs precedence list.

        Returns:
            List[str]: FDUs precedence list.
        """
        return self._fdu_prec_lt

    @fdu_prec_lt.setter
    def fdu_prec_lt(self, value: List[str]) -> None:
        """*fdu_prec_lt* Set the FDUs precedence list.

        Args:
            value (List[str]): FDUs precedence list.

        Raises:
            ValueError: If the FDUs precedence list is empty or invalid.
        """
        if not value or not all(isinstance(i, str) for i in value):
            _msg = "FDUs precedence list must be a non-empty list of strings. "
            _msg += f"Provided: {value}"
            raise ValueError(_msg)
        self._fdu_prec_lt = value

    @property
    def fdu_regex(self) -> str:
        """*fdu_regex* Get the FDUs matching regex pattern.

        Returns:
            str: FDUs matching regex pattern.
        """
        return self._fdu_regex

    @fdu_regex.setter
    def fdu_regex(self, value: str) -> None:
        """*fdu_regex* Set the FDUs matching regex pattern.

        Args:
            value (str): FDUs matching regex pattern.

        Raises:
            ValueError: If the FDUs matching regex pattern is empty or invalid.
        """
        if not value or not isinstance(value, str):
            _msg = "FDUs matching regex pattern must be a non-empty string. "
            _msg += f"Provided: {value}"
            raise ValueError(_msg)
        self._fdu_regex = value

    @property
    def fdu_pow_regex(self) -> str:
        """*fdu_pow_regex* Get the FDUs matching regex pattern for dimensions with exponent.

        Returns:
            str: FDUs matching regex pattern for dimensions with exponent.
        """
        return self._fdu_pow_regex

    @fdu_pow_regex.setter
    def fdu_pow_regex(self, value: str) -> None:
        """*fdu_pow_regex* Set the FDUs matching regex pattern for dimensions with exponent.

        Args:
            value (str): FDUs matching regex pattern for dimensions with exponent.

        Raises:
            ValueError: If the FDUs matching regex pattern is empty or invalid.
        """
        if not value or not isinstance(value, str):
            _msg = "FDUs matching regex pattern for dimensions with exponent must be a non-empty string. "
            _msg += f"Provided: {value}"
            raise ValueError(_msg)
        self._fdu_pow_regex = value

    @property
    def fdu_no_pow_regex(self) -> str:
        """*fdu_no_pow_regex* Get the FDUs matching regex pattern for dimensions without exponent.

        Returns:
            str: FDUs matching regex pattern for dimensions without exponent.
        """
        return self._fdu_no_pow_regex

    @fdu_no_pow_regex.setter
    def fdu_no_pow_regex(self, value: str) -> None:
        """*fdu_no_pow_regex* Set the FDUs matching regex pattern for dimensions without exponent.

        Args:
            value (str): FDUs matching regex pattern for dimensions without exponent.

        Raises:
            ValueError: If the FDUs matching regex pattern is empty or invalid.
        """
        if not value or not isinstance(value, str):
            _msg = "FDUs matching regex pattern for dimensions without exponent must be a non-empty string. "
            _msg += f"Provided: {value}"
            raise ValueError(_msg)
        self._fdu_no_pow_regex = value

    @property
    def fdu_sym_regex(self) -> str:
        """*fdu_sym_regex* Get the FDUs matching regex pattern for dimensions in Sympy symbolic processor.

        Returns:
            str: FDUs matching regex pattern for dimensions in Sympy symbolic processor.
        """
        return self._fdu_sym_regex

    @fdu_sym_regex.setter
    def fdu_sym_regex(self, value: str) -> None:
        """*fdu_sym_regex* Set the FDUs matching regex pattern for dimensions in Sympy symbolic processor.

        Args:
            value (str): FDUs matching regex pattern for Sympy processor.

        Raises:
            ValueError: If the FDUs matching regex pattern is empty or invalid.
        """
        if not value or not isinstance(value, str):
            _msg = "FDUs matching regex pattern for dimensions in Sympy symbolic processor must be a non-empty string. "
            _msg += f"Provided: {value}"
            raise ValueError(_msg)
        self._fdu_sym_regex = value

    def __str__(self) -> str:
        """*__str__()* get the string representation of the *RegexManager* instance.

        Returns:
            str: String representation of the *RegexManager* instance.
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
        """*__repr__()* get the string representation of the *RegexManager* instance.

        Returns:
            str: String representation of the *RegexManager* instance.
        """
        return self.__str__()
