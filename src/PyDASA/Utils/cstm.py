# -*- coding: utf-8 -*-
"""
Module **RegexManager** manages the regular expressions (regex) for validating the Fundamental Dimensional Unit (FDU) in *PyDASA*. It use a default or traditional dimensional system; plus, a working (custom or otherwise) regex for the user to define their own FDUs.

*IMPORTANT:* This code and its specifications for Python are based on the theory and subject developed by the following authors/books:

    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""

# native python modules
# import dataclass for defining the node class
from typing import List, Generic, Optional
from dataclasses import dataclass
# import modules for defining the MapEntry type

# custom modules
# generic error handling and type checking
from Src.PyDASA.Utils.err import error_handler as error
from Src.PyDASA.Utils.dflt import T

# import PyDASA's supported FDU frameworks
from Src.PyDASA.Utils.cfg import FDU_FWK_DT
# import PyDASA's default regex for FDU
from Src.PyDASA.Utils.cfg import PHY_FDU_PREC_DT
from Src.PyDASA.Utils.cfg import COMPU_FDU_PREC_DT
from Src.PyDASA.Utils.cfg import DIGI_FDU_PREC_DT
from Src.PyDASA.Utils.cfg import DFLT_POW_REGEX
# from Src.PyDASA.Utils.cfg import DFLT_FDU_SYM_REGEX

# import PyDASA's working regex for FDU, use 'as' to allow shared variable edition
from Src.PyDASA.Utils import cfg as config

# checking custom modules
assert error
assert T


@dataclass
class RegexManager(Generic[T]):
    """*RegexManager* class for managing the regex patterns for Fundamental Dimensional Units (FDU) in *PyDASA*.

    Args:
        Generic (T): Generic type for a Python data structure.

    Returns:
        RegexManager: A RegexManager object with the following attributes:
            - `custom`: A boolean flag indicating if a custom regex is being used.
            - `_fdu_prec_lt`: A list of strings representing the FDUs precedence list.
            - `_fdu_regex`: A string representing the FDUs matching regex pattern.
            - `_fdu_pow_regex`: A string representing the FDUs matching regex pattern for dimensions with exponent.
            - `_fdu_no_pow_regex`: A string representing the FDUs matching regex pattern for dimensions without exponent.
            - `_fdu_sym_regex`: A string representing the FDUs matching regex pattern for dimensions in Sympy symbolic processor.
        All private attributes are initialized with default, use the @property decorator to access and validate their user input.
    """

    # private attributes
    # FDU framework, linked to FDU_FWK_DT
    # :attr: _fwk
    _fwk: str = "PHYSICAL"
    """
    # Framework of the FDU: `PHYSICAL`, `COMPUTATION`, `DIGITAL`, or `CUSTOM`. By default, it is set to `PHYSICAL`.
    """

    # FDU precedence list for regex, linked to WKNG_FDU_PREC_LT.
    # :attr: _fdu_prec_lt
    _fdu_prec_lt: Optional[List[str]] = None
    """
    # FDU precedence list for regex, defining the order of FDUs in the dimensional matrix.
    """

    # FDUs matching regex pattern, linked to WKNG_FDU_REGEX.
    # :attr: _fdu_regex
    _fdu_regex: Optional[str] = None
    """
    Main FDU regex pattern for matching FDUs in *PyDASA*. It is a string for matching FDUs from a formula or parameter into the dimensional matrix. (e.g., 'M/L*T^-2' to 'M^1*L^-1*T^-2').
    """

    # FDUs matching regex pattern for dimensions with exponents, linked to WKNG_POW_REGEX.
    # :attr: _fdu_pow_regex
    _fdu_pow_regex: Optional[str] = None
    """
    Regex pattern for matching FDUs with exponents (e.g., 'M*L^-1*T^-2' to 'M^(1)*L^(-1)*T^(-2)').
    """

    # FDUs matching regex pattern for dimensions without exponents, linked to WKNG_NO_POW_REGEX.
    # :attr: _fdu_no_pow_regex
    _fdu_no_pow_regex: Optional[str] = DFLT_POW_REGEX
    """
    Regex pattern for matching FDUs without exponents (e.g., 'M*L*T' to 'M^(1)*L^(1)*T^(1)'). IMPORTANT: This regex does not change with custom regex.
    """

    # FDUs matching regex pattern for dimensions in Sympy symbolic processor, linked to WKNG_FDU_SYM_REGEX.
    # :attr: _fdu_sym_regex
    _fdu_sym_regex: Optional[str] = None
    """
    Regex pattern for matching FDUs in Sympy (e.g., 'M^(1)*L^(-1)*T^(-2)' to 'L**(-1)*M**(1)*T**(-2)').
    """

    # public attributes

    def __post_init__(self) -> None:
        """*__post_init__()* initializes the *RegexManager* instance and sets up regex patterns for FDUs.

        Raises:
            ValueError: error if the CUSTOM framework lacks a precedence list.
            ValueError: error if the framework is invalid.
        """
        # if the framework is not CUSTOM with NO FDU precedence list
        if self.fwk != "CUSTOM" and not self.fdu_prec_lt:
            # set the default FDU precedence list based on the framework
            self.fdu_prec_lt = self._get_dflt_prec_lt()
        # if the framework not CUSTOM with FDU precedence list
        elif self.fwk != "CUSTOM" and self.fdu_prec_lt:
            # set the framework to CUSTOM
            self.fwk = "CUSTOM"
        # if the framework is CUSTOM with NO FDU precedence list
        elif self.fwk == "CUSTOM" and not self.fdu_prec_lt:
            # raise an error
            _msg = "FDUs precedence list must be a non-empty list of strings. "
            _msg += f"Provided: {self.fdu_prec_lt}"
            raise ValueError(_msg)
        # otherwise, the framework is not valid
        else:
            _msg = "Invalid framework or FDU precedence list."
            _msg += "Framework must be one of the following: "
            _msg += f"{', '.join(FDU_FWK_DT.keys())}."
            raise ValueError(_msg)
        # always setup the working regex patterns based on the FDU precedence list.
        self.setup_wkng_regex()

    def _get_dflt_prec_lt(self) -> List[str]:
        """*_get_dflt_prec_lt()* get the default FDU precedence list based on the framework.

        Raises:
            ValueError: error if the framework is invalid.

        Returns:
            List[str]: Default FDU precedence list based on the framework.
        """
        # if the framework is one of the supported ones
        if self.fwk == "PHYSICAL":
            return list(PHY_FDU_PREC_DT.keys())
        if self.fwk == "COMPUTATION":
            return list(COMPU_FDU_PREC_DT.keys())
        if self.fwk == "DIGITAL":
            return list(DIGI_FDU_PREC_DT.keys())
        # otherwise, raise an error
        _msg = f"Invalid framework: {self.fwk}. "
        _msg += "Framework must be one of the following: "
        _msg += f"{', '.join(FDU_FWK_DT.keys())}."
        raise ValueError(_msg)

    def setup_wkng_regex(self) -> None:
        """*_setup_wkng_regex()* Initializes the *RegexManager* instance and sets up regex patterns for FDUs.

        Raises:
            ValueError: If the CUSTOM framework lacks a precedence list or if the framework is invalid.
        """
        if all(isinstance(i, str) for i in self.fdu_prec_lt):
            # compile the custom regex patterns for FDUs
            self._compile_wkng_vars()
            # update the global variables with the custom regex patterns
            self._update_wkng_vars()
        else:
            _msg = "FDUs precedence list must be a non-empty list of strings. "
            _msg += f"Provided: {self.fdu_prec_lt}"
            raise ValueError(_msg)

    def _compile_wkng_vars(self) -> None:
        """*_compile_wkng_vars()* compiles regex patterns for FDUs based on the provided precedence list. Creates regex patterns for:
            - FDUs with exponents.
            - FDUs without exponents.
            - FDUs in Sympy symbolic processor.
        """
        # FDU precedence list
        # check for valid custom FDU precedence list
        self.fdu_prec_lt = self._fdu_prec_lt
        # creating the FDU precedence list string
        _fdu_chars = ''.join(self.fdu_prec_lt)

        # compile FDU regex patterns
        self._fdu_regex = rf"^[{_fdu_chars}](\^-?\d+)?(\*[{''.join(self.fdu_prec_lt)}](?:\^-?\d+)?)*$"
        # check for valid FDU custom regex
        self.fdu_regex = self._fdu_regex

        # compile FDU regex patterns with exponent
        self._fdu_pow_regex = DFLT_POW_REGEX
        # check for valid FDU custom regex
        self.fdu_pow_regex = self._fdu_pow_regex

        # compile FDU regex patterns without exponent
        self._fdu_no_pow_regex = rf"[{_fdu_chars}](?!\^)"
        # check for valid FDU custom regex
        self.fdu_no_pow_regex = self._fdu_no_pow_regex

        # compile FDU regex patterns in Sympy symbolic processor
        self._fdu_sym_regex = rf"[{_fdu_chars}]"
        # check for valid FDU custom regex
        self.fdu_sym_regex = self._fdu_sym_regex

    def _update_wkng_vars(self) -> None:
        """*_update_wkng_vars()* Updates the global *config* variables with custom regex patterns for FDUs.
        """
        # update global variables with the custom regex patterns
        config.WKNG_FDU_PREC_LT = self._fdu_prec_lt
        config.WKNG_FDU_REGEX = self._fdu_regex
        config.WKNG_POW_REGEX = self._fdu_pow_regex
        config.WKNG_NO_POW_REGEX = self._fdu_no_pow_regex
        config.WKNG_FDU_SYM_REGEX = self._fdu_sym_regex

    @property
    def fwk(self) -> str:
        """*fwk* property to get the framework of the FDU.

        Returns:
            str: Framework of the FDU. It can be one of the following: `PHYSICAL`, `COMPUTATION`, `DIGITAL` or `CUSTOM`.
        """
        return self._fwk

    @fwk.setter
    def fwk(self, value: str) -> None:
        """*fwk* property of the allowed framework of the FDU.

        Args:
            value (str): Framework of the FDU. It can be one of the following: `PHYSICAL`, `COMPUTATION`, `DIGITAL` or `CUSTOM`.

        Raises:
            ValueError: If the framework is not one of the allowed values.
        """
        if value not in FDU_FWK_DT.keys():
            _msg = f"Invalid framework: {value}. "
            _msg += "Framework must be one of the following: "
            _msg += f"{', '.join(FDU_FWK_DT.keys())}."
            raise ValueError(_msg)
        self._fwk = value

    @property
    def fdu_prec_lt(self) -> List[str]:
        """*fdu_prec_lt* property to get the FDUs precedence list.

        Returns:
            List[str]: FDUs precedence list.
        """
        return self._fdu_prec_lt

    @fdu_prec_lt.setter
    def fdu_prec_lt(self, value: List[str]) -> None:
        """*fdu_prec_lt* property to set the FDUs precedence list.

        Args:
            value (List[str]): FDUs precedence list.

        Raises:
            ValueError: If the FDUs precedence list is empty or contains invalid characters.
        """
        if not value or not all(isinstance(i, str) for i in value):
            _msg = "FDUs precedence list must be a non-empty list of strings. "
            _msg += f"Provided: {value}"
            raise ValueError(_msg)
        self._fdu_prec_lt = value

    @property
    def fdu_regex(self) -> str:
        """*fdu_regex* property to get the FDUs matching regex pattern.

        Returns:
            str: FDUs matching regex pattern.
        """
        return self._fdu_regex

    @fdu_regex.setter
    def fdu_regex(self, value: str) -> None:
        """*fdu_regex* property to set the FDUs matching regex pattern.

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
        """*fdu_pow_regex* property to get the FDUs matching regex pattern for dimensions with exponent.

        Returns:
            str: FDUs matching regex pattern for dimensions with exponent.
        """
        return self._fdu_pow_regex

    @fdu_pow_regex.setter
    def fdu_pow_regex(self, value: str) -> None:
        """*fdu_pow_regex* property to set the FDUs matching regex pattern for dimensions with exponent.

        Args:
            value (str): FDUs matching regex pattern for dimensions with exponent.

        Raises:
            ValueError: If the FDUs matching regex pattern for dimensions with exponent is empty or invalid.
        """
        if not value or not isinstance(value, str):
            _msg = "FDUs matching regex pattern for dimensions with exponent must be a non-empty string. "
            _msg += f"Provided: {value}"
            raise ValueError(_msg)
        self._fdu_pow_regex = value

    @property
    def fdu_no_pow_regex(self) -> str:
        """*fdu_no_pow_regex* property to get the FDUs matching regex pattern for dimensions without exponent.

        Returns:
            str: FDUs matching regex pattern for dimensions without exponent.
        """
        return self._fdu_no_pow_regex

    @fdu_no_pow_regex.setter
    def fdu_no_pow_regex(self, value: str) -> None:
        """*fdu_no_pow_regex* property to set the FDUs matching regex pattern for dimensions without exponent.

        Args:
            value (str): FDUs matching regex pattern for dimensions without exponent.

        Raises:
            ValueError: If the FDUs matching regex pattern for dimensions without exponent is empty or invalid.
        """
        if not value or not isinstance(value, str):
            _msg = "FDUs matching regex pattern for dimensions without exponent must be a non-empty string. "
            _msg += f"Provided: {value}"
            raise ValueError(_msg)
        self._fdu_no_pow_regex = value

    @property
    def fdu_sym_regex(self) -> str:
        """*fdu_sym_regex* property to get the FDUs matching regex pattern for dimensions in Sympy symbolic processor.

        Returns:
            str: FDUs matching regex pattern for dimensions in Sympy symbolic processor.
        """
        return self._fdu_sym_regex

    @fdu_sym_regex.setter
    def fdu_sym_regex(self, value: str) -> None:
        """*fdu_sym_regex* property to set the FDUs matching regex pattern for dimensions in Sympy symbolic processor.

        Args:
            value (str): FDUs matching regex pattern for dimensions in Sympy symbolic processor.

        Raises:
            ValueError: If the FDUs matching regex pattern for dimensions in Sympy symbolic processor is empty or invalid.
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
