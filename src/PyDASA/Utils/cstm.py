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
import inspect

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

# import FDU working regex, use 'as' to allow shared variable edition
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
        """__post_init__ _summary_
        """
        try:
            # configure working FDUs precedence list
            self.setup_precedence()
            # set up regex patterns for FDUs
            self.setup_regex()
        except Exception as err:
            self._error_handler(err)

    def setup_precedence(self) -> None:
        """setup_precedence _summary_

        Raises:
            ValueError: _description_
        """
        # if the framework is supported
        if self.fwk in FDU_FWK_DT and self.fwk != "CUSTOM":
            # set up the default precedence list
            self.fdu_prec_lt = self._get_default_precedence()
        # if the framework is user-defined
        elif self.fwk == "CUSTOM" and self.fdu_prec_lt:
            # Use the provided custom precedence list
            # self.fdu_prec_lt = self._fdu_prec_lt
            pass
        # otherwise, raise an error
        else:
            _msg = f"Invalid Framework: {self.fwk}. "
            _msg += "Must be one of the following: "
            _msg += f"{', '.join(FDU_FWK_DT.keys())}."
            raise ValueError(_msg)

    def _get_default_precedence(self) -> List[str]:
        # map for easy access to the FDUs
        _frk_dt = {
            "PHYSICAL": PHY_FDU_PREC_DT,
            "COMPUTATION": COMPU_FDU_PREC_DT,
            "DIGITAL": DIGI_FDU_PREC_DT,
        }
        if self.fwk in _frk_dt:
            _prec_lt = list(_frk_dt[self.fwk].keys())
            return _prec_lt
        # otherwise, raise an error
        _msg = f"Invalid Framework: {self.fwk}. "
        _msg += "Must be one of the following: "
        _msg += f"{', '.join(FDU_FWK_DT.keys())}."
        raise ValueError(_msg)

    def _setup_custom_precedence(self) -> None:
        # FIXME unnecessary function, remove it later
        # check for valid custom FDU precedence list
        self.fdu_prec_lt = self._fdu_prec_lt

    def setup_regex(self) -> None:
        """*_setup_wkng_regex()* Initializes the *RegexManager* instance and sets up regex patterns for FDUs.

        Raises:
            ValueError: If the CUSTOM framework lacks a precedence list or if the framework is invalid.
        """
        # check for errors on the FDU precedence list
        # if not self.fdu_prec_lt or not all(isinstance(i, str) for i in self.fdu_prec_lt):
        if not all(isinstance(i, str) for i in self.fdu_prec_lt):
            _msg = "FDUs precedence list must be a non-empty list of strings. "
            _msg += f"Provided: {self.fdu_prec_lt}"
            raise ValueError(_msg)
        # compile the regex patterns for FDUs
        self._compile_regex()

    def _compile_regex(self) -> None:
        """*_compile_regex()* compiles regex patterns for FDUs based on the provided precedence list. Creates regex patterns for:
            - FDUs with exponents.
            - FDUs without exponents.
            - FDUs in Sympy symbolic processor.
        """
        # FDU precedence list
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
        """*update_global_regex()* Updates the global *config* variables with custom regex patterns for FDUs.
        """
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
            _msg = f"Invalid Framework: {value}. "
            _msg += "Must be one of the following: "
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

    def _error_handler(self, err: Exception) -> None:
        """*_error_handler()* to process the context (package/class), function name (method), and the error (exception) that was raised to format a detailed error message and traceback.

        Args:
            err (Exception): Python raised exception.
        """
        _context = self.__class__.__name__
        _function_name = inspect.currentframe().f_code.co_name
        error(_context, _function_name, err)

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
