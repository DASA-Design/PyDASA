# -*- coding: utf-8 -*-
"""
Module **RegexManager** manages the regular expressions (regex) for validating the Fundamental Dimensional Unit (FDU) in *PyDASA*. It use a default or traditional dimensional system; plus, an optional custom regex for the user to define their own FDUs.

*IMPORTANT:* This code and its specifications for Python are based on the theory and subject developed by the following authors/books:

    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""

# native python modules
# import dataclass for defining the node class
from typing import List, Generic
from dataclasses import dataclass, field
# import modules for defining the MapEntry type

# custom modules
# generic error handling and type checking
from Src.PyDASA.Utils.err import error_handler as error
from Src.PyDASA.Utils.dflt import T

# importing PyDASA's default regex for FDU
from Src.PyDASA.Utils.cfg import FDU_PREC_LT
from Src.PyDASA.Utils.cfg import DFLT_FDU_REGEX
from Src.PyDASA.Utils.cfg import DFLT_POW_REGEX
from Src.PyDASA.Utils.cfg import DFLT_NO_POW_REGEX
from Src.PyDASA.Utils.cfg import DFLT_FDU_SYM_REGEX

# importing PyDASA's custom regex for FDU
# using the 'as' allows shared variable edition
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
            - `_fdu_precedence_lt`: A list of strings representing the FDUs precedence list.
            - `_fdu_regex`: A string representing the FDUs matching regex pattern.
            - `_fdu_pow_regex`: A string representing the FDUs matching regex pattern for dimensions with exponent.
            - `_fdu_no_pow_regex`: A string representing the FDUs matching regex pattern for dimensions without exponent.
            - `_fdu_sym_regex`: A string representing the FDUs matching regex pattern for dimensions in Sympy symbolic processor.
        All private attributes are initialized with default, use the @property decorator to access and validate their user input.
    """

    # Custom flag to indicate if a custom regex is being used
    # :attr: custom
    custom: bool = False
    """
    Custom flag to indicate if a custom FDU regex is being used. If True, the user can define their own FDUs and regex patterns.
    """

    # FDUs precedence list for the regex
    # :attr: _fdu_precedence_lt
    _fdu_precedence_lt: List[str] = field(default_factory=lambda: FDU_PREC_LT)
    """
    The FDU's precedence list for the regex. It is a list of strings with the FDUs in precedence order for creating the dimensional matrix.
    """

    # FDUs matching regex pattern
    # :attr: _fdu_regex
    _fdu_regex: str = field(default_factory=lambda: DFLT_FDU_REGEX)
    """
    Main FDU regex pattern for matching FDUs in *PyDASA*. It is a string for matching FDUs from a formula or parameter into the dimensional matrix.
    """

    # FDUs matching regex pattern for dimensions with exponent
    # :attr: _fdu_pow_regex
    _fdu_pow_regex: str = field(default_factory=lambda: DFLT_POW_REGEX)
    """
    Regex pattern for matching FDUs with exponents in *PyDASA*. It is a string  for matching FDUs with exponents from a formula or parameter into the dimensional matrix.
    """

    # FDUs matching regex pattern for dimensions without exponent
    # :attr: _fdu_no_pow_regex
    _fdu_no_pow_regex: str = field(default_factory=lambda: DFLT_NO_POW_REGEX)
    """
    Regex pattern for matching FDUs without exponents in *PyDASA*. It is a string for matching FDUs without exponents from a formula or parameter into the dimensional matrix.

    IMPORTAN: This is the ONLY regex that doesnt change with the custom regex.
    """

    # FDUs matching regex pattern for dimensions in Sympy symbolic processor
    # :attr: _fdu_sym_regex
    _fdu_sym_regex: str = field(default_factory=lambda: DFLT_FDU_SYM_REGEX)
    """
    Regex pattern for matching FDUs in Sympy symbolic processor. It is a string for matching FDUs in Latex/str format into Python's Sympy symbolic processor for the dimensional matrix.
    """

    def __post_init__(self) -> None:
        """*__post_init__* method to configure the the *RegexManager* instance after initialization.

        Raises:
            ValueError: If the FDUs precedence list is empty or contains invalid characters.
        """
        # check if the user is using custom regex
        if self.custom:
            self.setup_cstm_regex()

    def setup_cstm_regex(self) -> None:
        """*setup_cstm_regex()* setup the custom regex patterns for FDUs in *PyDASA* based on the user-defined FDU precedence list.

        This method is called after the constructor of the *RegexManager* class if the user has defined desired.
        """
        # compile the custom regex patterns for FDUs
        self._compile_cstm_regex()
        # update the global variables with the custom regex patterns
        self._update_global_vars()

    def _compile_cstm_regex(self) -> None:
        """*_compile_cstm_regex()* compile custom regex patterns for FDUs in *PyDASA* based on the user-defined FDU precedence list.
        """
        # dimensional precedence list
        # check for valid custom dimensional precedence list
        self.fdu_precedence_lt = self._fdu_precedence_lt

        # compile dimensional regex patterns
        self._fdu_regex = rf"^[{''.join(self.fdu_precedence_lt)}](\^-?\d+)?(\*[{''.join(self.fdu_precedence_lt)}](?:\^-?\d+)?)*$"
        # check for valid dimensional custom regex
        self.fdu_regex = self._fdu_regex

        # compile dimensional regex patterns with exponent
        self._fdu_pow_regex = DFLT_POW_REGEX
        # check for valid dimensional custom regex
        self.fdu_pow_regex = self._fdu_pow_regex

        # compile dimensional regex patterns without exponent
        self._fdu_no_pow_regex = rf"[{''.join(self.fdu_precedence_lt)}](?!\^)"
        # check for valid dimensional custom regex
        self.fdu_no_pow_regex = self._fdu_no_pow_regex

        # compile dimensional regex patterns in Sympy symbolic processor
        self._fdu_sym_regex = rf"[{''.join(self.fdu_precedence_lt)}]"
        # check for valid dimensional custom regex
        self.fdu_sym_regex = self._fdu_sym_regex

    def _update_global_vars(self) -> None:
        """*update_global_vars()* updates the global variables with the custom regex patterns compiled with the FDU's precedence list of the user.
        """
        # update global variables with the custom regex patterns
        config.CSTM_FDU_PREC_LT = self._fdu_precedence_lt
        config.CSTM_FDU_REGEX = self._fdu_regex
        config.CSTM_POW_REGEX = self._fdu_pow_regex
        config.CSTM_NO_POW_REGEX = self._fdu_no_pow_regex
        config.CSTM_FDU_SYM_REGEX = self._fdu_sym_regex

    @property
    def fdu_precedence_lt(self) -> List[str]:
        """*fdu_precedence_lt* property to get the FDUs precedence list.

        Returns:
            List[str]: FDUs precedence list.
        """
        return self._fdu_precedence_lt

    @fdu_precedence_lt.setter
    def fdu_precedence_lt(self, value: List[str]) -> None:
        """*fdu_precedence_lt* property to set the FDUs precedence list.

        Args:
            value (List[str]): FDUs precedence list.

        Raises:
            ValueError: If the FDUs precedence list is empty or contains invalid characters.
        """
        if not value or not all(isinstance(i, str) for i in value):
            _msg = "FDUs precedence list must be a non-empty list of strings. "
            _msg += f"Provided: {value}"
            raise ValueError(_msg)
        self._fdu_precedence_lt = value

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
