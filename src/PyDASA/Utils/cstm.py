# -*- coding: utf-8 -*-
"""
Module to manage the regex patterns for dimension validation for the Fundamental Dimensional Unit (FDU) in *PyDASA*.
"""

# native python modules
# import dataclass for defining the node class
from typing import Optional, List, Generic
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
from Src.PyDASA.Utils.cfg import CSTM_FDU_PREC_LT
from Src.PyDASA.Utils.cfg import CSTM_FDU_REGEX
from Src.PyDASA.Utils.cfg import CSTM_POW_REGEX
from Src.PyDASA.Utils.cfg import CSTM_NO_POW_REGEX
from Src.PyDASA.Utils.cfg import CSTM_FDU_SYM_REGEX

# checking custom modules
assert error
assert T


@dataclass
class RegexManager(Generic[T]):
    """Manages the selection, validation, use, and retrieval of regex patterns for dimensional analysis in *PyDASA*.
    """

    # Custom flag to indicate if a custom regex is being used
    # :attr: custom
    custom: bool = False
    """
    TODO complete docstring
    """

    # FDUs precedence list for the regex
    # :attr: _fdu_precedence_lt
    _fdu_precedence_lt: List[str] = field(default_factory=lambda: FDU_PREC_LT)
    """
    TODO complete docstring
    """

    # FDUs matching regex pattern
    # :attr: _fdu_regex
    _fdu_regex: str = field(default_factory=lambda: DFLT_FDU_REGEX)
    """
    TODO complete docstring
    """

    # FDUs matching regex pattern for dimensions with exponent
    # :attr: _fdu_pow_regex
    _fdu_pow_regex: str = field(default_factory=lambda: DFLT_POW_REGEX)
    """
    TODO complete docstring
    """

    # FDUs matching regex pattern for dimensions without exponent
    # :attr: _fdu_no_pow_regex
    _fdu_no_pow_regex: str = field(default_factory=lambda: DFLT_NO_POW_REGEX)
    """
    TODO complete docstring
    """

    # FDUs matching regex pattern for dimensions in Sympy symbolic processor
    # :attr: _fdu_sym_regex
    _fdu_sym_regex: str = field(default_factory=lambda: DFLT_FDU_SYM_REGEX)
    """
    TODO complete docstring
    """

    def __post_init__(self) -> None:
        """__post_init__ _summary_

        Raises:
            ValueError: _description_
        """
        # TODO add docstring
        # check for valid dimensions
        if self.custom:
            print("Custom regex patterns are being used.")

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

    def update_global_vars(self) -> None:
        """update_global_vars _summary_
        """
        # TODO add docstring
        # FIXME this is not updating the global variables in the module!!!
        global CSTM_FDU_PREC_LT
        global CSTM_FDU_REGEX
        global CSTM_POW_REGEX
        global CSTM_NO_POW_REGEX
        global CSTM_FDU_SYM_REGEX

        # update global variables with the custom regex patterns
        CSTM_FDU_PREC_LT = self._fdu_precedence_lt
        CSTM_FDU_REGEX = self._fdu_regex
        CSTM_POW_REGEX = self._fdu_pow_regex
        CSTM_NO_POW_REGEX = self._fdu_no_pow_regex
        CSTM_FDU_SYM_REGEX = self._fdu_sym_regex
