# impots for the file
from dataclasses import dataclass, field
from typing import Optional
from collections import OrderedDict
import re


# global variables
# traditional dimensional unit dictionary
FDU_DICT = {
    "L": "Length",
    "M": "Mass",
    "T": "Time",
    # "thermodynamic temperature": "Θ",
}

# software architecture fundamental dimensional unit dictionary
SAFDU_DICT = {
    "A": "Abstraction",
    "D": "Data",
    "T": "Time",
    # "Space": "S",
}

# valid domains for the dimensional unit class
VALID_DOMAINS_LT = [
    "physical",
    "logical",
    # "software architecture",
    ]

# regex for the dimensional unit class
FDU_REGEX = r"([LMT]?)(\^?-?\d+)?"
SAFDU_REGEX = r"([ADT]?)(\^?-?\d+)?"

OP_REGEX = r"[\^\+"

# err msg for the dimensional unit class
DOM_ERR_MSG = f"the valid domains are: {VALID_DOMAINS_LT}"
FDU_ERR_MSG = f"the valid dimensions are: {FDU_DICT.keys()}"
SAFDU_ERR_MSG = f"the valid dimensions are: {SAFDU_DICT.keys()}"
DIM_ERR_MSG = "Invalid dimensions and exponents!, both must be the same length"


# fundamenta dimensional unit class
@dataclass
class Dimensions:
    domain: str = VALID_DOMAINS_LT[0]
    expression: Optional[str] = ""
    dimensions: Optional[list[str]] = field(default_factory=list)
    exponents: Optional[list[int]] = field(default_factory=list)
    dvector: Optional[list[tuple[str, int]]] = field(default_factory=list)

    def __post_init__(self):
        # check if the domain is valid
        if self.domain not in VALID_DOMAINS_LT:
            err_msg = f"{self.domain} is invalid, " + DOM_ERR_MSG
            raise ValueError(err_msg)
        # check if the expression is valid in  the physical domain
        # precondition dimensios and exponents are not None
        c_dims = (self.dimensions is not None)
        if self.domain == "physical" and c_dims:
            self._check_physical()
        # check if the expression is valid in the logical domain
        elif self.domain == "logical" and c_dims:
            self._check_logical()
        # check if the dimensions and exponents are None
        if self.dimensions is None:
            self.dimensions = list()
        if self.exponents is None:
            self.exponents = list()
        if self.dvector is None:
            self.dvector = list()

    def _check_physical(self):
        # check if the expression is valid in the physical domain
        if self.dimensions is not None:
            for dimension in self.dimensions:
                if dimension not in FDU_DICT.keys():
                    err_msg = "Invalid dimension!, " + FDU_ERR_MSG
                    raise ValueError(err_msg)
        else:
            raise ValueError("Dimensions is None")
        self._check_exponents_n_dimensions()

    def _check_logical(self):
        # check if the expression is valid in the logical domain
        if self.dimensions is not None:
            for dimension in self.dimensions:
                if dimension not in SAFDU_DICT.keys():
                    err_msg = "Invalid dimension!, " + SAFDU_ERR_MSG
                    raise ValueError(err_msg)
        else:
            raise ValueError("Dimensions is None")
        self._check_exponents_n_dimensions()

    def _check_exponents_n_dimensions(self):
        # check if the exponents is valid in the physical domain
        if self.exponents is not None:
            for exponents in self.exponents:
                if not isinstance(exponents, int):
                    err_msg = "Invalid exponents!, " + FDU_ERR_MSG
            # check if the dimensions and exponentss are the same length
            if len(self.dimensions) != len(self.exponents):
                err_msg = f"Dimension lenght: {len(self.dimensions)}"
                err_msg += f"and exponents lenght: {len(self.exponents)}"
                err_msg += "are not the same, " + DIM_ERR_MSG
                raise ValueError(err_msg)
        else:
            raise ValueError("Exponents is None")

    def _select_working_dimensions(self):
        working_dims = None
        if self.domain == VALID_DOMAINS_LT[0]:      # physical
            working_dims = list(FDU_DICT.keys())
        elif self.domain == VALID_DOMAINS_LT[1]:    # logical
            working_dims = list(SAFDU_DICT.keys())
        else:
            raise ValueError(f"Invalid domain: {self.domain}")
        return working_dims

    def _select_regex(self):
        work_regex = None
        if self.domain == VALID_DOMAINS_LT[0]:      # physical
            work_regex = FDU_REGEX
        elif self.domain == VALID_DOMAINS_LT[1]:    # logical
            work_regex = SAFDU_REGEX
        else:
            raise ValueError(f"Invalid domain: {self.domain}")
        return work_regex

    def _safe_str_to_int(self, exp: str):
        try:
            ans = int(exp)
            return ans
        except ValueError as err:
            err_msg = f"Convert exponent failed: {exp}, error: {err}"
            raise ValueError(err_msg)

    def complement_expression(self):

        # select working dimensions
        working_dims = self._select_working_dimensions()

        # find missing dimensions
        missing_dims = list()
        for dimension in working_dims:
            if dimension not in self.dimensions:
                missing_dims.append(dimension)

        # add missing dimensions
        for dimension in missing_dims:
            vector = (dimension, 0)
            self.dvector.append(vector)

        # sort and update the dimensional vector
        dvector = sorted(self.dvector, key=lambda x: x[0])
        self.dvector = dvector
        # # update the dimensions and exponents
        self.dimensions = [vector[0] for vector in self.dvector]
        self.exponents = [vector[1] for vector in self.dvector]

    def parse_expression(self, expression: str = None):
        try:
            # parse the expression
            # if there is a new expression
            if expression is not None:
                self.expression = expression
            # uppercase the expression
            self.expression = self.expression.upper()
            # parse the expression according to the domain
            matches = None
            # select working dimensions
            working_dims = self._select_working_dimensions()
            # select working regex
            work_regex = self._select_regex()
            # find the matches
            matches = re.findall(work_regex, self.expression)
            # remove empty matches
            matches = [match for match in matches if match[0] != ""]
            # remove repeated matches
            matches = OrderedDict(matches).items()

            # iterate over the matches
            for match in matches:
                # divide the match in dimension and exponent
                dimension, exponent = match
                # validate the dimension are correct
                if dimension not in working_dims:
                    err_msg = f"Invalid dimension: {dimension}, "
                    err_msg += f"the valid dimensions are: {working_dims}"
                    raise ValueError(err_msg)
                # validate the exponents are valid
                if isinstance(exponent, str):
                    # if the str is empty or None
                    # print("entre al instance str")
                    if exponent in (""):
                        # print("entre al exponent in (, None)")
                        exponent = 1
                    elif len(exponent) >= 1:
                        exponent = exponent.replace("^", "")
                        exponent = self._safe_str_to_int(exponent)

                self.dimensions.append(dimension)
                self.exponents.append(exponent)
                vector = (dimension, exponent)
                self.dvector.append(vector)
            # complement the expression with the missing dimensions
            self.complement_expression()
        except Exception as err:
            raise err


# main for the class
if __name__ == "__main__":
    test_domain = [
        "physical",
        "logical",
    ]

    test_re_lt = [
        "L",
        "LM",
        "LMT",
        "L",
        "L^2",
        "L^-2",
        "L^2M^-1",
        "L^2M^-1T^2",
        "L*M",
        "L^2*M^-1",
        "L^2*M^-1*T^2",
        "L^2*M^3*T^-2",
        "LT^-2"
    ]

    # print(test_dim.__repr__())
    for expression in test_re_lt:
        test_dim = Dimensions("physical")
        print(f"expression: {expression}")
        test_dim.parse_expression(expression)
        print(test_dim.__repr__(), "\n")
