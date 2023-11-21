# main imports for the parameters module
from dataclasses import dataclass


# dimensional parameter class
@dataclass
class Parameter:
    name: str
    symbol: str
    dimensions: str
    units: str
