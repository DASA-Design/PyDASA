# -*- coding: utf-8 -*-
# expose imports
# exposing analytics packages
# TODO conversion still in development
# from .pydasa.analysis.conversion import UnitConverter
from .analysis.scenario import Sensitivity
from .analysis.simulation import MonteCarlo

# exposing data pi-theorem packages
from .dimensional.buckingham import Coefficient

# exposing core packages
# TODO measurement still in development
from .dimensional.fundamental import Dimension
# from .pydasa.core.measurement import Unit
from .elements.parameter import Variable

# expose datastructures, its for internal use only but docs need it
# lists
from .structs.lists.arlt import ArrayList
from .structs.lists.sllt import SingleLinkedList
from .structs.lists.ndlt import Node, SLNode, DLNode
# from .structs.lists.dllt import DoubleLinkedList

# tables
from .structs.tables.htme import MapEntry
from .structs.tables.scht import Bucket, SCHashTable

# exposing dimensional packages
# TODO domain still in development
# from .pydasa.dimensional.domain import MeasurementSys
from .dimensional.framework import Schema
from .dimensional.model import Matrix

# exposing data handling packages
# TODO phenomena still in development
from .tasks.influence import SensitivityHandler
# from .pydasa.handlers.phenomena import DimSolver
from .tasks.practical import MonteCarloHandler

# exposing utility packages
# most utils are private
from .core.io import load, save
# importing queue factory function
# from .utils.queues import Queue
# from .utils.helpers import gfactorial

# asserting all imports
# assert UnitConverter
assert Sensitivity
assert MonteCarlo
assert Coefficient
assert Dimension
# assert Unit
assert Variable

assert ArrayList
assert SingleLinkedList
assert Node
assert SLNode
assert DLNode
assert MapEntry
assert Bucket
assert SCHashTable

# assert MeasurementSys
assert Schema
assert Matrix

assert SensitivityHandler
# assert DimSolver
assert MonteCarloHandler

assert load
assert save
# assert Queue
# assert gfactorial

# Optionally, define __all__ for wildcard imports
__all__ = [
    "Dimension",
    "Variable",
    "Schema",
    "Matrix",
    "Sensitivity",
    "SensitivityHandler",
    "MonteCarlo",
    "MonteCarloHandler",
]
