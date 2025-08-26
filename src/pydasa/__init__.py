# -*- coding: utf-8 -*-
# expose imports
# exposing analytics packages
# TODO conversion still in development
# from .pydasa.analysis.conversion import UnitConverter
from .analysis.scenario import DimSensitivity
from .analysis.simulation import MonteCarloSim

# exposing data pi-theorem packages
from .buckingham.vashchy import Coefficient

# exposing core packages
# TODO measurement still in development
from .core.fundamental import Dimension
# from .pydasa.core.measurement import Unit
from .core.parameter import Variable

# expose datastructures, its for internal use only but docs need it
# lists
from .datastructs.lists.arlt import ArrayList
from .datastructs.lists.sllt import SingleLinkedList
from .datastructs.lists.dllt import Node, SLNode, DLNode
# from .datastructs.lists.ndlt import DoubleLinkedList

# tables
from .datastructs.tables.htme import MapEntry
from .datastructs.tables.scht import Bucket, SCHashTable

# exposing dimensional packages
# TODO domain still in development
# from .pydasa.dimensional.domain import MeasurementSys
from .dimensional.framework import DimScheme
from .dimensional.model import DimMatrix

# exposing data handling packages
# TODO phenomena still in development
from .handlers.influence import SensitivityHandler
# from .pydasa.handlers.phenomena import DimSolver
from .handlers.practical import MonteCarloHandler

# exposing utility packages
# most utils are private
from .utils.io import load, save
# importing queue factory function
from .utils.queues import Queue

# asserting all imports
# assert UnitConverter
assert DimSensitivity
assert MonteCarloSim
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
assert DimScheme
assert DimMatrix

assert SensitivityHandler
# assert DimSolver
assert MonteCarloHandler

assert load
assert save
assert Queue

# Optionally, define __all__ for wildcard imports
__all__ = [
    "Dimension",
    "Variable",
    "DimScheme",
    "DimMatrix",
    "DimSensitivity",
    "SensitivityHandler",
    "MonteCarloSim",
    "MonteCarloHandler",
]
