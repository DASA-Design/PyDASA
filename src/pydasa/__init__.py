# -*- coding: utf-8 -*-
# expose imports
# exposing analytics modules
from .analysis.scenario import Sensitivity
from .analysis.simulation import MonteCarlo

# TODO conversion still in development
# from .context.conversion import UnitStandarizer
# from .context.system import MeasureSystem
# from .context.measurements import Unit

# exposing pi-theorem/dimensional analysis modules
from .dimensional.buckingham import Coefficient
from .dimensional.fundamental import Dimension
from .dimensional.framework import Schema
from .dimensional.model import Matrix

# exposing core modules
# exposing basic elements/variables modules
from .elements.parameter import Variable
# exposing parser/io modules
from .core.io import load, save

# exposing custom data structure modules
# TODO measurement still in development
# lists
from .structs.lists.arlt import ArrayList
from .structs.lists.sllt import SingleLinkedList
from .structs.lists.ndlt import Node, SLNode, DLNode
# from .structs.lists.dllt import DoubleLinkedList
# tables
from .structs.tables.htme import MapEntry
from .structs.tables.scht import Bucket, SCHashTable

# exposing validation, error and decorator modules
# exposing workflow modules
from .workflows.influence import SensitivityAnalysis
from .workflows.practical import MonteCarloSimulation
# from .workflows.phenomena import Solver

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

assert SensitivityAnalysis
# assert DimSolver
assert MonteCarloSimulation

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
    "SensitivityAnalysis",
    "MonteCarlo",
    "MonteCarloSimulation",
]
