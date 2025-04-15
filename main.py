
# Custom modules
# PyDASA modules
# data structures modules
# lists modules
from Src.PyDASA.DStructs.Lists.arlt import ArrayList
from Src.PyDASA.DStructs.Lists.sllt import SingleLinkedList
from Src.PyDASA.DStructs.Lists.ndlt import Node, SLNode, DLNode
# hash tables modules
from Src.PyDASA.DStructs.Tables.htme import MapEntry
from Src.PyDASA.DStructs.Tables.scht import SCHashTable
from Src.PyDASA.DStructs.Tables.scht import Bucket


# dimensionl analysis modules
# config module
from Src.PyDASA.Utils import cfg as config
# FDU regex management
from Src.PyDASA.Utils.cstm import RegexManager
# FDU modules
from Src.PyDASA.Units.fdu import FDU
# Parameter and Variable modules
from Src.PyDASA.Units.params import Parameter, Variable
# Dimensional Matrix Modelling module
from Src.PyDASA.Models.dim import DimensionalModel, DimensionalAnalyzer

# for FDU regex management
# for Dimensional Analysisis modules
# complete module withe the FDU's regex


def test_cmp(a, b) -> int:
    """Test comparison function."""
    if a < b:
        print(f"{a} < {b}")
        return -1
    elif a == b:
        print(f"{a} == {b}")
        return 0
    elif a > b:
        print(f"{a} > {b}")
        return 1
    else:
        raise TypeError(f"Invalid comparison between {type(a)} and {type(b)}")


a = ArrayList(iodata=[1, 2, 3],
              cmp_function=test_cmp)
print(a, "\n")

b = SingleLinkedList(iodata=[1, 2, 3])
print(b, "\n")
print(b.last, b.first, b.get(0), "\n")
print(b.index_of(2), b.index_of(4), "\n")
# print(b.pop_first(), b, "\n")
# n = DLNode(_data=1)
# print(n, "\n")

# m = MapEntry()
# print(m, "\n")

c = Bucket()
print(c, "\n")
_data = (
    {"_idx": 1, "_data": 1},
    {"_idx": 2, "_data": 2},
    {"_idx": 3, "_data": 3},
)
c = Bucket(iodata=_data)
print(c, "\n")
print(c.get(1), "\n")
print(c.get(2), "\n")

a = ArrayList(iodata=_data,
              cmp_function=test_cmp)
print(a, "\n")

ht = SCHashTable(iodata=_data)
print(ht, "\n")

nd = Node(_data=1)
print(nd, "\n")

nd = SLNode(_data=1)
print(nd, "\n")

nd.next = Node(_data=2)
print(nd, "\n")

dlnd = DLNode(_data=1)
print(dlnd, "\n")
dlnd.next = Node(_data=2)
dlnd.prev = Node(_data=0)
print(dlnd, "\n")

mp = MapEntry(_key="U_1",
              _value=1.0,)
print(mp, "\n")


# default regex for FDU
print("\n==== Default Regex ====")
print("\tDFLT_FDU_PREC_LT:", config.DFLT_FDU_PREC_LT)
print("\tDFLT_FDU_REGEX:", config.DFLT_FDU_REGEX)
print("\tDFLT_POW_REGEX:", config.DFLT_POW_REGEX)
print("\tDFLT_NO_POW_REGEX:", config.DFLT_NO_POW_REGEX)
print("\tDFLT_FDU_SYM_REGEX:", config.DFLT_FDU_SYM_REGEX)

# custom regex for FDU
print("\n==== Custom Regex ====")
print("\tWKNG_DFLT_FDU_PREC_LT:", config.WKNG_FDU_PREC_LT)
print("\tWKNG_FDU_REGEX:", config.WKNG_FDU_REGEX)
print("\tWKNG_POW_REGEX:", config.WKNG_POW_REGEX)
print("\tWKNG_NO_POW_REGEX:", config.WKNG_NO_POW_REGEX)
print("\tWKNG_FDU_SYM_REGEX:", config.WKNG_FDU_SYM_REGEX)

fdu = FDU()
print(fdu, "\n")

p = Parameter()
print(p, "\n")

v = Variable()
print(v, "\n")

p1 = Parameter(name="U_1",
               description="Service Rate",
               _sym="U_{1}",
               _fwk="DIGITAL",
               _idx=1,
               _cat="INPUT",
               _units="kPa",
               _dims="M*T^-2*L^-1",)
print(p1, "\n")

fdu = FDU()
print(fdu, "\n")

rm = RegexManager(_fdu_prec_lt=["T", "D", "C"],)
print(rm, "\n")

rm = RegexManager(_fwk="DIGITAL",)
print(rm, "\n")

# rm.update_global_vars()

fdu_lt = [
    {"_idx": 0, "_sym": "T", "_fwk": "CUSTOM", "description": "Time~~~~~~~~~~~~~"},
    {"_idx": 1, "_sym": "M", "_fwk": "CUSTOM", "description": "Mass~~~~~~~~~~~~"},
    {"_idx": 2, "_sym": "L", "_fwk": "CUSTOM", "description": "Longitude~~~~~~~~"},
]

b = SingleLinkedList(iodata=fdu_lt)
print(b.first, "\n", b.last, "\n")
print(b, "\n")
for fdu in b:
    print(fdu, "\n")


DAModel = DimensionalModel(_fwk="CUSTOM",
                           _idx=0,
                           io_fdu=fdu_lt)
print(DAModel, "\n")

# custom regex for FDU
print("\n==== Custom Regex ====")
print("\tWKNG_DFLT_FDU_PREC_LT:", config.WKNG_FDU_PREC_LT)
print("\tWKNG_FDU_REGEX:", config.WKNG_FDU_REGEX)
print("\tWKNG_POW_REGEX:", config.WKNG_POW_REGEX)
print("\tWKNG_NO_POW_REGEX:", config.WKNG_NO_POW_REGEX)
print("\tWKNG_FDU_SYM_REGEX:", config.WKNG_FDU_SYM_REGEX)

# Planar Channel Flow with a Moving Wall
# u = f(y, d, U, P, v)
# u: fluid velocity
# y: distance from the wall
# d: distance from the wall to the center of the channel (diameter)
# U: velocity of the wall
# P: pressure drop across the channel
# v: kinematic viscosity of the fluid

dim_relevance_lt = [
    Parameter(_sym="u",
              _fwk="CUSTOM",
              name="Fluid Velocity",
              description="Fluid velocity in the channel",
              relevant=True,
              _idx=0,
              _cat="OUTPUT",
              _units="m/s",
              _dims="L*T^-1",),
    Parameter(_sym="y",
              _fwk="CUSTOM",
              name="Distance from the wall",
              description="Distance from the wall to the center of the channel",
              relevant=True,
              _idx=1,
              _cat="INPUT",
              _units="m",
              _dims="L",),
    Parameter(_sym="d",
              _fwk="CUSTOM",
              name="Channel diameter",
              relevant=True,
              description="Diameter of the channel",
              _idx=2,
              _cat="INPUT",
              _units="m",
              _dims="L",),
    Parameter(_sym="U",
              _fwk="CUSTOM",
              name="Velocity of the wall",
              relevant=True,
              description="Velocity of the fluid wall",
              _idx=3,
              _cat="INPUT",
              _units="m/s",
              _dims="L*T^-1",),
    Parameter(_sym="P",
              _fwk="CUSTOM",
              name="Channel Pressure Drop",
              relevant=True,
              description="Pressure drop across the channel",
              _idx=4,
              _cat="CONTROL",
              _units="Pa",
              _dims="T^-2*L^1",),
    Parameter(_sym="v",
              _fwk="CUSTOM",
              name="Fluid Viscosity",
              relevant=True,
              description="Kinematic viscosity of the fluid",
              _idx=5,
              _cat="CONTROL",
              _units="m^2/s",
              _dims="L^2*T^-1",),
    Parameter(_sym="g",
              _fwk="CUSTOM",
              name="Gravity",
              description="Acceleration due to gravity",
              _idx=6,
              _cat="CONTROL",
              _units="m/s^2",
              _dims="L*T^-2",),
    Parameter(_sym="f",
              _fwk="CUSTOM",
              name="Fluid Frequency",
              description="Fluid frequency",
              _idx=7,
              _cat="CONTROL",
              _units="Hz",
              _dims="T^-1",),
]

print(type(dim_relevance_lt))
print("Dimensional relevance of the parameters:")
for p in dim_relevance_lt:
    print(p)

fdu_lt = [
    {"_idx": 0, "_sym": "Tt", "_fwk": "CUSTOM", "description": "Time~~~~~~~!!!~~~~~~"},
    {"_idx": 1, "_sym": "Mm", "_fwk": "CUSTOM", "description": "Mass~~~~~!!!!!~~~~~~~"},
    {"_idx": 2, "_sym": "Ll", "_fwk": "CUSTOM", "description": "Longitude~~~!!!!!!!~~~~~"},
]

print("Setting parameters for the dimensional analysis")
DAModel.param_lt = dim_relevance_lt
print(len(DAModel.param_lt), DAModel.param_lt, "\n")
print("Setting the relevance list for dimensional analysis")
DAModel.relevance_lt = dim_relevance_lt
print(len(DAModel.relevance_lt), DAModel.relevance_lt, "\n")

print(DAModel, "\n")

print(DAModel._n_param, "\n")

DAnalysis = DimensionalAnalyzer(_fwk="CUSTOM",
                                _idx=0,
                                io_fdu=fdu_lt,
                                _fdu_ht=DAModel._fdu_ht,
                                _param_lt=DAModel.param_lt,
                                _relevance_lt=DAModel.relevance_lt,)
print(DAnalysis.relevance_lt, "\n")
print(len(DAnalysis.relevance_lt), "\n")
print(DAnalysis, "\n")
print(DAnalysis._wrk_fdu_lt, "\n")
print(DAnalysis._fdu_ht, "\n")

for relv in DAnalysis.relevance_lt:
    print("blaaaaa", relv.idx, relv.cat, relv.sym, relv.name)
print(DAnalysis.output, "\n")

DAnalysis.create_matrix()
DAnalysis.solve_matrix()
print(len(DAnalysis.pi_coef_lt), "\n")
for pi in DAnalysis.pi_coef_lt:
    print(pi, "\n")
