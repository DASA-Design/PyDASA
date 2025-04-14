
# Custom modules
# PyDASA modules
# data structures modules
# lists modules
from Src.PyDASA.DataStructs.Lists.arlt import ArrayList
from Src.PyDASA.DataStructs.Lists.sllt import SingleLinkedList
from Src.PyDASA.DataStructs.Lists.ndlt import Node, SLNode, DLNode
# hash tables modules
from Src.PyDASA.DataStructs.Tables.htme import MapEntry
from Src.PyDASA.DataStructs.Tables.scht import SCHashTable
from Src.PyDASA.DataStructs.Tables.scht import Bucket


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
from Src.PyDASA.Models.dim import DimensionalModel

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
