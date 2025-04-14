from Src.PyDASA.DataStructs.Lists.arlt import ArrayList
from Src.PyDASA.DataStructs.Lists.sllt import SingleLinkedList
from Src.PyDASA.DataStructs.Lists.ndlt import Node, SLNode, DLNode
from Src.PyDASA.DataStructs.Tables.htme import MapEntry
from Src.PyDASA.DataStructs.Tables.scht import SeparateChainingTable
from Src.PyDASA.DataStructs.Tables.scht import Bucket

# for FDU regex management
from Src.PyDASA.Utils.cstm import RegexManager
# complete module withe the FDU's regex
from Src.PyDASA.Utils import cfg as config

from Src.PyDASA.Units.fdu import FDU
from Src.PyDASA.Units.params import Parameter, Variable


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
    {"_id": 1, "_data": 1},
    {"_id": 2, "_data": 2},
    {"_id": 3, "_data": 3},
)
c = Bucket(iodata=_data)
print(c, "\n")
print(c.get(1), "\n")
print(c.get(2), "\n")

a = ArrayList(iodata=_data,
              cmp_function=test_cmp)
print(a, "\n")

ht = SeparateChainingTable(iodata=_data)
print(ht, "\n")

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
    {"_id": 1, "_sym": "T", "_fwk": "PHYSICAL", "_prec": 0, "desription": "Time"},
    {"_id": 5, "_sym": "M", "_fwk": "PHYSICAL", "_prec": 4, "description": "Mass"},
    {"_id": 4, "_sym": "L", "_fwk": "PHYSICAL", "_prec": 3, "description": "Longitude"},
]
