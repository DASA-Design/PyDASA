from Src.PyDASA.DataStructs.Lists.arlt import ArrayList
from Src.PyDASA.DataStructs.Lists.sllt import SingleLinkedList
from Src.PyDASA.DataStructs.Tables.htme import MapEntry
from Src.PyDASA.DataStructs.Tables.scht import SeparateChainingTable
from Src.PyDASA.DataStructs.Tables.scht import Bucket

# for FDU regex management
from Src.PyDASA.Utils.cstm import RegexManager
# complete module withe the FDU's regex
from Src.PyDASA.Utils import cfg as config

from Src.PyDASA.Units.fdu import FDU
from Src.PyDASA.Units.params import Parameter, Variable


a = ArrayList(iodata=[1, 2, 3])
# print(a, "\n\n\n\n")

b = SingleLinkedList(iodata=[1, 2, 3])
# print(b)

m = MapEntry()
# print(m)

c = Bucket()
# print(c)

ht = SeparateChainingTable()
# print(ht)

fdu = FDU()
# print(fdu)

p = Parameter()
# print(p)

v = Variable()
# print(v)

rm = RegexManager(custom=True,
                  _fdu_prec_lt=["T", "D", "C"],)
print(rm)
# rm.update_global_vars()

p1 = Parameter(name="U_1",
               description="Service Rate",
               _sym="U_{1}",
               _fwk="DIGITAL",
               _idx=1,
               _cat="INPUT",
               _units="kPa",
               _dims="C*T^-1",)
print(p1)

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

# print(p1)
