from Src.PyDASA.DataStructs.Lists.arlt import ArrayList
from Src.PyDASA.DataStructs.Lists.sllt import SingleLinkedList
from Src.PyDASA.DataStructs.Tables.htme import MapEntry
from Src.PyDASA.DataStructs.Tables.scht import SeparateChainingTable
from Src.PyDASA.DataStructs.Tables.scht import Bucket

# for FDU regex management
from Src.PyDASA.Utils.cstm import RegexManager
# default regex for FDU
from Src.PyDASA.Utils.cfg import FDU_PREC_LT
from Src.PyDASA.Utils.cfg import DFLT_FDU_REGEX
from Src.PyDASA.Utils.cfg import DFLT_POW_REGEX
from Src.PyDASA.Utils.cfg import DFLT_NO_POW_REGEX
from Src.PyDASA.Utils.cfg import DFLT_FDU_SYM_REGEX

# custom regex for FDU
from Src.PyDASA.Utils.cfg import CSTM_FDU_PREC_LT
from Src.PyDASA.Utils.cfg import CSTM_FDU_REGEX
from Src.PyDASA.Utils.cfg import CSTM_POW_REGEX
from Src.PyDASA.Utils.cfg import CSTM_NO_POW_REGEX
from Src.PyDASA.Utils.cfg import CSTM_FDU_SYM_REGEX


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
                  _fdu_precedence_lt=["T", "D", "C"],)
print(rm)
rm.update_global_vars()

print("\n==== Default Regex ====")
print("\tFDU_PREC_LT:", FDU_PREC_LT)
print("\tDFLT_FDU_REGEX:", DFLT_FDU_REGEX)
print("\tDFLT_POW_REGEX:", DFLT_POW_REGEX)
print("\tDFLT_NO_POW_REGEX:", DFLT_NO_POW_REGEX)
print("\tDFLT_FDU_SYM_REGEX:", DFLT_FDU_SYM_REGEX)

print("\n==== Custom Regex ====")
print("\tCSTM_FDU_PREC_LT:", CSTM_FDU_PREC_LT)
print("\tCSTM_FDU_REGEX:", CSTM_FDU_REGEX)
print("\tCSTM_POW_REGEX:", CSTM_POW_REGEX)
print("\tCSTM_NO_POW_REGEX:", CSTM_NO_POW_REGEX)
print("\tCSTM_FDU_SYM_REGEX:", CSTM_FDU_SYM_REGEX)
