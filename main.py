from Src.PyDASA.DataStructs.Lists.arlt import ArrayList
from Src.PyDASA.DataStructs.Lists.sllt import SingleLinkedList
from Src.PyDASA.DataStructs.Tables.htme import MapEntry
from Src.PyDASA.DataStructs.Tables.scht import SeparateChainingTable
from Src.PyDASA.DataStructs.Tables.scht import Bucket


from Src.PyDASA.Units.fdu import FDU
from Src.PyDASA.Units.parameter import Parameter, Variable


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
print(p)

v = Variable()
print(v)
