
# Custom modules
# PyDASA modules
# data structures modules
# lists modules
from new.pydasa.datastruct.lists.arlt import ArrayList
from new.pydasa.datastruct.lists.sllt import SingleLinkedList
from new.pydasa.datastruct.lists.ndlt import Node, SLNode, DLNode
# hash tables modules
from new.pydasa.datastruct.tables.htme import MapEntry
from new.pydasa.datastruct.tables.scht import SCHashTable
from new.pydasa.datastruct.tables.scht import Bucket


# dimensional analysis modules
# config module
from new.pydasa.utils import config

# FDU modules
from new.pydasa.core.fundamental import Dimension

# FDU regex management
from new.pydasa.dimensional.framework import DimFramework

# Variable and Variable modules
from new.pydasa.core.parameters import Variable

# Dimensional Matrix Modelling module
from new.pydasa.dimensional.model import DimMatrix

from new.pydasa.analysis.influence import Sensitivity



# TODO need to know where to put the new
# from new.pydasa.analysis.influence import SensitivityAnalysis

# for FDU regex management
# for Dimensional Analysis modules
# complete module with the FDU's regex

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
print("\tDFLT_FDU_RE:", config.DFLT_FDU_RE)
print("\tDFLT_POW_RE:", config.DFLT_POW_RE)
print("\tDFLT_NO_POW_RE:", config.DFLT_NO_POW_RE)
print("\tDFLT_FDU_SYM_RE:", config.DFLT_FDU_SYM_RE)

# custom regex for FDU
print("\n==== Custom Regex ====")
print("\tWKNG_DFLT_FDU_PREC_LT:", config.WKNG_FDU_PREC_LT)
print("\tWKNG_FDU_RE:", config.WKNG_FDU_RE)
print("\tWKNG_POW_RE:", config.WKNG_POW_RE)
print("\tWKNG_NO_POW_RE:", config.WKNG_NO_POW_RE)
print("\tWKNG_FDU_SYM_RE:", config.WKNG_FDU_SYM_RE)

fdu = Dimension()
print(fdu, "\n")

fdu = Dimension("Length",
                "Length of a physical quantity",
                1, "L", "PHYSICAL", "m")
print(fdu, "\n")


v = Variable()
print(v, "\n")

p1 = Variable(name="U_1",
              description="Service Rate",
              _sym="U_{1}",
              _fwk="DIGITAL",
              _idx=1,
              _cat="INPUT",
              _units="kPa",
              _dims="M*T^-2*L^-1",)
print(p1, "\n")

rm = DimFramework(_fwk="SOFTWARE",)
print(rm, "\n")


rm.update_global_config()
print(rm, "\n")

fdu_lt = [
    {"_idx": 0, "_sym": "M", "_fwk": "CUSTOM", "description": "Mass~~~~~~~~~~~~", "_unit": "kg", "name": "Mass"},
    {"_idx": 1, "_sym": "L", "_fwk": "CUSTOM", "description": "Longitude~~~~~~~~", "_unit": "m", "name": "Longitude"},
    {"_idx": 2, "_sym": "T", "_fwk": "CUSTOM", "description": "Time~~~~~~~~~~~~~", "_unit": "s", "name": "Time"},
]

rm = DimFramework(_fdus=fdu_lt, _fwk="CUSTOM")

rm.update_global_config()
print(rm, "\n")

b = SingleLinkedList(iodata=fdu_lt)
print(b.first, "\n", b.last, "\n")
print(b, "\n")
for fdu in b:
    print(fdu, "\n")


# DAModel = DimensionalModel(_fwk="CUSTOM",
#                            _idx=0,
#                            io_fdu=fdu_lt)
# print(DAModel, "\n")

# custom regex for FDU
print("\n==== Custom Regex ====")
print("\tWKNG_DFLT_FDU_PREC_LT:", config.WKNG_FDU_PREC_LT)
print("\tWKNG_FDU_RE:", config.WKNG_FDU_RE)
print("\tWKNG_POW_RE:", config.WKNG_POW_RE)
print("\tWKNG_NO_POW_RE:", config.WKNG_NO_POW_RE)
print("\tWKNG_FDU_SYM_RE:", config.WKNG_FDU_SYM_RE)

# Planar Channel Flow with a Moving Wall
# u = f(y, d, U, P, v)
# u: fluid velocity
# y: distance from the wall
# d: distance from the wall to the center of the channel (diameter)
# U: velocity of the wall
# P: pressure drop across the channel
# v: kinematic viscosity of the fluid

dim_relevance_lt = [
    Variable(_sym="\\miu",
             _varsym="miu",
             _fwk="CUSTOM",
             name="Fluid Velocity",
             description="Fluid velocity in the channel",
             relevant=True,
             _idx=0,
             _cat="OUTPUT",
             _units="m/s",
             _dims="L*T^-1",),
    Variable(_sym="y",
             _varsym="y",
             _fwk="CUSTOM",
             name="Distance from the wall",
             description="Distance from the wall to the center of the channel",
             relevant=True,
             _idx=1,
             _cat="INPUT",
             _units="m",
             _dims="L",),
    Variable(_sym="d",
             _varsym="d",
             _fwk="CUSTOM",
             name="Channel diameter",
             relevant=True,
             description="Diameter of the channel",
             _idx=2,
             _cat="INPUT",
             _units="m",
             _dims="L",),
    Variable(_sym="U",
             _varsym="U",
             _fwk="CUSTOM",
             name="Velocity of the wall",
             relevant=True,
             description="Velocity of the fluid wall",
             _idx=3,
             _cat="INPUT",
             _units="m/s",
             _dims="L*T^-1",),
    Variable(_sym="P",
             _varsym="P",
             _fwk="CUSTOM",
             name="Channel Pressure Drop",
             relevant=True,
             description="Pressure drop across the channel",
             _idx=4,
             _cat="CONTROL",
             _units="Pa",
             _dims="T^-2*L^1",),
    Variable(_sym="v",
             _varsym="v",
             _fwk="CUSTOM",
             name="Fluid Viscosity",
             relevant=True,
             description="Kinematic viscosity of the fluid",
             _idx=5,
             _cat="CONTROL",
             _units="m^2/s",
             _dims="L^2*T^-1",),
    Variable(_sym="g",
             _varsym="g",
             _fwk="CUSTOM",
             name="Gravity",
             description="Acceleration due to gravity",
             _idx=6,
             _cat="CONTROL",
             _units="m/s^2",
             _dims="L*T^-2",),
    Variable(_sym="f",
             _varsym="f",
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

# print("Setting parameters for the dimensional analysis")
# DAModel.param_lt = dim_relevance_lt
# print(len(DAModel.param_lt), DAModel.param_lt, "\n")
# print("Setting the relevance list for dimensional analysis")
# DAModel.relevance_lt = dim_relevance_lt
# print(len(DAModel.relevance_lt), DAModel.relevance_lt, "\n")

# print(DAModel, "\n")

# print(DAModel._n_param, "\n")

# DAnalysis = DimensionalAnalyzer(_fwk="CUSTOM",
#                                 _idx=0,
#                                 io_fdu=fdu_lt,
#                                 _fdu_mp=DAModel._fdu_mp,
#                                 _param_lt=DAModel.param_lt,
#                                 _relevance_lt=DAModel.relevance_lt,)
# # print(DAnalysis.relevance_lt, "\n")
# # print(len(DAnalysis.relevance_lt), "\n")
# # print(DAnalysis, "\n")
# # print(DAnalysis._wrk_fdu_lt, "\n")
# # print(DAnalysis._fdu_mp, "\n")

# for relv in DAnalysis.relevance_lt:
#     print("blaaaaa", relv.idx, relv.cat, relv.sym, relv.name)
# print(DAnalysis.output, "\n")

# DAnalysis.create_matrix()
# DAnalysis.solve_matrix()

# print(DAnalysis._pivot_cols, "\n")
# print(DAnalysis, "\n")

# for k, v in vars(DAnalysis).items():
#     print(f"{k}: {v}")
# print("\n")

# # print(len(DAnalysis.pi_coef_lt), "\n")
# for pi in DAnalysis.pi_coef_lt:
#     print(pi.sym, "=", pi.pi_expr, "\n")

# vars_lt = []
# extr_data_lt = [
#     {   # Fluid velocity
#         "_min": 0.0,
#         "_max": 15.0,
#         "_avg": 7.50,
#         "_std_units": "m/s",
#         "_std_min": 0.0,
#         "_std_max": 15.0,
#         "_std_avg": 7.50,
#         "_std_step": 0.1,
#     },
#     {   # Distance from the wall
#         "_min": 0.0,
#         "_max": 10.0,
#         "_avg": 5.0,
#         "_std_units": "m",
#         "_std_min": 0.0,
#         "_std_max": 10.0,
#         "_std_avg": 5.0,
#         "_std_step": 0.1,
#     },
#     {   # Channel diameter
#         "_min": 0.0,
#         "_max": 5.0,
#         "_avg": 2.5,
#         "_std_units": "m",
#         "_std_min": 0.0,
#         "_std_max": 5.0,
#         "_std_avg": 2.5,
#         "_std_step": 0.1,
#     },
#     {   # Velocity of the wall
#         "_min": 0.0,
#         "_max": 15.0,
#         "_avg": 7.50,
#         "_std_units": "m/s",
#         "_std_min": 0.0,
#         "_std_max": 15.0,
#         "_std_avg": 7.50,
#         "_std_step": 0.1,
#     },
#     {   # Pressure drop across the channel
#         "_min": 0.0,
#         "_max": 100000.0,
#         "_avg": 50000.0,
#         "_std_units": "Pa",
#         "_std_min": 0.0,
#         "_std_max": 100000.0,
#         "_std_avg": 50000.0,
#         "_std_step": 100.0,
#     },
#     {   # Kinematic viscosity of the fluid
#         "_min": 0.0,
#         "_max": 1.0,
#         "_avg": 0.5,
#         "_std_units": "m^2/s",
#         "_std_min": 0.0,
#         "_std_max": 1.0,
#         "_std_avg": 0.5,
#         "_std_step": 0.01,
#     },
# ]

# print("Variables:")
# for param, extra in zip(DAnalysis.param_lt, extr_data_lt):
#     var = Variable(_sym=param.sym,
#                    _fwk=param.fwk,
#                    name=param.name,
#                    description=param.description,
#                    _idx=param.idx,
#                    _cat=param.cat,
#                    _units=param.units,
#                    _dims=param.dims,
#                    **extra,)
#     # print(var, "\n")
#     vars_lt.append(var)

# print("Dimensionless Coefficients:")
# for coef in DAnalysis.pi_coef_lt:
#     print(f"{coef.sym} = {coef.pi_expr}")
#     print(coef, "\n")


# print("=== Sensitivity Analysis: ===")
# print(f"Coefficients: {DAnalysis.pi_coef_lt[0]}\n")

# sen = Sensitivity(_idx=0,
#                   _sym="S_{0}",
#                   _fwk="CUSTOM",
#                   name="Sensitivity",
#                   description="Sensitivity Analysis",
#                   _pi_expr=DAnalysis.pi_coef_lt[0].pi_expr,
#                   _variables=list(DAnalysis.pi_coef_lt[0].par_dims.keys())
#                   )
# print("=== Sensitivity: ===")
# print(sen, "\n")
# td = DAnalysis.pi_coef_lt[0].par_dims
# td["d"] = 5.05
# td["y"] = 5.05
# print(td, "\n")
# r = sen.analyze_symbolically(td)
# print(r, "\n")
# r = sen.analyze_numerically([[0.1, 10.0]] * len(sen.variables))
# print(r, "\n")

# print("\n=== Sensitivity Analysis: === \n")
# # print(sena)
# sena = SensitivityAnalysis(_idx=0,
#                            _sym="SA_{0}",
#                            _fwk="CUSTOM",
#                            name="Sensitivity Analysis",
#                            description="Sensitivity Analysis",
#                            _relevance_lt=vars_lt,
#                            _coefficient_lt=DAnalysis.pi_coef_lt,)
# # sena.analyze_pi_sensitivity(cutoff="avg")
# sena.analyze_pi_sensitivity(category="NUMERIC")
# # print(sena._coefficient_mp.keys(), "\n")
# # print(sena._coefficient_mp.get_entry("\\Pi_{0}"))
# # montecarlo
