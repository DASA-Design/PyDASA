# PyDASA Test Coverage Report

**Date:** 2026-04-08
**Version:** 0.7.0
**Python:** 3.12.10
**Tests:** 680 passed, 0 failed, 1 warning
**Duration:** 36.67s
**Overall Coverage:** 80%

---

## Summary

680 tests pass across all modules. Overall coverage is 80%, pulled down primarily by the `structs/` package (custom data structures) which sits at 20-45% coverage. The core domain modules (dimensional analysis, workflows, validation, elements) are all above 90%.

---

## Coverage by Module

### Core (97-100%)

| Module | Stmts | Miss | Cover | Missing Lines |
|--------|-------|------|-------|---------------|
| `__init__.py` | 40 | 0 | 100% | |
| `_version.py` | 1 | 0 | 100% | |
| `core/basic.py` | 109 | 3 | 97% | 62, 263, 266 |
| `core/constants.py` | 3 | 0 | 100% | |
| `core/io.py` | 33 | 0 | 100% | |
| `core/setup.py` | 79 | 2 | 97% | 165-169 |

### Dimensional Analysis (91-100%)

| Module | Stmts | Miss | Cover | Missing Lines |
|--------|-------|------|-------|---------------|
| `dimensional/buckingham.py` | 248 | 15 | 94% | 156, 195, 414, 436-437, 480-482, 533-534, 550, 586-590, 595 |
| `dimensional/fundamental.py` | 36 | 0 | 100% | |
| `dimensional/model.py` | 391 | 37 | 91% | 281, 315-317, 445-448, 454, 457-461, 477, 479, 518-519, 635-637, 645, 651, 786, 826, 840, 849, 863, 872, 886, 895, 909, 995, 997, 1004, 1072-1076 |
| `dimensional/vaschy.py` | 227 | 19 | 92% | 153-154, 160-163, 187-214, 252, 257, 282, 533, 578, 614-615 |

### Elements (91-100%)

| Module | Stmts | Miss | Cover | Missing Lines |
|--------|-------|------|-------|---------------|
| `elements/parameter.py` | 66 | 6 | 91% | 121-123, 211-215 |
| `elements/specs/conceptual.py` | 39 | 0 | 100% | |
| `elements/specs/numerical.py` | 180 | 4 | 98% | 517, 534-536 |
| `elements/specs/statistical.py` | 92 | 2 | 98% | 143-144 |
| `elements/specs/symbolic.py` | 136 | 11 | 92% | 98, 121-122, 145-146, 190-191, 218-219, 251-252 |

### Analysis (91-92%)

| Module | Stmts | Miss | Cover | Missing Lines |
|--------|-------|------|-------|---------------|
| `analysis/scenario.py` | 364 | 29 | 92% | 255-256, 259-260, 280, 298-302, 350-353, 409, 616, 629, 638, 647, 656, 679, 703, 714, 793, 812, 852-856, 861 |
| `analysis/simulation.py` | 503 | 43 | 91% | 357-358, 363-364, 366-367, 369-370, 382, 396, 398, 473, 478-480, 498, 522, 530-531, 540-544, 575, 591-596, 639, 677, 682, 689, 694-696, 745, 974, 987-988, 1257-1258, 1268, 1275 |

### Workflows (90-96%)

| Module | Stmts | Miss | Cover | Missing Lines |
|--------|-------|------|-------|---------------|
| `workflows/basic.py` | 154 | 14 | 91% | 117-120, 152-153, 160, 171-172, 315, 354-355, 364, 373 |
| `workflows/influence.py` | 115 | 5 | 96% | 167, 177, 187, 228, 267 |
| `workflows/phenomena.py` | 122 | 12 | 90% | 110-111, 144-146, 170, 207-209, 270-272 |
| `workflows/practical.py` | 227 | 22 | 90% | 135, 197, 208-210, 316-318, 349-351, 384-385, 396-397, 401, 409-411, 430-432 |

### Validations (94-100%)

| Module | Stmts | Miss | Cover | Missing Lines |
|--------|-------|------|-------|---------------|
| `validations/decorators.py` | 230 | 13 | 94% | 206-207, 219-220, 438-439, 445, 479-480, 484-488 |
| `validations/error.py` | 38 | 2 | 95% | 68, 75 |
| `validations/patterns.py` | 21 | 0 | 100% | |

### Serialization (91%)

| Module | Stmts | Miss | Cover | Missing Lines |
|--------|-------|------|-------|---------------|
| `serialization/parser.py` | 148 | 14 | 91% | 131-133, 208-209, 235-236, 270, 376, 417-418, 435-437, 466 |

### Data Structures (20-100%)

| Module | Stmts | Miss | Cover | Missing Lines |
|--------|-------|------|-------|---------------|
| `structs/lists/arlt.py` | 213 | 156 | **27%** | 97-120, 135-142, 151, 160, ... |
| `structs/lists/dllt.py` | 0 | 0 | 100% | (empty) |
| `structs/lists/ndlt.py` | 85 | 47 | **45%** | 66-74, 89-94, 104-106, ... |
| `structs/lists/sllt.py` | 303 | 243 | **20%** | 108-129, 144-151, 160, ... |
| `structs/tables/htme.py` | 61 | 35 | **43%** | 68-76, 90-94, 108-112, ... |
| `structs/tables/scht.py` | 310 | 209 | **33%** | 95-96, 104-105, 248-293, ... |
| `structs/tools/hashing.py` | 8 | 0 | 100% | |
| `structs/tools/math.py` | 43 | 0 | 100% | |
| `structs/tools/memory.py` | 16 | 0 | 100% | |
| `structs/types/functions.py` | 59 | 0 | 100% | |
| `structs/types/generics.py` | 11 | 0 | 100% | |

### Context (stubs)

| Module | Stmts | Miss | Cover |
|--------|-------|------|-------|
| `context/conversion.py` | 0 | 0 | 100% |
| `context/system.py` | 0 | 0 | 100% |
| `context/units.py` | 0 | 0 | 100% |

---

## Coverage by Category

| Category | Modules | Avg Coverage | Status |
|----------|---------|-------------|--------|
| Core | 6 | **99%** | Excellent |
| Dimensional | 4 | **94%** | Above target |
| Elements | 5 | **96%** | Above target |
| Analysis | 2 | **91%** | Above target |
| Workflows | 4 | **92%** | Above target |
| Validations | 3 | **96%** | Above target |
| Serialization | 1 | **91%** | Above target |
| Structs | 11 | **52%** | Below target |
| Context | 3 | **100%** | Stubs (0 code) |

---

## Key Observations

The domain modules (everything the library does for dimensional analysis) are all above 90% — the project target. The gap is in `structs/` where the custom data structures (ArrayList, SingleLinkedList, SCHashTable) have 20-45% coverage. These are general-purpose data structures not specific to dimensional analysis.

The 3 `context/` modules are empty stubs (planned unit conversion feature, not yet implemented).

HTML coverage report with per-line detail is available at `htmlcov/index.html`.

---

**TOTAL: 4722 statements, 943 missed, 80% coverage**
