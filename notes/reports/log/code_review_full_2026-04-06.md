# PyDASA Full Code Review Report

**Scope:** `src/pydasa/` (53 .py files)
**Date:** 2026-04-06
**Version:** 0.7.0

---

## 1. Abstractions

| File:Lines | Element | Finding | Severity |
|------------|---------|---------|----------|
| `dimensional/model.py:60-997` | `Matrix` | God object: owns variable management, matrix construction, RREF solving, nullspace computation, coefficient generation, coefficient derivation, serialization, and property re-declarations that shadow parent (`idx`, `sym`, `alias`, `fwk`). Over 900 lines with at least 5 distinct responsibilities. | warning |
| `analysis/simulation.py:63-700+` | `MonteCarlo` | God object: manages expression parsing, distribution sampling, dataset generation, coefficient evaluation, statistics computation, caching, and serialization. Over 700 lines mixing orchestration with low-level numpy loops. | warning |
| `analysis/scenario.py:52-864` | `Sensitivity` | Large class (~800 lines) that combines expression parsing, symbolic differentiation, SALib numerical analysis, symbol mapping, and serialization. Could separate parsing/mapping from analysis execution. | warning |
| `elements/parameter.py:37` | `Variable` | Four-way multiple inheritance (`ConceptualSpecs`, `SymbolicSpecs`, `NumericalSpecs`, `StatisticalSpecs`) is an ambitious composition model. Works in practice but makes MRO debugging non-trivial and `clear()` must explicitly call each parent. | note |
| `dimensional/buckingham.py:46-612` | `Coefficient` | ~570 lines combining identity (Foundation), bounds (BoundsSpecs), variable management, expression building, data generation, setpoint calculation, and serialization. Duplicates `to_dict`/`from_dict` patterns from `Variable` almost verbatim. | warning |
| `structs/lists/arlt.py` and `structs/lists/sllt.py` | `ArrayList` / `SingleLinkedList` | Near-identical APIs (append, prepend, insert, remove, get, index_of, swap, sublist, concat, clone, `_error_handler`, `_validate_type`, `__str__`, `__repr__`). These two classes duplicate ~80% of their logic with no shared base class or protocol. | warning |
| `core/basic.py:55-64` | `SymBasis.__post_init__` | The `if not self._sym: self._sym = self._sym.strip()` pattern does nothing useful: if `_sym` is empty (falsy), stripping it still yields `""`. Same for `_fwk` and `_alias`. This is dead logic, not validation. | note |
| `dimensional/vaschy.py:37-650` | `Schema` | Large class (~600 lines) but responsibilities (FDU management, regex generation, serialization) are cohesive. Acceptable size given domain complexity. | note |

---

## 2. Dependencies

### Internal dependency graph (simplified)

```
workflows/phenomena  -> dimensional/model -> dimensional/buckingham -> elements/parameter
workflows/influence  -> analysis/scenario -> dimensional/buckingham, serialization/parser
workflows/practical  -> analysis/simulation -> dimensional/buckingham, elements/parameter
workflows/basic      -> elements/parameter, dimensional/buckingham, dimensional/vaschy

dimensional/model    -> dimensional/buckingham, dimensional/vaschy, elements/parameter,
                        serialization/parser, core/setup, validations/*

analysis/scenario    -> serialization/parser (parse_latex, create_latex_mapping),
                        SALib, sympy

analysis/simulation  -> serialization/parser (parse_latex, create_latex_mapping),
                        scipy.stats, sympy, numpy

elements/parameter   -> elements/specs/*, serialization/parser, dimensional/vaschy
elements/specs/symbolic -> validations/decorators (TYPE_CHECKING: dimensional/vaschy)
elements/specs/conceptual -> dimensional/vaschy, core/setup, core/basic

dimensional/buckingham -> elements/parameter, dimensional/vaschy, serialization/parser
dimensional/vaschy     -> dimensional/fundamental, core/basic, core/setup
dimensional/fundamental -> core/basic, core/setup, validations/error

core/basic   -> core/setup, validations/patterns, validations/decorators
core/setup   -> core/io, core/constants (loads JSON at import time)

structs/*    -> validations/error, structs/types/*
serialization/parser -> sympy.parsing.latex, validations/patterns
```

### Issues

| From | To | Issue |
|------|----|-------|
| `core/setup.py:41` | `core/io.py` | Imports `Path` from `io` instead of `pathlib` directly: `from pydasa.core.io import Path, load`. `Path` is not defined in `io.py` -- it is re-exported implicitly from the `from pathlib import Path` at io.py:17. This creates a fragile transitive dependency. |
| `elements/specs/symbolic.py:31-32` | `dimensional/vaschy` | Uses `TYPE_CHECKING` guard to avoid circular import. Correct pattern, but the runtime `self._schema` access works only because `ConceptualSpecs` sets it first via MRO. Fragile if class hierarchy changes. |
| `analysis/scenario.py:21-22` | `SALib` | `SALib.sample.fast_sampler` and `SALib.analyze.fast` are imported unconditionally. SALib is a heavy dependency used only in `analyze_numerically()`. Could be lazy-imported. |
| `analysis/simulation.py:28` | `scipy.stats` | Imported at module level but never used anywhere in the file. Unused import. |
| `structs/tables/scht.py:27` | `random` | Used only in `__post_init__` for MAD hash parameters. Acceptable but noted. |
| `dimensional/fundamental.py:28` | `validations/error` | Imported as `error`, asserted, but never called anywhere in the `Dimension` class. Unused import. |

---

## 3. Good Ideas

| File:Lines | Pattern | Why it works |
|------------|---------|--------------|
| `validations/decorators.py:28-641` | Decorator-based property validation (`validate_type`, `validate_range`, `validate_choices`, `validate_pattern`, etc.) | Eliminates repetitive validation boilerplate across 20+ property setters. Composable, readable, and consistent error messages. This is the library's strongest engineering decision. |
| `core/setup.py:54-170` | `str, Enum` enums for Frameworks, VarCardinality, CoefCardinality, AnaliticMode, SimulationMode | String-compatible enums enable direct comparison with JSON config values while maintaining type safety. Each has a `.description` property for human-readable output. |
| `core/setup.py:176-263` | Frozen dataclass singleton `PyDASAConfig` loading from JSON | Single source of truth for FDU framework definitions. JSON config drives the dimensional domain, not hardcoded Python. Frozen dataclass prevents accidental mutation. |
| `elements/specs/` package | Four-perspective composition (Conceptual, Symbolic, Numerical, Statistical) | Clean conceptual separation of what a Variable IS, how it is WRITTEN, what VALUES it takes, and how UNCERTAINTY is modeled. Each perspective is independently testable. |
| `dimensional/vaschy.py:273-304` | Dynamic regex generation from FDU symbols | Regex patterns for dimensional expression validation are auto-generated from the current schema's FDU list, so adding a new dimension automatically updates all validation. |
| `serialization/parser.py:85-137` | `create_latex_mapping()` with fallback aliases | Carefully handles the gap between SymPy's `parse_latex` (which can mangle multi-letter subscripts) and the actual variable names. Fallback aliases prevent silent lookup failures. |
| `workflows/basic.py:91-177` | `_convert_to_objects()` and `_convert_to_schema()` | Accepts multiple input formats (dict, object, string, list) and normalizes them. Makes the API flexible for notebook users while maintaining internal type consistency. |
| `validations/patterns.py:27-39` | Nested brace regex for LaTeX subscripts (5 levels) | Pragmatic solution to match deeply nested LaTeX like `M_{a*(c*t_{R_{P*(A*(C*S))}})}`. Built incrementally from `_BRACE_L0` through `_BRACE_L4` for readability. |
| `structs/tools/math.py:108-149` | `gfactorial()` supporting float inputs via gamma function | Cleanly extends standard `math.factorial` to handle fractional factorials using `math.gamma(x+1)`, with optional precision control. |

---

## 4. Risks

| File:Lines | Risk | Impact | Likelihood |
|------------|------|--------|------------|
| `core/setup.py:189-199` | `PyDASAConfig.__post_init__` uses `object.__setattr__` to bypass frozen constraint and loads a JSON file at module import time. If `cfg/default.json` is missing or malformed, every import of `pydasa` fails with an opaque error. | Library completely unusable | low |
| `core/setup.py:258-263` | `PYDASA_CFG` is instantiated at module level, NOT via `get_instance()`. This means two instances exist: `PYDASA_CFG` and the one created by `get_instance()`. The singleton pattern is partially broken. | Inconsistent config state if someone mutates via `get_instance()` | medium |
| `dimensional/model.py:420-422` | `_extract_fdus()` uses set comprehension `{fdus[i] for i in range(len(fdus))}` which destroys FDU precedence order. The list is then returned as `list()` of a set, giving arbitrary order. | Incorrect FDU ordering in working_fdus | medium |
| `analysis/simulation.py:501` | `self._sym_func = self._sym_func` -- self-assignment does nothing. The parsed expression from `parse_latex` is already stored in `self._sym_func` on line 495. | No functional impact but suggests confused logic | low |
| `analysis/simulation.py:679-682` | `# FIXME: hotfix for queue functions` -- runtime type-checking with `isinstance(v, (list, tuple, np.ndarray))` to extract last elements from array values. This is a known fragile workaround. | Incorrect coefficient evaluation if data shape changes | medium |
| `elements/specs/statistical.py:109-121` | `dist_type` setter validates against a hardcoded list `["uniform", "normal", "triangular", "exponential", "lognormal", "custom"]` instead of using a config-driven enum or the decorator pattern used everywhere else. | Inconsistent validation, list can drift from actual supported types | low |
| `dimensional/buckingham.py:420-464` | `calculate_setpoint()` takes a `vars` parameter that shadows the built-in `vars()`. While technically safe, it can confuse readers and tools. | Readability/maintenance | low |
| `dimensional/model.py:334` | `IN = VarCardinality.IN.value` assigns to local `IN` which shadows Python's `input` conceptually and could confuse readers (though doesn't shadow a builtin). | Readability only | low |
| `structs/tables/scht.py:242-293` | `SCHashTable.__post_init__` catches all exceptions via `except Exception as err: self._error_handler(err); raise`. The error handler re-raises with modified traceback. Double-raise can produce confusing stack traces. | Debugging difficulty | low |
| `workflows/practical.py:317` | `print(f"Warning: {_msg}")` -- uses `print` instead of `warnings.warn()` or `logging.warning()`. Not capturable by standard Python warning filters. | Silent in production, invisible in testing frameworks | medium |

---

## 5. Bugs and Complications

| File:Lines | Type | Description |
|------------|------|-------------|
| `core/basic.py:59-64` | bug | `SymBasis.__post_init__`: `if not self._sym: self._sym = self._sym.strip()` -- when `_sym` is `""` (falsy), it strips `""` which is still `""`. The condition should be `if self._sym:` (truthy) to strip non-empty strings. Same bug for `_fwk` (line 61) and `_alias` (line 63). |
| `dimensional/fundamental.py:69-70` | bug | Same pattern: `if not self.unit: self._unit = self._unit.strip()` does nothing when unit is empty. Should be `if self.unit:`. |
| `dimensional/fundamental.py:127-128` | bug | `from_dict` reads `_name` with key `"_name"` but `to_dict` writes it as `"name"` (line 106, key has no underscore prefix). Roundtrip `to_dict -> from_dict` will lose the name field. |
| `dimensional/model.py:420-422` | bug | `_extract_fdus()` converts `fdus` list to a set (`{fdus[i] for i in ...}`), destroying the precedence ordering that downstream code depends on. Should use `dict.fromkeys(fdus)` or an ordered dedup approach. |
| `analysis/simulation.py:501` | dead code | `self._sym_func = self._sym_func` is a self-assignment with no effect. |
| `analysis/simulation.py:28` | dead code | `from scipy import stats` is imported but `stats` is never referenced in the module. |
| `dimensional/fundamental.py:27-33` | dead code | `handle_error` imported as `error`, asserted, but never called in `Dimension`. |
| `structs/lists/sllt.py:547` | complication | `IndexError("Index", pos2, "is out of range")` passes a tuple of 3 values to `IndexError`, producing a confusing error representation like `('Index', 5, 'is out of range')` instead of a clean message string. Compare with line 548 which uses the same wrong pattern. The `arlt.py` version (line 405) correctly uses an f-string. |
| `structs/tables/htme.py:53-60` | complication | Docstrings for `_key` and `_value` are in Spanish ("Es la llave del registro del mapa" / "Es el valor del registro del mapa") while the rest of the codebase is in English. |
| `dimensional/buckingham.py:518-560` and `elements/parameter.py:150-185` | complication | `to_dict()` methods are near-identical 40-line blocks with the same field iteration, Schema detection by class name string, numpy conversion, callable skipping, and underscore stripping. This is copy-paste code that should be a shared utility. |
| `dimensional/vaschy.py:555-588` and `dimensional/model.py:960-1000+` | complication | More `to_dict()` duplicates with the same pattern. Four classes implement nearly identical serialization logic. |
| `structs/lists/arlt.py:490-504`, `structs/lists/sllt.py:655-669`, `structs/lists/ndlt.py:57-74`, `structs/tables/htme.py:62-76`, `structs/tables/scht.py:653-667` | complication | `_error_handler()` is copy-pasted identically across 5 classes. Each does the same `inspect.currentframe()` + `f_back` pattern. Should be a mixin or standalone utility. |
| `structs/lists/arlt.py:551-578`, `structs/lists/sllt.py:719-746`, `structs/lists/ndlt.py:119-147`, `structs/tables/htme.py:154-182` | complication | `__str__`/`__repr__` methods are copy-pasted across 4+ struct classes with identical logic (iterate `vars()`, format attributes). Should be a mixin from `Foundation.__str__` or a shared utility. |

---

## 6. Technical Debt

| File:Lines | Debt Type | Description | Effort |
|------------|-----------|-------------|--------|
| `context/conversion.py:11` | stub | `# TODO add UnitConverter for one variable/parameter` -- entire module is a stub (2 lines of comments). | large |
| `context/system.py:17` | stub | `# TODO add UnitManager class` -- entire module is a stub. | large |
| `context/units.py:15` | stub | `# TODO complete implementation of Measurement class` -- entire module is a stub. | large |
| `structs/lists/dllt.py:18` | stub | `# TODO complete the DoubleLinkedList implementation` -- module has only a docstring and TODO. | medium |
| `__init__.py:15-16` | TODO | `# TODO conversion still in development` -- UnitStandarizer, MeasureSystem, Unit imports commented out. | large |
| `__init__.py:34` | TODO | `# TODO measurement still in development` -- DoubleLinkedList import commented out. | medium |
| `serialization/parser.py:43` | TODO | `# TODO this regex doesnt work, check latter` -- known broken regex in `latex_to_python()`. Current fallback uses string replacement. | small |
| `serialization/parser.py:142-143` | TODO | `# TODO improve using sympy + pint later` -- dimensional expression parsing could leverage established unit libraries. | large |
| `elements/specs/symbolic.py:96` | TODO | `# TODO improve this ignoring null or empty strings for constants` -- validation of null/empty dims bypassed. | small |
| `elements/specs/symbolic.py:150` | TODO | `# TODO move '*' as global operator to cfg module?` | small |
| `elements/specs/symbolic.py:171-172` | TODO | `# TODO move '*' and '* ' as global operator to cfg module?` and `# TODO do I use also regex for this?` | small |
| `elements/specs/statistical.py:108` | TODO | `# TODO improve this for later` -- hardcoded distribution type list. | small |
| `validations/decorators.py:468` | TODO | `# TODO improve msg construction` in `validate_pattern`. | small |
| `structs/lists/ndlt.py:65` | TODO | `# TODO check utility of this error handling` in `Node._error_handler`. | small |
| `structs/lists/sllt.py:703` | TODO | `# TODO do I need the try/except block?` in `SingleLinkedList.__iter__`. | small |
| `structs/tables/scht.py:681` | TODO | `# TODO check usability of this function` in `SCHashTable._validate_type`. | small |
| `structs/tools/hashing.py:45` | TODO | `# TODO data should be hashable?` | small |
| `dimensional/buckingham.py:450` | TODO | `# TODO can be make in symbolic, sympy is worth it?` in `calculate_setpoint`. | small |
| `workflows/phenomena.py:88` | TODO | `# TODO check this after refector` in `AnalysisEngine.__post_init__`. | small |
| `analysis/simulation.py:679` | FIXME | `# FIXME: hotfix for queue functions` -- acknowledged workaround for array-shaped values. | medium |
| `core/setup.py:134` | naming | `AnaliticMode` is a misspelling of "Analytic". Used across 4+ files. | small |
| `core/setup.py:28` | naming | Module docstring says `Module config.py` but filename is `setup.py`. | small |
| `serialization/parser.py:4` | naming | Module docstring says `Module latex.py` but filename is `parser.py`. | small |
| `structs/types/functions.py:4` | naming | Module docstring says `Module default.py` but filename is `functions.py`. | small |
| `context/system.py:4` | naming | Module docstring says `Module domain.py` but filename is `system.py`. | small |
| `dimensional/buckingham.py:1-612` + `elements/parameter.py:150-185` + `dimensional/vaschy.py:555-588` + `dimensional/model.py:960-1000` | duplication | Four classes implement near-identical `to_dict()`/`from_dict()` serialization logic (~40 lines each). Should be extracted into a shared mixin or utility. | medium |
| `structs/lists/arlt.py` + `structs/lists/sllt.py` | duplication | ArrayList (579 lines) and SingleLinkedList (747 lines) share ~80% identical method signatures and logic (append, prepend, insert, remove, get, index_of, swap, sublist, concat, clone, compare, _validate_type, _error_handler, __str__, __repr__). No shared abstract base class. | medium |
| `_version.py:2` | naming | Comment says `MAYOR` instead of `MAJOR`. | small |

---

## 7. Summary

### Health Score

**Needs attention.** The core dimensional analysis engine (Schema, Dimension, Variable, Coefficient, Matrix) is well-designed and functional. The decorator-based validation system and JSON-driven configuration are genuinely good engineering. However, several classes have grown beyond maintainable size, serialization logic is heavily duplicated, and the `structs/` package duplicates logic across data structure implementations. The `context/` package is entirely unimplemented (3 stub files). Multiple module docstrings reference wrong filenames, and there is one confirmed data-loss bug in `Dimension.from_dict()`.

### Top 3 Priorities

1. **Fix `_extract_fdus()` set-ordering bug** (`dimensional/model.py:420-422`). Using a set destroys FDU precedence order, which can silently produce wrong dimensional matrix column ordering. This is the highest-severity correctness issue found.

2. **Fix `Dimension.from_dict()` key mismatch** (`dimensional/fundamental.py:127-128`). `to_dict()` writes `"name"` but `from_dict()` reads `"_name"`, breaking roundtrip serialization. Any workflow that saves and reloads Dimension objects will lose the name field.

3. **Extract shared `to_dict()`/`from_dict()` into a mixin** (buckingham.py, parameter.py, vaschy.py, model.py). Four near-identical 40-line serialization methods are the largest source of copy-paste risk. A single `SerializableMixin` would eliminate ~160 lines of duplication and prevent future divergence.

### Preservation List

These patterns should be protected during any refactoring:

- **Decorator-based validation system** (`validations/decorators.py`) -- composable, consistent, and eliminates boilerplate across 20+ property setters.
- **Four-perspective Variable composition** (`elements/specs/`) -- clean conceptual model that is independently testable.
- **JSON config as source of truth** (`core/setup.py` + `cfg/default.json`) -- frozen singleton ensures consistent framework definitions.
- **Dynamic regex generation from FDU symbols** (`dimensional/vaschy.py:273-304`) -- self-updating validation that stays in sync with the schema.
- **`WorkflowBase._convert_to_objects()` and `_convert_to_schema()`** (`workflows/basic.py:91-177`) -- flexible input normalization for notebook ergonomics.
- **Nested brace regex for LaTeX subscripts** (`validations/patterns.py:27-39`) -- pragmatic, well-documented solution to a genuinely hard parsing problem.
