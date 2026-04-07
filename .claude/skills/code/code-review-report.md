# Code Review Report Skill

Analyze source code and produce a structured report on its internal quality. This skill reads code only — not docs, not architecture, not comments. It evaluates what the code actually does, how it is structured, and where it has problems.

---

## Step 0 — Read the Code

Before writing anything, read every `.py` file in the target scope. If a module or directory is specified, read all files in it. If the whole package is specified, scan the structure first and then read each module systematically.

Do not skip files. Do not skim. The report must be grounded in what the code actually says, not what comments or docstrings claim.

Build a map of:
- Every class and its public/private methods
- Every standalone function
- Every import (internal and external)
- Every global or module-level variable
- Call chains: what calls what

---

## Report Structure

The report has seven sections. Every finding must reference a specific file and line range. No generic observations.

### 1. Abstractions

Evaluate the abstractions present in the code:

**What to look for:**
- Classes that do too many things (god objects)
- Classes that do almost nothing (thin wrappers with no logic)
- Functions longer than ~50 lines that mix multiple responsibilities
- Inheritance hierarchies: are they justified or could composition work?
- Abstract base classes: are subclasses actually using the contract?
- Data classes or containers that should be plain dicts (or vice versa)
- Duplicated logic across modules that suggests a missing abstraction
- Premature abstractions: generic frameworks built for one use case

**Report format:**
```
#### Abstractions

| File:Lines | Element | Finding | Severity |
|------------|---------|---------|----------|
| module.py:15-80 | ClassName | Does X, Y, and Z — three responsibilities | warning |
| other.py:42 | helper_fn | Wraps a single stdlib call with no added logic | note |
```

For each finding, write 1-2 sentences explaining what you see and why it matters.

---

### 2. Dependencies

Map how modules depend on each other and on external packages.

**What to look for:**
- Circular imports or near-circular dependency chains
- Modules that import from too many siblings (high fan-in coupling)
- External dependencies used in only one place (could be isolated)
- Hard-coded imports that could be parameterized or injected
- Unused imports
- Version-sensitive external APIs (pinned versions in pyproject.toml vs actual usage)

**Report format:**
```
#### Dependencies

Internal dependency graph (simplified):
  workflows -> dimensional -> elements -> core
  analysis -> dimensional, elements
  serialization -> elements, dimensional
  validations -> elements

Issues:
| From | To | Issue |
|------|----|-------|
| module_a.py | module_b.py | Circular: A imports B, B imports A through C |
| module_x.py | scipy.special | Used once on line 47; rest of module is pure Python |
```

---

### 3. Good Ideas

Identify patterns, decisions, or implementations that are well done. This section exists because not everything is a problem — good code deserves recognition, and these patterns should be preserved or extended.

**What to look for:**
- Clean separation of concerns that makes testing easy
- Validation patterns that catch errors early
- Consistent internal conventions followed across modules
- Clever but readable solutions to domain-specific problems
- Good use of Python features (generators, context managers, descriptors, etc.)
- Configuration-driven behavior that avoids hardcoding

**Report format:**
```
#### Good Ideas

| File:Lines | Pattern | Why it works |
|------------|---------|-------------|
| vaschy.py:30-45 | Schema loaded from JSON config | Single source of truth, easy to extend |
| workflows/basic.py:12-20 | Decorator-based validation | Catches bad input at definition time, not runtime |
```

---

### 4. Risks

Code that works now but is fragile, brittle, or likely to break under change.

**What to look for:**
- Functions that depend on dict key order or insertion order
- Silent failures: bare `except`, swallowed exceptions, empty catch blocks
- Type coercion that could produce wrong results silently (int vs float, str vs bytes)
- Mutable default arguments
- Global state that multiple functions read/write
- Race conditions in any concurrent or async code
- Magic numbers or magic strings without named constants
- Assumptions about input size, shape, or type that aren't validated
- Code paths that are never tested (check against test file imports)

**Report format:**
```
#### Risks

| File:Lines | Risk | Impact | Likelihood |
|------------|------|--------|------------|
| core/io.py:88 | Bare except swallows all errors | Data loss on malformed input | medium |
| analysis/sim.py:120 | Mutable default dict in function signature | Shared state across calls | low |
```

---

### 5. Bugs and Complications

Actual bugs or code that is unnecessarily complicated.

**What to look for:**
- Off-by-one errors in loops or slicing
- Incorrect operator precedence without parentheses
- Variable shadowing (local name hides outer scope or import)
- Dead code: unreachable branches, unused variables, commented-out blocks
- Copy-paste errors: similar blocks with inconsistent changes
- Overly nested logic (3+ levels of if/for/try) that could be flattened
- String formatting inconsistencies (f-strings mixed with .format mixed with %)
- Index/key access without checking existence

**Report format:**
```
#### Bugs and Complications

| File:Lines | Type | Description |
|------------|------|-------------|
| dimensional/model.py:95 | bug | Loop variable `i` shadows import from line 3 |
| structs/scht.py:40-70 | complication | 4-level nested if/for; could extract inner logic |
| core/basic.py:22 | dead code | Variable `_tmp` assigned but never read |
```

---

### 6. Technical Debt

Accumulated shortcuts, deferred work, and structural issues that slow down future development.

**What to look for:**
- TODO/FIXME/HACK/XXX comments in the code (list them all)
- Stub modules or placeholder implementations
- Functions that duplicate logic from another module instead of sharing
- Inconsistent naming: same concept called different names across modules
- Missing type hints on public functions
- Missing or outdated docstrings on public classes
- Test files that test implementation details instead of behavior
- Hard-coded paths, URLs, or configuration values
- Modules that have grown beyond ~300 lines and should be split

**Report format:**
```
#### Technical Debt

| File:Lines | Debt Type | Description | Effort |
|------------|-----------|-------------|--------|
| context/conversion.py | stub | Entire module is placeholder, imported but commented out | large |
| core/setup.py:15 | TODO | "# TODO: support nested config merging" | medium |
| dimensional/*.py | naming | `_mtx` in model.py vs `_matrix` in buckingham.py for same concept | small |
```

Effort estimates: small (< 1 hour), medium (1-4 hours), large (> 4 hours).

---

### 7. Summary

Close the report with:

1. **Health score**: one-line assessment (healthy / needs attention / concerning)
2. **Top 3 priorities**: the three findings that should be addressed first, with reasoning
3. **Preservation list**: patterns from "Good Ideas" that should not be broken during refactoring

### 8. Proposals

End the report with concrete ideas and suggestions that come from the analysis. These are not fixes — they are opportunities, directions, and things worth exploring. Write them as short paragraphs, not tables. Examples:
- A refactoring that would simplify multiple modules at once
- A new abstraction suggested by the duplication patterns found
- A PoC worth trying in `lab/` based on a risk or complication
- An architectural change that would prevent a class of bugs
- A testing strategy that would cover the gaps identified

The goal is to give the maintainer ideas, not just problems.

---

## Scope Options

The user can request a report at different scopes:

| Scope | What gets analyzed |
|-------|--------------------|
| `module` | One `.py` file |
| `package` | One directory under `src/pydasa/` (e.g., `dimensional/`) |
| `full` | The entire `src/pydasa/` codebase |

For `full` scope, analyze packages in dependency order (core first, then elements, dimensional, analysis, workflows) so later sections can reference earlier findings.

---

## Writing Style

- Each section must open with a **plain-language paragraph** summarizing the findings before any table. Explain what you found as if talking to a colleague — what matters, why it matters, what it means for the project.
- Tables are for reference. Prose is for understanding. Both are required.
- Avoid jargon without explanation. If you use a term like "fan-in coupling" or "god object", briefly say what it means.
- Keep insights clear and concise. One good sentence beats three vague ones.
- The reader is a researcher, not a DevOps engineer. Frame findings in terms of "this will bite you when..." or "this means you can safely..." rather than abstract severity ratings.

## Rules

- Every finding must cite a file path and line range. No vague claims.
- Read the code. Do not rely on docstrings, comments, or READMEs to understand what the code does.
- Do not suggest fixes in this report. This is diagnosis only. Fixes belong in a separate task.
- Do not flag style issues that match the project's conventions (underscore locals, acronym names, verb-first functions). Read `.claude/skills/develop/coding-conventions.md` first.
- Severity levels: `bug` (broken), `risk` (fragile), `warning` (problematic pattern), `note` (observation), `debt` (accumulated shortcut).
- Be honest. If the code is clean, say so. Do not invent problems to fill sections.
- For `full` scope reports, save output to `notes/reports/code_review_<scope>_<date>.md`.
- After generating, add a devlog entry summarizing key findings and the report's stage (analysis).
- When comparing with third-party reports later, note alignments (both agree), inconsistencies (disagree), and unique findings (only one found it).
