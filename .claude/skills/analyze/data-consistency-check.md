# Data Consistency Check Skill

Use this skill to verify that verifiable claims in prose coincide with their authoritative sources (tables, figure captions, data logs, notebook outputs, bib entries). Catches stale numbers, parameter drift, and narrative/data mismatches before they reach LaTeX.

## When to Use

- Before migrating a section from tgt notes to LaTeX
- After rewriting a results, analysis, or evaluation section
- When tables, experimental data, or notebooks have been updated
- Before chapter sign-off

## Scope

**Always work section by section.** Never sweep a whole file at once. Ask the user which section to check, verify it, present the report, apply approved fixes, then stop and wait before moving to the next section.

## What it Verifies

### Quantitative claims
- **Numerical values** — specific numbers in prose (e.g., `λ₀ = 100 req/s`, `25,200 experiments`)
- **Parameter bindings** — variable assignments (e.g., `c_DB = 1`, `K_IW = 32`, `ρ_DB ≈ 0.50`)
- **Ranges and sweeps** — stated bounds vs. actual sweeps (e.g., `c ∈ [1,2,4]`, `α ∈ {1.00, 0.80, ...}`)
- **Counts** — number of components, experiments, scenarios, references, figures, tables
- **Thresholds / goals met** — claims like `W_j ≤ 500 ms` vs. measured values
- **Formulas** — claimed formula matches the formula in the source

### Qualitative claims (feature-based comparison)

Qualitative claims ARE in scope. Verify them through **analytical, feature-based comparison**: extract the defining features/characteristics of the claim, and check that those characteristics coincide with the data.

For each qualitative claim:
1. **Extract the defining features** — what characteristics must be true for the claim to hold?
2. **Look up those features in the data** — find the corresponding metrics/values in tables, charts, or logs
3. **Check coincidence** — do the features from the claim match the features in the data?

Examples:

| Claim | Defining features | Data check | Status |
|---|---|---|---|
| "***DB*** is the bottleneck" | highest utilisation; lowest service rate; receives aggregate load from both paths | ρ_DB = 0.50 (highest), μ_DB = 200 (lowest), λ_DB ≈ 99 (aggregate of λ_W + λ_R) | ✓ features coincide |
| "horizontal scaling beats vertical" | increasing c_j yields larger marginal improvement in σ/θ than increasing μ_j or K_j | coefficient sensitivity table shows c_j sweep drops ρ_j more than μ_j sweep | ✓ / ⚠ based on data |
| "broker nodes are overprovisioned" | ρ ≪ safe-zone limit; service rate ≫ arrival rate | ρ_IB = 0.10, μ_IB/λ_IB ratio high | ✓ features coincide |
| "error compounds along the pipeline" | per-hop ε accumulates; end-to-end χ < per-node χ | χ_e2e vs per-node χ shows cumulative loss across layers | ✓ / ⚠ based on data |

If the defining features cannot be derived from the data (e.g., data doesn't report them), flag as `⚠ features not verifiable`.

### Natural-language qualifier mapping

Translate common NL qualifiers in prose into mathematical relationships before checking against data:

| Prose qualifier | Relationship | Check |
|---|---|---|
| "at least X" | `X ≤ reported` | reported value must be ≥ X |
| "at most X" / "no more than X" / "up to X" | `reported ≤ X` | reported value must be ≤ X |
| "between X and Y" | `X ≤ reported ≤ Y` | reported value must sit in the interval |
| "greater than X" / "exceeds X" | `reported > X` | strict inequality |
| "less than X" / "below X" | `reported < X` | strict inequality |
| "approximately X" / "≈ X" / "~ X" | `|reported − X|/X ≤ 0.05` | within 5% tolerance |
| "roughly X" / "around X" / "on the order of X" | `|reported − X|/X ≤ 0.10` | within 10% tolerance (looser language) |
| "equal to X" / "= X" | `reported == X` | exact or within 5% for derived |
| "dominates X" / "highest X" | `reported[subject] > reported[others]` | subject's value must exceed all others |
| "lowest X" / "minimum X" | `reported[subject] < reported[others]` | subject's value must be less than all others |
| "most / mostly X" | majority of cases satisfy property X | count-based check |
| "scales linearly with X" | `dY/dX ≈ constant across sweep` | trend check over the sweep |
| "saturates at X" | `reported → X as input → limit` | asymptotic check |
| "A, B, C" enumeration | claimed set = reported set | set equality check |

**Partial verification is acceptable.** Not every qualifier will map cleanly to available data. When the relationship can be checked, check it; when it can't, flag as `⚠ features not verifiable` and move on. This skill is a step toward rigour, not a complete verification oracle.

## Sources of Truth

**The table wins.** When prose and table disagree, treat the table as authoritative and flag the prose for correction. Only override the table if the user explicitly confirms the table is stale.

Priority order when multiple sources exist:

1. **LaTeX tables**: `assets/tbl/0N/*.tex` — authoritative for rendered output
2. **Tables in tgt notes**: `assets/md/tgt/0N/*.md` — authoritative working values
3. **Structured data**: `.json`, `.csv`, notebook outputs under `src/notebooks/`
4. **Figure captions**: parameters stated alongside figures
5. **Bib file**: `db/references.bib` — for reference keys and citations

When sources disagree, **always report the conflict explicitly** rather than silently choosing one. Never auto-edit a source to resolve a conflict.

## Tolerance

**Minimum 95% match.** A claim passes if it matches the source within ±5%.

- **Exact match required** for: counts, integer parameters, discrete sweeps, table/figure numbers, reference keys, units
- **±5% tolerance** for: derived quantities, percentages, rates marked with `≈` or `~`, rounded values
- **Exact match required** for formulas (ignoring whitespace and equivalent notation)
- **Units must match** (req/s vs req/sec, ms vs milliseconds, MB vs Megabytes)
- **Feature-based match** for qualitative claims: all defining features must coincide with the data

## Process

### Step 1 — Confirm scope
- Ask user which section to check (file + section heading or line range)
- Ask which sources to check against if not obvious from the section's table/figure references

### Step 2 — Extract claims
Read the prose and pull out verifiable claims:
- Numerical values with units
- Variable bindings from math expressions
- Counts and discrete enumerations
- Thresholds, ranges, and sweep specifications
- Table/Figure/List references
- Qualitative claims with their defining features

### Step 3 — Locate authoritative source per claim
For each claim, identify the most authoritative source using the priority order above. For qualitative claims, identify the data that would confirm or refute each defining feature. If no source is located, flag as `⚠ source not located` — do NOT assume the claim is correct.

### Step 4 — Compare
Apply the 95% tolerance rule for numerics. For qualitative claims, check each defining feature against the data and mark the claim as `✓ features coincide` only if all features match.

### Step 5 — Produce a discrepancy report

| # | Line | Claim | Source | Expected | Found | Status |
|---|---|---|---|---|---|---|
| 1 | 407 | "7 components" | Table 6 | 7 | 7 | ✓ match |
| 2 | 471 | "μ_DB = 200 req/s" | `dasa_pacs2_eqs_write.tex` | 200 | 200 | ✓ match |
| 3 | 495 | "25,200 experiments" | — | ? | 25,200 | ⚠ source not located |
| 4 | 525 | "DB is the bottleneck" | coefficient ranking | ρ_DB highest, μ_DB lowest, λ_DB aggregate | all 3 features confirmed | ✓ features coincide |
| 5 | 404 | "λ_W = 20 req/s" | `dasa_pacs2_eqs_write.tex` | 19.8 | 20 | ✓ match (within 5%) |

Status legend:
- `✓ match` — numerical claim matches source within 95% tolerance
- `✓ features coincide` — qualitative claim's defining features all match the data
- `⚠ mismatch` — numerical claim disagrees with source beyond 5% tolerance
- `⚠ feature mismatch` — one or more defining features contradict the data
- `⚠ source not located` — no authoritative source found
- `⚠ features not verifiable` — defining features not present in available data
- `⚠ source conflict` — two sources disagree; user must decide

### Step 6 — Await user decision
Ask the user which discrepancies to fix, per line. Options:
- **Fix prose** to match the table (default, since the table wins)
- **Fix source** (only if user confirms the table is stale)
- **Accept** with rationale (e.g., deliberate rounding)
- **Investigate** (source not located or features not verifiable)

**Never auto-apply fixes.** Always wait for per-line user approval.

### Step 7 — Apply approved fixes
Apply fixes one at a time. Re-verify each fixed claim to confirm the discrepancy is resolved.

## What This Skill Does NOT Do

- Does not verify prose style, grammar, flow, or rhetoric (that's `style-polish`)
- Does not verify that citations exist in `db/references.bib` (that's `reference-check`)
- Does not re-run simulations, notebooks, or compute new values
- Does not auto-edit prose or source files without per-line user approval
- Does not generate missing data — it flags the gap and stops

## Example Invocations

- "Check data consistency in §5.6.2.3 of `assets/md/tgt/05/dasa.md` against the iter2 tables"
- "Verify all numerical claims in §5.6.1.3 against `assets/tbl/05/iter1/`"
- "Cross-check the Refinement Opportunities section against the coefficient sensitivity findings"

## Additional Checks (learned from real usage)

Beyond per-claim verification, sweep for these cross-cutting patterns during each invocation:

### Symbol convention collision (cross-scope check)

When a single symbol (e.g., α, β, γ) appears in multiple places, verify it carries the **same meaning** throughout. Symbols are easy to define locally and drift globally.

Example pattern to catch: α defined as "write fraction" in E-QS tables but used as "read fraction" in a routing matrix, with both scopes internally consistent but mutually contradictory. To detect:
1. List every definition of the symbol (explicit labels like "write fraction α" or implicit via formulas)
2. Compare meanings across scopes
3. Pick one canonical meaning with the user, then propagate across all scopes (tables, formulas, prose, captions, scenario labels)

### Magnitude-mismatch / factor-of-10 typos

Leading-zero drift (`0.1` ↔ `0.01`) and trailing-zero drift (`100` ↔ `1000`) are recurring typo patterns. After verifying individual values, sweep for:
- Adjacent references to the same quantity with different magnitudes (e.g., `ρ_req,WN` appearing as 0.1 in a table row and 0.01 in prose)
- Related quantities where the ratio between them is suspicious (all `ρ_req` values should lie in a physically plausible range relative to each other)

### Citation cascade after cite-key fixes

When a cite key is corrected in one section (e.g., §Search Results RD-08 → `numrich_metric_2008`), **sweep all other sections** for references to the same RD/PS identifier or the concept it describes. Single cite-key fixes can leave orphaned wrong citations elsewhere.

### Table header/column-swap audit

Compare the current edited table headers against the original Excel2LaTeX export (usually preserved as comments below the active table). Column label swaps during manual editing are common and silent — the data values stay in place while labels flip.

### Synthesis/supplementary notes as tie-breakers

When a claim looks wrong against the primary-task notes (e.g., RQ answer files), check for secondary synthesis files (`thirdparty_*.md`, `*summary*.md`, `rw-*.md`) before flagging. These often contain paper-specific terminology (e.g., "LFT system") that the primary notes abstract away.

### Joint-feature decomposition

Claims like "A and B both show X but also Y" are conjunctions. Evaluate each feature independently, then check the intersection. Single-feature analysis often produces false positives: the full claim set may match the conjunction even when individual features have wider scope.

## Cross-Source Priority Reminder

When verifying, always walk the full source priority chain (LaTeX table → tgt notes table → structured data → figure caption → bib), not just the first match. Conflicts between sources are themselves findings worth reporting.