# Coding Conventions Skill

Use this skill when writing or reviewing Python code for the PyDASA project.

## Style Rules

- No excessive column-alignment spacing (do not align `=` signs across lines)
- No inline command chaining (do not nest function calls as arguments)
- Break operations into short, sequential steps with intermediate variables
- Use type hints on all function signatures
- Use docstrings with Args/Returns format
- Pass config lists as parameters rather than relying on closures or globals
- Local variables inside functions or methods must start with `_` (e.g., `_lam`, `_cfg`)
- Variable names should use acronyms to avoid search homonyms with comments or notes (e.g., `_mu` not `_service_rate`, `_rho` not `_utilization`)
- Function names must start with a full verb describing the action, followed by `_` and up to 5 acronyms for related context (e.g., `compute_dc_theta`, `aggregate_net_cfg_vars`, `plot_yoly_2d`)

## Example

Instead of:
```python
new_data[_vtag(var, suffix, tags)] = _sum_or_zero(df, idx, var, tags)
```

Write:
```python
v_tag = _vtag(var, suffix, tags)
new_data[v_tag] = _sum_or_zero(df, idx, var, tags)
```

## Project-Specific

- Source of truth: JSON config files have authoritative naming and data over code or markdown comments in notebooks
- PACS node acronyms (2-3 uppercase letters): IB, IWS, IRS, DB, WAS, RAS, OB
- Key notebooks: `PyDASA-PACS-Ex-Iter1.ipynb` (single node), `PyDASA-PACS-Ex-Iter2.ipynb` (7-node network)
