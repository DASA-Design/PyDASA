# Lab - Proof of Concepts

Disposable space for testing ideas before they become modules.

## Rules

- Each PoC is a folder: `lab/<name>/`
- Max 2 `.py` files + 1 supporting file (data, config, notebook) per PoC
- Keep code short (~500 lines total)
- Name the folder after the idea, not the date
- Include a one-line comment at the top of the main `.py` describing the hypothesis

## Lifecycle

A PoC either graduates into `src/pydasa/` or stays here as a dead end (documented in `notes/devlog.md`).

## Example

```
lab/
  sparse_matrix_pi/
    sparse_pi.py        # test if scipy.sparse speeds up large dimensional matrices
    bench_data.json
  lazy_validation/
    lazy_val.py          # defer constraint checks to computation time instead of init
    test_lazy.py
```

## Status

This directory is NOT packaged or tested in CI. It is a scratchpad.
