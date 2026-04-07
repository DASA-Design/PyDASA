Run the PyDASA test suite and report results.

## Steps

1. Verify the venv is active: run `echo $VIRTUAL_ENV` and confirm it points to this project. If not, activate with `source venv/Scripts/activate` first.
2. Run `pytest tests/ -v --tb=short --cov=src/pydasa --cov-report=term` from the project root
2. If any tests fail:
   - Read the failing test file and the source file it tests
   - Identify the root cause (do NOT guess — read the code)
   - Propose a fix and wait for approval before editing
3. If all tests pass, report the coverage summary
4. Flag any module below 90% coverage

## Optional arguments

If the user specifies a module (e.g., `/test dimensional`), run only:
```
pytest tests/pydasa/<module>/ -v --tb=short --cov=src/pydasa/<module> --cov-report=term
```
