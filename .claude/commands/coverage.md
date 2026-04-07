Run a detailed coverage analysis and identify gaps.

## Steps

1. Verify the venv is active: run `echo $VIRTUAL_ENV` and confirm it points to this project. If not, activate with `source venv/Scripts/activate` first.
2. Run `pytest tests/ --cov=src/pydasa --cov-report=term-missing --cov-report=html -q`
2. Parse the coverage output and identify:
   - Modules below 90% coverage (the project target)
   - Specific uncovered line ranges in each module
3. For the lowest-coverage modules, read the source and uncovered lines
4. Suggest specific test cases that would improve coverage, grouped by module
5. Report:
   - Overall coverage percentage
   - Per-module breakdown (sorted worst to best)
   - Top 3 highest-impact test additions (most uncovered lines addressable)

If the user specifies a module (e.g., `/coverage dimensional`), focus the analysis on that module only.
