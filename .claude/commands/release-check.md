Check if the current branch is ready for a release to main.

## Steps

1. Verify the venv is active: run `echo $VIRTUAL_ENV` and confirm it points to this project. If not, activate with `source venv/Scripts/activate` first.
2. Run `git log --oneline main..HEAD` to see what commits will be included
2. Verify all commits follow Conventional Commits format (`feat:`, `fix:`, `docs:`, etc.)
   - Flag any commits that don't follow the convention (these won't trigger version bumps correctly)
3. Run the full test suite: `pytest tests/ -v --tb=short --cov=src/pydasa --cov-report=term`
4. Check that `src/pydasa/_version.py` matches the latest tag: `git describe --tags --abbrev=0`
5. Run `python -m build` to verify the package builds cleanly
6. Summarize:
   - Number of commits and their types (feat/fix/etc.)
   - Expected version bump (major/minor/patch/none)
   - Test results and coverage
   - Build status
   - Any issues that need fixing before merge
