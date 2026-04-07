Preview or draft a CHANGELOG entry for the upcoming release.

## Steps

1. Determine the current version from `src/pydasa/_version.py`
2. Run `git log --oneline $(git describe --tags --abbrev=0)..HEAD` to get all commits since the last tag
3. Group commits by type:
   - **Features** (`feat:`)
   - **Bug Fixes** (`fix:`)
   - **Performance** (`perf:`)
   - **Other** (docs, refactor, test, ci, build, chore)
4. Determine the expected next version based on commit types:
   - Any `feat:` -> minor bump
   - Only `fix:`/`perf:` -> patch bump
   - Only docs/style/refactor/test/build/ci/chore -> no bump
5. Draft the CHANGELOG entry in the existing format from `CHANGELOG.md`
6. Present it for review — do NOT edit CHANGELOG.md directly (semantic-release handles that)
