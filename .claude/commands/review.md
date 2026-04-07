Review code changes on the current branch for quality and conventions.

## Steps

1. Run `git diff main...HEAD` (or `git diff dev...HEAD` if targeting dev) to get all changes
2. Review each changed file against PyDASA conventions:
   - Local variables prefixed with `_`
   - Function names: verb-first with acronym suffixes
   - Type hints on signatures
   - Docstrings with Args/Returns
   - No inline chaining; sequential steps with intermediates
   - No excessive column alignment
3. Check for:
   - Missing or inadequate test coverage for new/changed code
   - Breaking changes to the public API (classes in `__init__.py.__all__`)
   - Hardcoded values that should come from JSON config
   - Security issues (OWASP top 10)
4. Report findings grouped by severity:
   - **Must fix**: convention violations, missing tests, bugs
   - **Should fix**: style improvements, minor issues
   - **Consider**: optional improvements, refactoring opportunities
