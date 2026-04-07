Check dependency health for the project.

## Steps

1. Verify the venv is active: run `echo $VIRTUAL_ENV` and confirm it points to this project. If not, activate with `source venv/Scripts/activate` first.
2. Read `pyproject.toml` to get current dependency specs
3. Run `pip list --outdated` to find packages with newer versions available
3. For each outdated core dependency (numpy, scipy, sympy, matplotlib, pandas, SALib, antlr4-python3-runtime):
   - Note current pinned/minimum version vs latest available
   - Flag any that are significantly behind (2+ minor versions)
4. Check for known vulnerabilities: `pip audit` (if available) or `gh api /repos/DASA-Design/PyDASA/dependabot/alerts --jq '.[].security_advisory.summary'`
5. Report:
   - Which deps are up to date
   - Which deps could be bumped (with risk assessment)
   - Any security advisories
   - Suggested `pyproject.toml` changes (if any)

Do NOT edit pyproject.toml without explicit approval.
