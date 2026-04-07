Check the status of CI workflows on GitHub.

## Steps

1. Run `gh run list --limit 10` to see recent workflow runs
2. If any runs failed, run `gh run view <run-id>` to get details
3. For failed runs, identify the failing step and summarize the error
4. Check if there are any open PRs with failing checks: `gh pr list --state open`
5. Report:
   - Status of the last run on each branch (main, dev, current)
   - Any failures with a brief root-cause summary
   - Whether the current branch's checks are passing
