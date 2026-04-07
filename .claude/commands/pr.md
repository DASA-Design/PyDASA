Create a pull request from the current branch.

## Steps

1. Run `git status` and `git log --oneline main..HEAD` to understand the scope
2. Verify all commits use Conventional Commits format
3. Run the test suite to make sure everything passes
4. Determine the target branch:
   - If current branch is `dev`, target is `main`
   - Otherwise, target is `dev`
5. Push the current branch if not already pushed: `git push -u origin HEAD`
6. Create the PR with `gh pr create`:
   - Title: derived from commits (if single feat/fix, use that; otherwise summarize)
   - Body: grouped commit summary, test status, and coverage
7. Return the PR URL

If the user provides arguments (e.g., `/pr main` or `/pr --title "Add X"`), use those overrides.
