# PyDASA Commands & Quick Reference

Central reference for everything you (and Claude) can do in this project.

---

## Acronym Dictionary

| Acronym | Meaning | Context |
|---------|---------|---------|
| ADD | Attribute-Driven Design | Software architecture method (Bass, Clements & Kazman) |
| API | Application Programming Interface | Public classes/functions others use |
| ASR | Architecturally Significant Requirement | Quality attribute scenario in ADD |
| BPMN | Business Process Model and Notation | Diagram standard for workflows |
| C4 | Context, Containers, Components, Code | Architecture diagramming model |
| CD | Continuous Deployment/Delivery | Auto-publish after merge to main |
| CI | Continuous Integration | Auto-run tests on push/PR |
| CLI | Command Line Interface | Terminal tools (e.g., Claude Code, gh) |
| CRUD | Create, Read, Update, Delete | Basic data operations |
| DSL | Domain-Specific Language | Custom syntax for a narrow task |
| FR | Functional Requirement | What the system must do |
| GPL | GNU General Public License | PyDASA's licence (GPL-3.0-or-later) |
| HTML | HyperText Markup Language | Web page format, Sphinx output |
| IDE | Integrated Development Environment | VS Code, PyCharm, etc. |
| MP4 | MPEG-4 Part 14 | Video format (Gource exports) |
| OWASP | Open Web Application Security Project | Security vulnerability checklist |
| PACS | (Project-specific) | PyDASA simulation domain acronym |
| PATH | System search path for executables | OS environment variable |
| PEP | Python Enhancement Proposal | Python style/feature standards (e.g., PEP 8) |
| PoC | Proof of Concept | Quick experiment in `lab/` |
| PR | Pull Request | GitHub: propose merging a branch |
| PyPI | Python Package Index | Where `pip install pydasa` downloads from |
| QA | Quality Attribute | Non-functional requirement (performance, testability, etc.) |
| README | Read Me | Project description file at repo root |
| reST | reStructuredText | Markup used by Sphinx docs |
| SDK | Software Development Kit | Library/tools for building on a platform |
| UML | Unified Modeling Language | Diagram standard for software design |
| venv | Virtual Environment | Isolated Python install (`python -m venv`) |

---

## Claude Code Commands

Slash commands available in Claude Code sessions (`.claude/commands/`):

### Research Workflow

| Command | What it does |
|---------|-------------|
| `/poc <name>` | Create a new proof of concept in `lab/` |
| `/poc list` | List existing PoCs and status |
| `/poc promote <name>` | Plan how to move a PoC into `src/pydasa/` |
| `/devlog` | Show recent devlog entries |
| `/devlog add` | Add a new experiment journal entry |
| `/devlog status` | List open experiments (not yet decided) |
| `/stage` | Show all work items and their current stages |
| `/stage <name> <stage>` | Advance a work item to the next stage |

### Testing & Quality

| Command | What it does |
|---------|-------------|
| `/test` | Run full test suite with coverage |
| `/test <module>` | Run tests for a specific module (e.g., `/test dimensional`) |
| `/coverage` | Detailed coverage analysis with gap identification |
| `/coverage <module>` | Coverage for a specific module |
| `/review` | Review branch changes against PyDASA conventions |
| `/code-report <scope>` | Generate a code review report (abstractions, deps, risks, debt, bugs) |

### CI/CD & Release

| Command | What it does |
|---------|-------------|
| `/release-check` | Verify branch is ready for merge to main |
| `/changelog` | Preview next CHANGELOG entry from commits |
| `/ci-status` | Check GitHub Actions status, diagnose failures |
| `/pr` | Create a PR with auto-generated title/body |
| `/deps` | Check dependency health and security |

### Research Skills (`.claude/skills/`)

| Skill | Location | What it does |
|-------|----------|-------------|
| data-analysis | analyze/ | PACS simulation data, notebooks, plotting |
| data-consistency-check | analyze/ | Verify numerical claims against sources |
| reference-check | analyze/ | Map references to BibTeX cite keys |
| code-documentation | code/ | Generate inline docs, docstrings, READMEs |
| code-review-report | code/ | Analyze code: abstractions, deps, risks, bugs, debt |
| software-architecture | code/ | ADD framework architecture documentation |
| coding-conventions | develop/ | PyDASA style rules |
| debugging | develop/ | Debugging rules and common pitfalls |
| notebook-editing | develop/ | Safe Jupyter notebook editing |
| rewrite | write/ | 3 rewrite options for paragraphs |
| style-polish | write/ | Prose improvement for sections/chapters |

---

## Development Stages

```
proof-of-concept -> planning -> dev -> test -> analysis -> refinement -> merge -> publish
```

---

## Testing

```bash
# Full suite with coverage
pytest tests/ -v --tb=short --cov=src/pydasa --cov-report=term

# Specific module
pytest tests/pydasa/dimensional/ -v

# With missing lines report
pytest tests/ --cov=src/pydasa --cov-report=term-missing

# Coverage as HTML report
pytest --cov=pydasa --cov-report=html tests/

# Notebooks
pytest --nbval-lax --nbval-current-env tests/notebooks/

# Clean coverage cache
del .coverage
rmdir /s /q htmlcov
```

---

## Building & Distribution

```bash
# Dev install
pip install -e ".[dev]"

# Verify install
python -c "import pydasa; print(pydasa.__version__)"

# Clean old builds
Remove-Item -Recurse -Force dist, build, src\pydasa.egg-info

# Build package
python -m build

# Or the old way
python setup.py bdist_wheel sdist

# Test on TestPyPI (optional)
python -m twine upload --repository testpypi dist/*

# Upload to PyPI
python -m twine upload dist/*

# Install from PyPI
pip install pydasa

# Install from built wheel (testing)
pip install dist/pydasa-0.7.0-py3-none-any.whl
```

---

## Git & Release

```bash
# Conventional commit examples
git commit -m "feat(dimensional): add sparse matrix support"
git commit -m "fix(core): correct JSON config loading for nested keys"
git commit -m "docs: update quickstart with new API"

# Check what's going to release
git log --oneline $(git describe --tags --abbrev=0)..HEAD

# Semantic release (CI handles this, manual only if needed)
semantic-release version
semantic-release publish
```

---

## GitHub CLI

```bash
# CI status
gh run list --limit 10
gh run view <run-id>

# PRs
gh pr list --state open
gh pr create --title "feat: ..." --body "..."

# Dependabot alerts
gh api /repos/DASA-Design/PyDASA/dependabot/alerts
```

---

## Documentation (Sphinx + ReadTheDocs)

Hosted at: https://pydasa.readthedocs.io

```bash
# Build HTML docs
cd docs && make html

# Initialize Sphinx (one-time)
sphinx-quickstart

# Multi-language support
make gettext
sphinx-build -b gettext . _build/gettext
sphinx-intl update -p _build/gettext -l es
sphinx-build -b html -D language='es' . _build/html/es
```

### Docs structure

```
docs/
  source/
    conf.py               # Sphinx config (Python 3.11, pydata-sphinx-theme)
    index.rst             # Main landing page
    _static/              # CSS, logos, images
    _templates/           # Custom HTML templates
    getting_started/      # Installation, quickstart, tutorial
    user_guide/           # Dimensional analysis, variables, simulation, sensitivity
    api/                  # Core, buckingham, dimensional, analysis, handler, utils
    examples/             # Basic, Monte Carlo, sensitivity analysis
    developer/            # Contributing, architecture, testing
    changelog.rst
```

---

## Gource Visualization

```bash
# Simple visualization
gource

# With title and settings
gource --title "PyDASA Development" --seconds-per-day 1 --auto-skip-seconds 0.5 --highlight-users --stop-at-end

# High quality full HD
gource --1920x1080 --stop-at-end --key --highlight-users
```

### Export to MP4

Requires ffmpeg (`winget install Gyan.FFmpeg`).

```bash
# Standard quality
gource --title "PyDASA Development" --seconds-per-day 1 --auto-skip-seconds 1 --1920x1080 --hide mouse,filenames --highlight-users --stop-at-end --output-ppm-stream - | ffmpeg -y -r 60 -f image2pipe -vcodec ppm -i - -vcodec libx264 -preset medium -pix_fmt yuv420p -crf 23 -threads 0 -bf 0 pydasa_development.mp4

# Quick preview (lower quality, faster)
gource --seconds-per-day 0.5 --1280x720 --output-ppm-stream - | ffmpeg -y -r 30 -f image2pipe -vcodec ppm -i - -vcodec libx264 -preset ultrafast -crf 28 pydasa_quick.mp4

# High quality (larger file, slower)
gource --seconds-per-day 3 --1920x1080 --output-ppm-stream - | ffmpeg -y -r 60 -f image2pipe -vcodec ppm -i - -vcodec libx264 -preset slow -crf 18 pydasa_hq.mp4
```

### Useful options

| Option | What it does |
|--------|-------------|
| `--title "Your Title"` | Set visualization title |
| `--seconds-per-day 2` | Speed (lower = faster) |
| `--auto-skip-seconds 1` | Skip inactive periods |
| `--1920x1080` | Resolution |
| `--hide mouse,filenames` | Clean view |
| `--highlight-users` | Highlight user activity |
| `--stop-at-end` | Stop instead of loop |
| `--key` | Show date/time legend |
| `--start-date "2026-01-01"` | Filter by date range |
| `--file-filter "src/\|docs/"` | Focus on specific directories |

### Windows PATH setup

```powershell
# Add Gource to PATH permanently
[Environment]::SetEnvironmentVariable(
    "Path",
    [Environment]::GetEnvironmentVariable("Path", "User") + ";C:\Program Files\Gource",
    "User"
)
# Then restart PowerShell

# Or alias for current session
Set-Alias gource "C:\Program Files\Gource\gource.exe"
```

---

## Virtual Environments

```bash
python -m venv venv
.\venv\Scripts\activate   # Windows
deactivate

pip freeze > requirements.txt
pip install -r requirements.txt
```

---

## Key Paths

| What | Where |
|------|-------|
| Source code | `src/pydasa/` |
| Tests | `tests/pydasa/` |
| JSON configs (source of truth) | `src/pydasa/core/cfg/` |
| Lab / PoCs | `lab/` |
| Dev log | `notes/devlog.md` |
| This file | `notes/commands.md` |
| Docs source | `docs/source/` |
| Sphinx config | `docs/source/conf.py` |
| CI workflows | `.github/workflows/` |
| Claude commands | `.claude/commands/` |
| Research skills | `.claude/skills/` |
| Architecture report | `notes/reports/pydasa_architecture_report.md` |
| Version | `src/pydasa/_version.py` |
| ReadTheDocs config | `.readthedocs.yaml` |
