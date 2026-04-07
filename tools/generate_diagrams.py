"""
Generate class and package diagrams for PyDASA.

Uses pyreverse (from pylint) and Graphviz to produce UML class
and package dependency diagrams in both PNG and SVG formats.

Usage:
    python tools/generate_diagrams.py # all modules, both formats
    python tools/generate_diagrams.py --overview # bird's-eye view (names only, no attributes)
    python tools/generate_diagrams.py --module dimensional # one module only
    python tools/generate_diagrams.py --full-only # only the full package diagram
"""
import argparse
import subprocess
import sys
from pathlib import Path

# modules to generate individual diagrams for
MODULES = [
    "core",
    "dimensional",
    "elements",
    "workflows",
    "analysis",
    "validations",
    "serialization",
    "structs",
]


def _get_version() -> str:
    """Read the current version from _version.py."""
    _version_file = Path("src/pydasa/_version.py")
    _text = _version_file.read_text()
    for _line in _text.splitlines():
        if _line.startswith("__version__"):
            return _line.split('"')[1]
    return "unknown"


def _build_output_dir(version: str) -> Path:
    """Create and return the versioned output directory."""
    _out = Path(f"docs/source/_static/development/v{version}")
    _out.mkdir(parents=True, exist_ok=True)
    return _out


def _run_pyreverse(target: str,
                   output_dir: Path,
                   project_name: str,
                   fmt: str = "png",
                   filter_mode: str = "PUB_ONLY",
                   overview: bool = False) -> list[Path]:
    """Run pyreverse and return list of generated files."""

    _cmd = [
        "pyreverse",
        "-o", fmt,
        "-p", project_name,
        "-d", str(output_dir),
        "--filter-mode", filter_mode,
        "--colorized",
        target,
    ]

    if overview:
        _cmd.insert(-1, "--only-classnames")
        _cmd.insert(-1, "--no-standalone")

    _result = subprocess.run(_cmd, capture_output=True, text=True)

    if _result.returncode != 0:
        # fallback to module invocation
        _cmd_fallback = [
            sys.executable, "-m", "pylint.pyreverse.main",
            "--output", fmt,
            "--project", project_name,
            "--output-directory", str(output_dir),
            "--filter-mode", filter_mode,
            "--colorized",
        ]
        if overview:
            _cmd_fallback.extend(["--only-classnames", "--no-standalone"])
        _cmd_fallback.append(target)
        _result = subprocess.run(_cmd_fallback, capture_output=True, text=True)
        if _result.returncode != 0:
            print(f"  ERROR: {_result.stderr.strip()}")
            return []

    _files = sorted(output_dir.glob(f"*{project_name}*.{fmt}"))
    return _files


def _generate_module(module: str,
                     output_dir: Path,
                     formats: list[str],
                     filter_mode: str,
                     overview: bool = False) -> None:
    """Generate diagrams for a single module in all requested formats."""
    _target = f"src/pydasa/{module}"
    _project = f"pydasa-{module}"

    if overview:
        _project = f"pydasa-{module}-overview"

    if not Path(_target).exists():
        print(f"  SKIP: {_target} does not exist")
        return

    print(f"\n  {module}/")

    for _fmt in formats:
        _files = _run_pyreverse(
            _target, output_dir, _project, _fmt, filter_mode, overview,
        )
        for _f in _files:
            _size_kb = _f.stat().st_size / 1024
            print(f"    {_f.name} ({_size_kb:.0f} KB)")


def _generate_full(
    output_dir: Path,
    formats: list[str],
    filter_mode: str,
    overview: bool = False,
) -> None:
    """Generate diagrams for the full package."""
    _target = "src/pydasa"
    _project = "pydasa-overview" if overview else "pydasa"

    print("\n  full package")

    for _fmt in formats:
        _files = _run_pyreverse(
            _target, output_dir, _project, _fmt, filter_mode, overview,
        )
        for _f in _files:
            _size_kb = _f.stat().st_size / 1024
            print(f"{_f.name} ({_size_kb:.0f} KB)")


def main() -> None:
    _parser = argparse.ArgumentParser(
        description="Generate UML class and package diagrams for PyDASA"
    )
    _parser.add_argument(
        "--module",
        default=None,
        help="Generate for one module only (e.g., 'dimensional').",
    )
    _parser.add_argument(
        "--full-only",
        action="store_true",
        help="Generate only the full package diagram, skip per-module.",
    )
    _parser.add_argument(
        "--png-only",
        action="store_true",
        help="Generate PNG only (default: both PNG and SVG).",
    )
    _parser.add_argument(
        "--overview",
        action="store_true",
        help="Bird's-eye view: class names only, no attributes/methods, "
             "skip isolated classes.",
    )
    _parser.add_argument(
        "--all-members",
        action="store_true",
        help="Include private and special methods (default: public only).",
    )
    _args = _parser.parse_args()

    _version = _get_version()
    _out_dir = _build_output_dir(_version)
    _filter = "ALL" if _args.all_members else "PUB_ONLY"
    _formats = ["png"] if _args.png_only else ["png", "svg"]

    print(f"PyDASA v{_version} — Diagram Generation")
    print(f"Output: {_out_dir}/")
    print(f"Formats: {', '.join(_formats)}")
    print(f"{'=' * 40}")

    _is_overview = _args.overview

    if _args.module:
        _generate_module(
            _args.module, _out_dir, _formats, _filter, _is_overview,
        )
    elif _args.full_only or _is_overview:
        _generate_full(_out_dir, _formats, _filter, _is_overview)
    else:
        # per-module diagrams (readable size)
        print("\nPer-module diagrams:")
        for _mod in MODULES:
            _generate_module(_mod, _out_dir, _formats, _filter)

        # full package diagram
        print("\nFull package:")
        _generate_full(_out_dir, _formats, _filter)

    print(f"\n{'=' * 40}")
    print(f"Done. All diagrams in: {_out_dir}")


if __name__ == "__main__":
    main()
