import os
import re
from setuptools import setup, find_packages


def read_version() -> str:
    """read the project version in a preset file.

    Raises:
        RuntimeError: if the version string is not found.

    Returns:
        str: project version.
    """
    version_file = os.path.join("src", "pydasa", "_version.py")
    with open(version_file, "r", encoding="utf-8") as f:
        content = f.read()
    m = re.search(r'__version__\s*=\s*[\'"]([^\'"]+)[\'"]', content)
    if not m:
        raise RuntimeError("Unable to find version string.")
    return m.group(1)


def read_long_description() -> str:
    """Read the long description from the README file.

    Returns:
        str: project long description.
    """
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
    return long_description


setup(
    name="pydasa",
    version=read_version(),
    description="Python package for Dimensional Analysis for Scientific Applications and Software Architecture (PyDASA).",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/DASA-Design/PyDASA",
    # TODO update author information, beware of the email address!!!
    author="@SFAM",
    author_email="your_email@example.com",
    # TODO check classifiers!!!
    license="GPL-3.0",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        # "Topic :: Scientific/Engineering :: Physics",
    ],
    install_requires=[
        "antlr4-python3-runtime==4.11",
        "numpy>=1.26.4",
        "scipy>=1.13.0",
        "sympy>=1.12",
        "matplotlib>=3.8.0",
        "pandas>=2.1.0",
        "SALib>=1.4.5",
    ],
    extras_require={
        "dev": [
            "pytest>=8.1.1",
            "twine>=6.1.0",
        ],
        "docs": [
            "sphinx>=7.3.7",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    python_requires=">=3.10",
)
