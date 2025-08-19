from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pydasa",
    version="0.0.1",
    description="Python package for Dimensional Analysis for Scientific Applications and Software Architecture (PyDASA).",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DASA-Design/PyDASA",
    # TODO update author information, be aware of the email address!!!
    author="@SFAM",
    author_email="your_email@example.com",
    # TODO check classifiers!!!
    license="GPL-3.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    # TODO check for min version in trhe requirements!!!
    install_requires=[
        "numpy",
        "scipy",
        "sympy",
        # "pandas",
        # "matplotlib",
    ],
    extras_require={
        "dev": [
            "pytest",
            "twine",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
    python_requires=">=3.10",
)
