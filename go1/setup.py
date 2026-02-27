from setuptools import setup

setup(
    name="go1",
    version="0.0.1",
    # The package root is this directory itself, which contains __init__.py, agile.py, mdp/, agents/, etc.
    # We map the top-level package name "go1" to the current directory.
    packages=["go1"],
    package_dir={"go1": "."},
)