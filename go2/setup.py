from setuptools import setup

setup(
    name="go2",
    version="0.0.1",
    # The package root is this directory itself, which contains __init__.py, agile.py, mdp/, agents/, etc.
    # We map the top-level package name "go2" to the current directory.
    packages=["go2"],
    package_dir={"go2": "."},
)