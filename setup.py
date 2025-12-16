
from setuptools import setup, find_packages

setup(
    name="gbayesdesign",
    version="0.1.0",
    description="Small toolkit for Bayesian design and power calculations and sampling.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["numpy"],
    extras_require={
        "full": ["scipy"],
        "plots": ["matplotlib"]
    },
    python_requires=">=3.9",
)
