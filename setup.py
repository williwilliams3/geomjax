from setuptools import setup, find_packages

# Import version information from _version.py
from geomjax._version import __version__

# Read the long description from a README file
with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="geomjax",
    version=__version__,
    author="Blackjax Authors, Williams B., Yu H.",
    description="Geometric MCMC Samplers in Jax",
    url="https://github.com/williwilliams3/geomjax",
    packages=find_packages(),
    install_requires=["numpy", "blackjax", "jax", "matplotlib"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
)
