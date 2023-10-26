from setuptools import setup, find_packages

# Import version information from _version.py
from your_package_name._version import __version__

# Read the long description from a README file
with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="geomjax",
    version=__version__,
    author="Blackjax Authors, Williams",
    description="Geometric MCMC Samplers",
    url="https://github.com/williwilliams3/geomjax",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "blackjax-nightly",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)