from setuptools import setup

setup(
    name="datalib",
    version="0.1",
    description="A module to load and visualise package data.",
    author="Sebastian Callh",
    author_email="sebastian.calh@gmail.com",
    packages=["datalib"],
    install_requires=["pandas", "seaborn"],
)
