from setuptools import setup, find_packages

setup(
    name="hrl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.3",
        "matplotlib>=3.7.1",
        "gymnasium>=1.0.0",
        "pyquaticus>=0.0.1",
        "ray[rllib]>=2.5.0",
        "pygame>=2.4.0",
        "tensorflow>=2.0.0",
        "torch>=2.6.0",
        "pandas>=2.2.3",
        "seaborn>=0.13.2",
        "scipy>=1.14.1",
        "tqdm>=4.67.1",
    ],
) 