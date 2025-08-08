from setuptools import setup, find_packages

setup(
    name="process-management-project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.2.1",
        "matplotlib>=3.8.3",
        "sortedcontainers>=2.4.0",
        "scikit-learn>=1.6.1",
        "holidays>=0.64",
    ],
    python_requires=">=3.8",
) 