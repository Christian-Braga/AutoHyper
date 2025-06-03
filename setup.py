from setuptools import setup, find_packages

setup(
    name="autohyper",
    version="0.1.0",
    author="Christian Braga",
    description="A lightweight framework for hyperparameter optimization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "pandas",
        "xgboost",
        "scikit-learn",
        "matplotlib",
        "seaborn",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
