from setuptools import setup, find_packages

setup(
    name="autohyper",
    version="0.1.0",
    description="Lightweight and modular hyperparameter optimization framework for machine learning models.",
    author="Christian Braga",
    author_email="braga.ch22@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "colorlog==6.9.0",
        "matplotlib==3.10.3",
        "numpy==2.2.6",
        "pandas==2.2.3",
        "scikit-learn==1.6.1",
        "seaborn==0.13.2",
    ],
    python_requires=">=3.11",
)
