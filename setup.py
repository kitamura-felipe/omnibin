from setuptools import setup, find_packages

setup(
    name="omnibin",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "seaborn>=0.11.0",
    ],
    author="Felipe Campos Kitamura",
    author_email="kitamura.felipe@gmail.com",
    description="A package for generating comprehensive binary classification reports with visualizations and confidence intervals",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kitamura-felipe/omnibin",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 