import setuptools

with open("README.md", "r") as file:
    long_description = file.read()


setuptools.setup(
    name="PortFawn",
    version="0.0.1",
    author="Moein Kareshk",
    author_email="mkareshk@outlook.com",
    description="Portfolio Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mkareshk/portfawn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "dimod>=0.10.7",
        "dwave-neal>=0.5.8",
        "joblib>=1.1.0",
        "matplotlib>=3.4.3",
        "numpy>=1.21.3",
        "pandas>=1.3.4",
        "scikit-learn>=1.0.1",
        "scipy>=1.7.1",
        "seaborn>=0.11.2",
        "yfinance>=0.1.64",
    ],
    python_requires=">=3.8",
)
