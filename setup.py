import setuptools

with open("README.md", "r") as file:
    long_description = file.read()


setuptools.setup(
    name="PortFawn",  # Replace with your own username
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
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "pandas",
        "joblib",
        "yfinance",
        "pytest",
        "dwave-neal",
        "sklearn",
    ],
    python_requires=">=3.8",
)
