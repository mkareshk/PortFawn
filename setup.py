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
        "dwave-system>=1.15.0",
        "dwave-neal>=0.5.9",
        "scikit-learn>=1.1.1",
        "seaborn>=0.11.2",
        "pyod>=1.0.3",
        "dafin"
    ],
    python_requires=">=3.10",
    extras_require={
        "test": ["pytest", "pytest-cov", "parameterized", "pylint"],
    },
)
