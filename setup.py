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
        "scikit-learn",
        "seaborn",
        "dafin>=0.0.3",
    ],
    python_requires=">=3.10",
    extras_require={
        "quantum": ["dwave-system>=1.25.0", "dwave-neal>=0.6.0"],
        "dev": [
            "pre-commit",
            "pytest",
            "pytest-runner",
            "pytest-cov",
            "parameterized",
            "pylint",
        ],
    },
)
