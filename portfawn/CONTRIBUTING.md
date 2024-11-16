# Contributing to Portfawn

We are excited that you want to contribute to Portfawn! Whether it's reporting bugs, suggesting new features, improving documentation, or developing new features, your contributions are invaluable. This guide outlines the process to contribute effectively and ensure a smooth collaboration.


## How Can You Contribute?

### 1. Report Issues
If you encounter a bug, have a question, or want to suggest a feature, please open an issue in the GitHub repository:
- Include a detailed description of the problem or suggestion.
- Provide steps to reproduce the issue (if applicable).
- Attach any relevant logs or screenshots.

### 2. Improve Documentation
We welcome enhancements to the documentation:
- Fix typos or errors in the README or other docs.
- Add new tutorials or examples.
- Clarify existing instructions.

### 3. Submit Code Contributions
If you're ready to dive into the codebase, here's how to make a contribution:
- Fix bugs.
- Implement new features.
- Optimize existing functionality.
- Add or improve tests.


## Development Workflow

### Step 1: Fork the Repository
1. Visit the [Portfawn repository](https://github.com/mkareshk/portFawn).
2. Click the "Fork" button to create your own copy of the repository.

### Step 2: Clone Your Fork
```bash
git clone https://github.com/<your-username>/portfawn.git
cd portfawn
```

### Step 3: Set Up the Environment
Use your favorite way of managing Python environments, for example:
```bash
python -m venv venv-portfawn
source ./venv-portfawn/bin/activate
```

Then, install the package locally:
```bash
make install_dev
```

### Step 4: Create a Feature Branch
Create a new branch for your work:
```bash
git checkout -b feature-name
```

### Step 5: Make Your Changes
1. Follow the code structure and conventions in the existing codebase.
2. Add or update unit tests to cover your changes.

Run tests locally:
```bash
make test
```

### Step 6: Commit Your Changes
1. Write clear and descriptive commit messages.
2. Include relevant issue numbers in your commit messages, if applicable.

```bash
git add .
git commit -m "Fix issue #123: Description of fix"
```

### Step 7: Push Your Changes
Push your branch to your fork:
```bash
git push origin feature-name
```

### Step 8: Open a Pull Request
1. Go to the original repository on GitHub.
2. Click "New Pull Request" and select your branch.
3. Describe your changes and link to relevant issues.


## Code Style Guidelines

- **Python Style**: Follow [PEP 8](https://peps.python.org/pep-0008/).
- **Type Annotations**: Use type hints wherever possible.
- **Docstrings**: Include clear and concise docstrings in your functions and classes. Use Google-style docstrings.

Example:
```python
def add(a: int, b: int) -> int:
    """
    Add two numbers together.

    Args:
        a (int): The first number.
        b (int): The second number.

    Returns:
        int: The sum of the two numbers.
    """
    return a + b
```


## Testing

All new features or fixes should include appropriate tests. We use **pytest** for testing.

Run the full test suite locally:
```bash
make test
```


## Pull Request Guidelines

1. Ensure your PR passes all tests and CI checks.
2. Include detailed descriptions of your changes.
3. Keep your PR focused on a single issue or feature.
4. Use descriptive titles and link to relevant issues.


## Community Guidelines

- Be respectful and inclusive.
- Follow [GitHub's Community Guidelines](https://docs.github.com/en/github/site-policy/github-community-guidelines).


## Need Help?

If you need help at any step of the contribution process:
- Open an issue and describe your question.
- Reach out to us at **mkareshk@outlook.com**.

We look forward to your contributions! 🎉
