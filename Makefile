SHELL := /bin/bash

install:
	pip install .

install_code:
	pip install -e .[test]

test:
	pytest --cov-report term-missing --cov=portfawn tests/

uml:
	pyreverse -o png -p portfawn portfawn

install_precommit:
	pip install pre-commit
	pre-commit install

run_precommit:
	pre-commit run --all-files
