SHELL := /bin/bash

install:
	pip install .

install_code:
	pip install -e .[test]

test:
	pytest --cov-report term-missing --cov=portfawn tests/

uml:
	pyreverse -o png -p portfawn portfawn 