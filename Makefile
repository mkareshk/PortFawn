install:
	apt install python3-pip python3-setuptools
	pip3 install -r requirements.txt

test:
	docker-compose up --build
	chmod 777 -R results

test_cov:
	pytest --cov-report term-missing --cov=portfawn tests/

uml:
	pyreverse -o png -p portfawn portfawn 

test_cov_docker:
	docker build -t portfawn:test -f Dockerfile.test .
	docker run  portfawn:test