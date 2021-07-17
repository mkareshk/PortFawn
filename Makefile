install:
	apt install python3-pip python3-setuptools
	pip3 install -r requirements.txt

test:
	docker-compose up --build
	chmod 777 -R results

uml:
	pyreverse -o png -p portfawn portfawn 
