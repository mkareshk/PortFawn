install:
	apt install python3-pip python3-setuptools
	pip3 install -r requirements.txt
run:
	python3 -m portfolio_optimization
uml:
	pyreverse -o png -p portfolio_optimization portfolio_optimization 
