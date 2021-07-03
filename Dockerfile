FROM ubuntu:20.04

WORKDIR /home/portfawn

RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y python3-pip && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install -U pip

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

RUN pip3 install .

CMD [ "python3" , "-m", "portfawn"]