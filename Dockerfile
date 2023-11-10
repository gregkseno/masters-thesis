FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3.10-dev python3.10-distutils git curl
RUN ln -s /usr/bin/python3.10 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

RUN curl --silent --show-error --retry 5 https://bootstrap.pypa.io/get-pip.py | python3.10

RUN pip install torch==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install jupyterlab ipywidgets
