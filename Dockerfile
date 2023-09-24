ARG BASE_IMAGE=docker.ops.iszn.cz/mlops/nvidia/cuda:devel
FROM ${BASE_IMAGE}

LABEL maintainer="srch.mlops@firma.seznam.cz" \
      org.label-schema.schema-version="1.0.0" \
      org.label-schema.vendor="Seznam, a.s." \
      org.label-schema.name="kubeflow-ldap-groups" \
      org.label-schema.description="Image pro získání LDAP skupin." \
      org.label-schema.url="https://gitlab.seznam.net/relevance/experiments/"

RUN apt-get update && apt-get install -y perl libtool gettext autoconf automake texinfo autopoint git

ADD aspell /tmp/aspell
ADD aspell-cs-0.51-0  /tmp/aspell-cs-0.51-0
ADD aspell-python  /tmp/aspell-python
ADD code /code

# install fixed Aspell
WORKDIR /tmp/aspell
RUN   ./autogen && \
      ./configure && \
      make && \
      make install
RUN ldconfig

# install Aspell dictionary for Czech
WORKDIR /tmp/aspell-cs-0.51-0
RUN   ./configure && \
      make && \
      make install
ENV LANG=cs_CZ

# install aspell-python
WORKDIR  /tmp/aspell-python
RUN   python setup.3.py build &&\
      python setup.3.py install

# prepare code
WORKDIR /code
RUN python -m pip install -r requirements.txt

RUN mkdir -p /code/src/utils/batch_size/results_store