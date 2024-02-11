FROM docker.ops.iszn.cz/mlops/petr/train:base

ENV PATH="/root/miniconda3/bin:${PATH}"
RUN bash /root/miniconda3/etc/profile.d/conda.sh 
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]


RUN git clone https://github.com/ufal/errant_czech.git

RUN cd errant_czech && \
    python -m pip install -e . && \
    python -m pip install -r errant/cs/requirements.txt && \ 
    curl --remote-name-all 'https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1674{/czech-morfflex-pdt-160310.zip}' && \
    unzip czech-morfflex-pdt-160310.zip && \
    cp czech-morfflex-pdt-160310/czech-morfflex-160310.dict errant/cs/resources/