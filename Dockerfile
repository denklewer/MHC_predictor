FROM continuumio/miniconda3:4.10.3

MAINTAINER Denis Kleverov
ARG DEBIAN_FRONTEND=noninteractive

ENV PATH /opt/conda/bin:$PATH

# Resolve issue with undefined user name for LDAP users, e.g remove warnings
#  group..
#  username..
RUN apt-get update && \
 apt-get upgrade -y && \
 apt-get install -y --no-install-recommends libnss-sss

# Solve locale issues when running bash.
#   /bin/bash: warning: setlocale: LC_ALL: cannot change locale (en_US.UTF-8)
#
# It breaks conda version check in snakemake:
RUN apt-get clean && apt-get update && apt-get install -y locales && \
    echo "LC_ALL=en_US.UTF-8" >> /etc/environment  && \
    echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen  && \
    echo "LANG=en_US.UTF-8" > /etc/locale.conf  && \
    locale-gen en_US.UTF-8

# JAVA 8 missing fonts workaround:
# see https://bugs.debian.org/793210
# and https://github.com/docker-library/java/issues/46#issuecomment-119026586
# P.S: our java comes from conda, seems the same issue
RUN apt-get update && apt-get install -y --no-install-recommends libfontconfig1

#  GNU/GCC compiler and related tools (such as make, debugger, man pages) collection
RUN apt-get update && apt-get install -y build-essential autoconf automake

# Install additional tools: ps, htop, vim, mc
RUN apt-get update && apt-get install -y procps htop vim mc nano tmux screen tree less wget

# Install additional dev tools: git
RUN apt-get update && apt-get install -y git

RUN apt-get update && apt-get install -y locate
RUN apt-get install -y graphviz graphviz-dev


# Fix PATH and profile when running in LSF cluster
RUN echo "PATH=/opt/conda/condabin:/opt/conda/bin:\$PATH" > /opt/conda/.bashrc && \
    cat /root/.bashrc >> /opt/conda/.bashrc

RUN conda create --name pomegranate --yes python==3.9

# Simulate `conda init` and write config to global bash config. It is needed for docker containers
# which are launched with custom user folder and thus lack of original ~/.bashrc settings
RUN echo "export  PATH=/opt/conda/condabin:\$PATH" > /etc/bash.bashrc && \
    echo "# >>> conda initialize >>>" >> /etc/bash.bashrc && \
    echo "__conda_setup=\"\$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)\"" >> /etc/bash.bashrc && \
    echo "eval \"\$__conda_setup\"" >> /etc/bash.bashrc && \
    echo "unset __conda_setup" >> /etc/bash.bashrc && \
    echo "# <<< conda initialize <<<" >> /etc/bash.bashrc
	
# Make RUN commands use the new environment:
RUN echo "conda activate pomegranate" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]
	
RUN cd $HOME
RUN pip install --no-input cython pandas wheel nose numpy scipy sphinx networkx networkx pyyaml
RUN pip install --no-input jupyterlab
RUN pip install --no-input mhcgnomes mhcnames tqdm logomaker pyvis


RUN git clone https://github.com/denklewer/pomegranate
RUN cd pomegranate && python setup.py develop

ADD D:/Projects/PyCharm Projects/MHC_predictor /MHC_predictor
ADD D:/Google Drive/Work/ITMO/data/length_dependent_hmm/per_length_per_kfold_split /data



CMD ["/bin/bash", "-l"]

