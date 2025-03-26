FROM nvcr.io/nvidia/pytorch:25.01-py3

# Install Miniconda
RUN apt-get update && apt-get install -y wget
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
RUN /bin/bash /tmp/miniconda.sh -b -p /opt/conda
RUN rm -rf /tmp/*

# Set environment variables
ENV PATH /opt/conda/bin:$PATH

# Copy environment.yml into the container
COPY environment.yml .

# Update the environment
RUN conda env update --file environment.yml && conda clean -tipy

# Activate the environment (optional, for interactive sessions)
# RUN conda init bash
# RUN echo "conda activate base" >> /root/.bashrc
