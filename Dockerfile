FROM continuumio/miniconda3
WORKDIR /app
COPY . /app
RUN conda env update --file environment.yml
RUN echo \"conda activate vj279_project\" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]
