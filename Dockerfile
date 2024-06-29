FROM continuumio/miniconda3:24.5.0-0
WORKDIR /app
COPY . /app
# Run the setup script
RUN ./setup.sh
RUN echo \"conda activate vj279_project_dev\" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]



# FROM continuumio/miniconda3

# RUN apt-get update
# RUN apt-get install -y build-essential python3-dev

# COPY src /m2_cw/src
# COPY configs /m2_cw/configs
# COPY environment.yml /m2_cw/environment.yml

# WORKDIR /m2_cw

# RUN conda env update --file environment.yml

# RUN echo "conda activate m2_cw_env" >> ~/.bashrc
# SHELL ["/bin/bash", "--login", "-c"]
