# Build basic Dockerfile
touch Dockerfile
# use basic miniconda image
echo "FROM continuumio/miniconda3" >> Dockerfile
echo "" >> Dockerfile
# create project directory
echo "RUN mkdir -p $repo_name" >> Dockerfile
echo "" >> Dockerfile
# copy the repository in
echo "COPY . /$repo_name" >> Dockerfile
echo "WORKDIR /$repo_name" >> Dockerfile
echo "" >> Dockerfile
# install the conda environment
echo "RUN conda env update --file environment.yml" >> Dockerfile
echo "" >> Dockerfile
# activate the conda environment
# can't do it with dockerfile
# instead we have to edit bashrc to load it on login
echo "RUN echo \"conda activate $repo_name\" >> ~/.bashrc" >> Dockerfile
echo "SHELL [\"/bin/bash\", \"--login\", \"-c\"]" >> Dockerfile
echo "" >> Dockerfile
# as we are in the conda enviroment we can install pre-commit hooks
echo "RUN pre-commit install" >> Dockerfile

echo "Creating Git repository..."
echo "**********************************************"
# Initialize the Git repository
if hash git 2>/dev/null; then # check if git is installed
    git init
    git add .
    git commit -m "Initial commit"
else
    echo "Git not installed, exiting..."
    exit 1
fi
