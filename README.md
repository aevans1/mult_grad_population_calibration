- We recommend installing the project in a virtual environment, such as a python `venv`. An example script for creating a venv `parsimonious_ensembles` in a parent directory `VENVS_DIR`, and then activating the environment, is
```
python -m venv VENVS_DIR/parsimonious_ensembles
source VENVS_DIR/parsimonious_ensembles/bin/activate
```
- After activating a virtual environment: 
  - [install JAX](https://docs.jax.dev/en/latest/installation.html) with either CPU or GPU support. 
  - clone the directory and install the repository package with pip. This can be be done via:
    ```
    git clone https://github.com/aevans1/parsimonious_ensembles.git
    cd parsimonious_ensembles
    python -m pip install .
    ```
