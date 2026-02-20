- We recommend installing the project in a virtual environment, such as a python `venv`. An example script for creating a venv `mult_grad_population_calibration` in a parent directory `VENVS_DIR`, and then activating the environment, is
```
python -m venv VENVS_DIR/mult_grad_population_calibration
source VENVS_DIR/mult_grad_population_calibration/bin/activate
```
- After activating a virtual environment: 
  - [install JAX](https://docs.jax.dev/en/latest/installation.html) with either CPU or GPU support. 
  - clone the directory and install the repository package with pip. This can be be done via:
    ```
    git clone https://github.com/aevans1/mult_grad_population_calibration.git
    cd mult_grad_population_calibration
    python -m pip install .
    ```
