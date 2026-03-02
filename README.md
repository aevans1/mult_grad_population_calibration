## Overview
This repo is for method development on using the `multiplicative gradient` algorithm (name from [Renbo Zhao's work](https://arxiv.org/abs/2109.05601)) for estimating mixture model proportions, most directly as implemented in [this paper](https://www.biorxiv.org/content/10.1101/2025.03.27.644168v1) and forthcoming work by the authors (stay tuned!), for estimating properties of conformations in heterogenous cryo-EM datasets.

This repo will contain much more explanation of the methods and diagnostics, so **stay tuned for updates**!

For now, the best explanation for our intended context is in the supplementary material of 
[this paper](https://www.biorxiv.org/content/10.1101/2025.03.27.644168v1), section 2.1 for ensemble reweighting.

### Related Work
This optimization method is equivalent to the expectation maximization algorithm on just mixture weights for a mixture model, where the parameters in the mixtures are kept fixed. It is not a new technique, there is much historical context mentioned in the papers above. However, reframing the problem can be extremely helpful, as it is easier to analyze than expectation maximization in general. More in this in the **stay tuned** 

## Trying the code
For now, it's easiest to run
```
python example_1d_mixture.py
```
to see how this works on a 1d example, and what diagnostics are output.

In this case, the un-observed true data is sampled from a gaussian mixture in 1-D, and the observed data has had gaussian noise added. 

**NOTE**: soon there will be a much more workable library here, thank you for your patience.

## Installation
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
