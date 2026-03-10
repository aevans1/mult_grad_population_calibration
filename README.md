## Overview
This repo is for method development on the Multiplicative Gradient for Population Calibration (MGPC) framework for estimating mixture proportions, most directly as implemented in [this paper](https://www.biorxiv.org/content/10.1101/2025.03.27.644168v1) and forthcoming (stay tuned!), for calibrating conformational probabilities in heterogenous cryo-EM datasets. We refer to this as a **calibration** procedure as opposed to estimation, because we assume that a lot of hard work has already been done - estimating conformations and computing likelihood matrices - which we update to better fit data but without overfitting.

This repo will contain much more explanation of the methods and diagnostics, so stay tuned for updates!

For now, the best explanation for our intended context is in the supplementary material of 
[this paper](https://www.biorxiv.org/content/10.1101/2025.03.27.644168v1), section 2.1 for ensemble reweighting.

### Related Work
#### Optimization 
The baseline optimization method in MGPC is equivalent to the expectation maximization algorithm on just mixture weights for a mixture model, where the parameters in the mixtures are kept fixed. It is not a new technique, there is much historical context mentioned in the papers above. However, reframing the problem can be extremely helpful, as it is easier to analyze than expectation maximization in general. Further, our framework allows for various regularization and cross-validation strategies tailored to noisy datasets such as cryo-EM.
There will be future updates that expound on this.

#### Software
- This library was initiated as the `multiplicative_gradient` code in the repo [counting_particles_paper](https://github.com/aevans1/counting_particles_paper)
- This library will be implemented as an extension usable with likelihood computation via the [cryojax](https://github.com/michael-0brien/cryojax) library. Again, stay tuned for more updates.
## Trying the code
For now, it's easiest to run
```
python example_1d_mixture.py
```
to see how this works on a 1d example, and what diagnostics are output. For the example, you may notice that the gradient `gap` can be an erratic quantity comparatively to the loss!

In this example, the un-observed true data is sampled from a gaussian mixture in 1-D, and the observed data has had gaussian noise added. 

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
## Documentation
Stay tuned!