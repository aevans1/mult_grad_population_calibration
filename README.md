## Overview
This repo is for method development on the Multiplicative Gradient for Population Calibration (MGPC) framework for estimating mixture proportions, most directly as implemented in [this paper](https://www.biorxiv.org/content/10.1101/2025.03.27.644168v1) and forthcoming (stay tuned), for calibrating conformational probabilities in heterogenous cryo-EM datasets. We refer to this as a **calibration** procedure as opposed to estimation, because we assume that a lot of hard work has already been done - estimating conformations and computing likelihood matrices - which we update to better fit data but without overfitting.

This repo will contain much more explanation of the methods and diagnostics, so stay tuned for updates!

For now, the best explanation for our intended context is in the supplementary material of 
[this paper](https://www.biorxiv.org/content/10.1101/2025.03.27.644168v1), section 2.1 for ensemble reweighting.
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
## Trying the code
### Log Likelihood matrix
The main function [`muliplicative_gradient`](https://github.com/aevans1/mult_grad_population_calibration/blob/main/src/mult_grad_population_calibration/optimize_weights.py) requires a `log_likelihood` matrix as input. `log_likelihood` must be a `jax.Array` with `num_data` rows  and `num_nodes` columns. It's expected that entry `log_likelihood[i, j]` corresponds to `log p(y_|x_j)` for a data point `y_i`, node `x_j` and some likelihood function `p(y|x)`. For the cryo-EM settings the "data" are the images and the "nodes" are the conformations.

**NOTE**: It is crucial that the above matrix is a log-likelihood matrix and not a negative log-likelihood matrix.  

### Default Options
The below example code runs the default options, with a default set `max_iterations` and stopping tolerance, and no extra stopping criteria or saving of weights. 
```
import jax
import jax.numpy as jnp
import mult_grad_population_calibration.optimize_weights as opt

# log likelihood is (num_data x num_nodes) jax.numpy array
weights, info = opt.multiplicative_gradient(log_likelihood)
```
The outputs of `multiplicative gradient` are:
- `weights`: `jax.Array`, the optimized weights from multiplicative gradient.
- `info`: `Dict` of information from optimization. By default, has the fields:
  - `losses`: `jax.Array` of the loss (negative marginal log likelihood) values per iteration.
  - `gaps`: `jax.Array` of the gradient gap at each iteration. This `gap` is used for stopping the iteration.
  - `weights_history`: `jax.Array` of weights computed at every `weights_frequency` iterations. Empty if `weights_frequency=0`.
  - `final_idx`: `int` of last index reached in simulation.

### Train-Test Stopping Criteria
The below example code optimizes with an additional stopping criteria, a `train_test` split used to estimate overfitting.
```
import jax
import jax.numpy as jnp
import mult_grad_population_calibration.optimize_weights as opt

seed_train_test = 0
key = jax.random.key(seed_train_test)

# log likelihood is a (num_data x num_nodes) jax.numpy array
weights, info = opt.multiplicative_gradient(log_likelihood,
                                            train_test_key=key,
                                            train_test=True)
```
The output `weights` will now return whichever stopping criteria stopped first.
Both the default `gap` criteria weights and the train-test weights are returned in `info`.

The new outputs returned from `info` are
- `train_test_idx`: `int` of the index where the train-test riteria has stopped the iteration.
- `weights_train_test`: `jax.Array` of the weights from the train-test stopping criteria, at `train_test_idx`. 
- `gap_idx`: `int` of the index where the default `gap` criteria stopped the iteration.
- `weights_gap`:`jax.Array` of the weights from the the default `gap` criteria.

### Saving weights and running to max iterations
The below example code optimizes which runs to max iterations, sets a custom tolerance, and saves weights for checking the history of the optimization.
```
import jax
import jax.numpy as jnp
import mult_grad_population_calibration.optimize_weights as opt

seed_train_test = 0
key = jax.random.key(seed_train_test)

# log likelihood is a (num_data x num_nodes) jax.numpy array
weights, info = opt.multiplicative_gradient(log_likelihood,
                                            max_iterations=1298,
                                            tol=0.08,
                                            diagnostic=True)
```
The output `weights` will now return the weights at the maximum number of iterations.
The default `gap` criteria weights and the train-test weights (if set to true above) are returned in `info`.

The new outputs returned from `info` are
- `train_test_idx`: `int` of the index where the train-test riteria has stopped the iteration.
- `weights_train_test`: `jax.Array` of the weights from the train-test stopping criteria, at `train_test_idx`. 
- `gap_idx`: `int` of the index where the default `gap` criteria stopped the iteration.
- `weights_gap`:`jax.Array` of the weights from the the default `gap` criteria.

### Quick Script for Exploration and Diagnostics
To mess with the all various options on a quick example, you can try messing with the `[example_1d_mixture](https://github.com/aevans1/mult_grad_population_calibration/blob/main/example_1d_mixture.py)script:
```
python example_1d_mixture.py
```
This is fast running example where its easier to see how the methods work on a 1d example, and what diagnostics are output.
In that example, the un-observed true data is sampled from a gaussian mixture in 1-D, and the observed data has had gaussian noise added. 

This example above is a work progress, and will be updated in the future for clarity.
## Related Work
### Optimization 
The baseline optimization method in MGPC is equivalent to the expectation maximization algorithm on just mixture weights for a mixture model, where the parameters in the mixtures are kept fixed. It is not a new technique, there is much historical context mentioned in the papers above. However, reframing the problem can be extremely helpful, as it is easier to analyze than expectation maximization in general. Further, our framework allows for various regularization and cross-validation strategies tailored to noisy datasets such as cryo-EM.
There will be future updates that expound on this.

### Software
- This library was initiated as the `multiplicative_gradient` code in the repo [counting_particles_paper](https://github.com/aevans1/counting_particles_paper)
- This library will be implemented as an extension usable with likelihood computation via the [cryojax](https://github.com/michael-0brien/cryojax) library. Again, stay tuned for more updates.
