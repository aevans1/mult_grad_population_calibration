import jax
import jax.numpy as jnp
import pytest
import mult_grad_population_calibration.optimize_weights as opt

# Work in progress unit tests!
# These need to be split up 

# TODO: some Nan tests, for a `badly valued' likelihood matrix
def test_compute_loss():
    num_data = 10000 
    num_nodes = 100
    seed = 1
    key = jax.random.PRNGKey(seed)
    likelihood = jax.random.uniform(key, shape=(num_data, num_nodes)) 
    weights = jax.random.uniform(key, shape=(num_nodes,))
    loss_val = opt.compute_loss(weights, likelihood)   
    assert isinstance(loss_val, jax.Array), "output should be a jax array"
    assert isinstance(loss_val.item(), float), "output should be singleton jax array with a float in there"
    return

# TODO
def test_update_weights():
    return

# TODO
# update_info() may change over time! good to have tests for this
def test_update_info():
    return

# TODO
# e.g: test that this has appropriate errors if the dividing scale is close to 0
def test_scaled_gap():
    return

# TODO: lots of tests in here
def test_multiplicative_gradient():
    return()

# TODO: lots of tests in here
def test_multiplicative_gradient_cross_val():
    return()







