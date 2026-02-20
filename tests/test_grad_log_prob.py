import jax
import jax.numpy as jnp
import pytest
import mult_grad_population_calibration.optimize_weights as opt

# Work in progress unit tests!

# TODO: tests for proper intputs/outputs, methods should fail when given improper inputs. e.g: when given a list or scalr
# TODO: tests for when zero or negative weights or likelihoods are given
# TODO: organize this
def test_grad():
    num_data = 1000 
    num_nodes = 100
    seed = 1
    key = jax.random.PRNGKey(seed)
    likelihood = jax.random.uniform(key, shape=(num_data, num_nodes)) 
    weights = jax.random.uniform(key, shape=(num_nodes,))
    grad = opt.compute_grad(weights, likelihood)
    assert isinstance(grad, jax.Array), "output should be a Jax array"
    assert (grad >= 0).all(), "gradient elements should be positive"
    assert (grad*weights >= 0).all(), "gradient elements times weights should be positive"
    assert jnp.sum(grad*weights) == pytest.approx(1.0) , "grad times weights should sum to 1"
    return 

