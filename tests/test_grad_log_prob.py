import jax
import jax.numpy as jnp
from parsimonious_ensembles.optimize_weights import compute_grad

# TODO: tests for proper intputs/outputs, methods should fail when given improper inputs. e.g: when given a list or scalr

def test_grad():
    num_data = 1000 
    num_nodes = 100
    for i in range(5): 
        seed = i
        key = jax.random.PRNGKey(seed)
        likelihood = jax.random.uniform(key, shape=(num_data, num_nodes)) 
        weights = jax.random.uniform(key, shape=(num_nodes,))
        grad = compute_grad(weights, likelihood)
        assert (grad >= 0).all(), "gradient elements should be positive"
        assert (grad*weights >= 0).all(), "gradient elements times weights should be positive"
        print(jnp.sum(grad*weights))
        assert jnp.abs(jnp.sum(grad*weights == 1)) < 1e-10 , "grad times weights should sum to 1"
    return 


# straightforward
def test_compute_loss():
    return

# straightforward
def test_update_weights():
    return

# this function may change over time! good to have tests for this
def test_update_info():
    return

# e.g: test that this has appropriate errors if the dividing scale is close to 0
def test_scaled_gap():
    return

# TODO: lots of tests in here
def test_multiplicative_gradient():
    return()

# TODO: lots of tests in here
def test_multiplicative_gradient_cross_val():
    return()







