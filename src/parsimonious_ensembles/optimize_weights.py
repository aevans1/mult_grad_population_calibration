import jax
import jax.numpy as jnp

from parsimonious_ensembles.utils import cross_val_split, find_increase

@jax.jit
def grad_log_prob(weights, likelihood):
    """
    Evaluate the gradient of the log-likelihood of the data given the weights.

    This computes the "probabilistic model" for the image prob density with weights w
    - \sum_m p(y_i |x_j) w_j 
    And then computes the gradient of \sum_i (1/num_images)*log (\sum_m p(y_i|x_j) w_j)
    - (1/num_images)*p(y_i|x_j) / \sum_m p(y_i | x_j) w_j

    Parameters
    ----------
    weights: jax.Array
        weights of the structures.
    likelihood: jax.Array
        (unnormalized)_likelihood of generating image i from cluster j.
        must be of shape (num_images x num_structures) 

    Returns
    -------
    gradient of log marginal likelihood: jax.Array
    """

    model = likelihood @ weights
    grad = jnp.mean(likelihood/model[:, jnp.newaxis], axis=0)
    return grad


@jax.jit
def update_weights(weights, grad):
    weights = weights*grad
    return weights

@jax.jit
def compute_loss(weights, likelihood):
    #return -jnp.mean(jnp.log(jnp.sum(likelihood*weights, axis=1)))
    return -jnp.mean(jnp.log(likelihood @ weights))

@jax.jit
def compute_loss_log_likelihood(weights, log_likelihood):
    return -jnp.mean(jax.scipy.special.logsumexp(a=log_likelihood,b=weights[None, :], axis=1))

@jax.jit
def update_info(
    weights,
    likelihood,
    reference_weights=None
):
    # TODO: other stats will go in here
    loss = compute_loss(weights, likelihood)
    return loss

@jax.jit
def scaled_gap(grad, weights, scale):
    """
    Find maximum of gradient, only at nonzero indices of weights, and rescale 
    """
    grad = jnp.where(weights > 0, grad, 0)
    return (jnp.amax(grad) - 1) / scale


def multiplicative_gradient(
    log_likelihood,
    tol=1e-2,
    max_iterations=100000,
    weights_frequency=0,
    split_seed = 119,
    cross_val=True,
    verbose=False,
    
):
    """
    optimizes the weights with the multiplicative gradient method.

    parameters
    ----------
    log_likelihood: jax.array
        log-likelihood of generating image i from conformation j.
    tol: float
        tolerance for the stopping criteria
    max_iterations: int
        max iterations if stopping criteria isn't met
    info_frequency: int
        stats are computed at every (stats frequency) iterations
    weights_frequency: int
        if larger than 0, weights are saved at every weights_frequency iterations
    split_seed: int
        seed for splitting into train, split for cross validation
    cross val: bool
        If true, a stopping index based on cross validation will be picked,
        then compared with the gap stopping criteria
    verbose: bool
        if true, some print statements will happen every info_frequency iterations
    
    returns
    -------
    weights: jax.array 
    """

    num_images, num_structures = log_likelihood.shape

    ## initialize weights
    weights = (1/num_structures)*jnp.ones(num_structures)

    ## subtracting the largest entry from each row of likelihood
    ## the gradient is invariant to row scaling of likelihood, so this is valid
    ## with this, we avoid working in log space for the grad and loss
    log_likelihood = log_likelihood - jnp.max(log_likelihood, 1)[:, jnp.newaxis]

    # note: we cannot exponentiate this if previous step hasn't happened!
    likelihood = jnp.exp(log_likelihood)

    # initialize scaling for gap stopping criteria
    gap_scale = scaled_gap(grad_log_prob(weights, likelihood), weights, scale=1)

    # Do cross_validation index picking
    print("Getting cross validation stopping index")
    cross_val_idx = multiplicative_gradient_cross_val(
                    log_likelihood,
                    lag=2,
                    max_iterations=max_iterations,
                    split_seed=split_seed,
                    train_pct = 0.8)   

    # initialize info tracked
    info = {}
    info["losses"] = []
    info["gaps"] = []
    info["gap_idx"] = max_iterations-1
    info["cross_val_idx"] = cross_val_idx
    info["weights"] = []
    for k in range(max_iterations):
        # update info
        loss = update_info(weights, likelihood)
        info["losses"].append(loss)
        # info["your_favorite_stat"].append(...)

        if weights_frequency > 0 and k % weights_frequency == 0:
            info["weights"].append(weights)

        if verbose:
            print(f"#iterations: {k}")
            print(f"loss: {loss}")
            print("\n")

        ## update grad
        grad = grad_log_prob(weights, likelihood)

        ## check stopping criterion
        gap = scaled_gap(grad, weights, gap_scale)
        info["gaps"].append(gap)
        
        #TODO: make stopping that hits both cross val and grad stopping 
        #if gap < tol:
        #    print(f"gap tolerance met, at idx: {k}")
        ##    print("exiting")
        #    info["gap_idx"] = k
        #    info["weights_gap_idx"] = weights
        
        #if k == cross_val_idx and gap :
        #    print("exiting")
        #    break  
        
        #if gap < tol:
        #    print(f"gap tolerance met, at idx: {k}")
        #    print("exiting")
        #    info["gap_idx"] = k
        #    info["weights_gap_idx"] = weights
        #    break

        ## update weights
        weights = update_weights(weights, grad)

    ## collect info in array format, and save weights and corresponding indices if requested
    info["final_idx"] = k
    info["losses"] = jnp.stack(info["losses"])
    info["gaps"] = jnp.stack(info["gaps"])
    if weights_frequency > 0:
        info["weights"] = jnp.stack(info["weights"])
        info["weights_idx"] = jnp.arange(0, k, weights_frequency)
    return weights, info


def multiplicative_gradient_cross_val(
    log_likelihood,
    lag=2,
    max_iterations=10000,
    split_seed=298,
    train_pct = 0.8
):
    """
    Rudimentary cross validation for finding a stopping index 
    of multiplicative gradient.

    parameters
    ----------
    log_likelihood: jax.array
        log-likelihood of generating image i from conformation j.
    max_iterations: int
        max iterations if stopping criteria isn't met
    
    returns
    -------
    stopping_idx: int
    """
    num_images, num_structures = log_likelihood.shape

    key = jax.random.PRNGKey(split_seed)
    log_likelihood_train, log_likelihood_test, _, _ = cross_val_split(key, 
                                                                      log_likelihood, 
                                                                      train_pct)

    ## subtracting the largest entry from each row of likelihood
    ## the gradient is invariant to row scaling of likelihood, so this is valid
    ## with this, we avoid working in log space for the grad and loss
    log_likelihood_train -= jnp.amax(log_likelihood_train, axis=1)[:, None]
    log_likelihood_test -= jnp.amax(log_likelihood_test, axis=1)[:, None]

    ## initialize weights
    weights = (1/num_structures)*jnp.ones(num_structures)

    # note: we cannot exponentiate this if previous step hasn't happened!
    likelihood_train = jnp.exp(log_likelihood_train)
    likelihood_test = jnp.exp(log_likelihood_test)

    # Initialize cross validation
    count = 0
    for k in range(max_iterations):
    
        ## update grad
        grad = grad_log_prob(weights, likelihood_train)

        ## update weights
        weights_new = update_weights(weights, grad)

        ## check stopping criterion: increase in validation loss
        losses_diff = compute_loss(weights_new, likelihood_test) - compute_loss(weights, likelihood_test)
        print(losses_diff)
        if losses_diff > 0:
            print('increase')
            count +=1 
        else:
            count = 0
        if count > lag:
            break
        weights = weights_new
    
    return k



