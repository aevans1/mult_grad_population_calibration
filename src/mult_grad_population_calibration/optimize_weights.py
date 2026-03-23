import re

import jax
import jax.numpy as jnp

from mult_grad_population_calibration.utils import train_test_split, normalize_log_likeli_to_likeli


@jax.jit
def compute_grad(weights, likelihood):
    """
    Evaluate the gradient of the log-likelihood of the data given the weights.

    This computes the "probabilistic model" for the data prob density with weights w
    - sum_j p(y_i |x_j) w_j 
    And then computes the gradient of (1/num_data)*sum_i log (sum_j p(y_i|x_j) w_j):
    - (1/num_data)*sum_i ((p(y_i|x_j) / sum_k p(y_i | x_k) w_j))

    Parameters
    ----------
    weights : jax.Array
        weights of the nodes
    likelihood : jax.Array
        likelihood of data-point i from node j.
        must be of shape (num_data x num_nodes) 

    Returns
    -------
    gradient of log marginal likelihood: jax.Array
    """
    model = likelihood @ weights
    grad = jnp.mean(likelihood/model[:, jnp.newaxis], axis=0)
    return grad


@jax.jit
def compute_loss(weights, likelihood):
    """
    Computes negative marginal log likelihood loss

    Parameters
    ----------
    weights : jax.Array
         weights of the nodes
    likelihood : jax.Array
        likelihood of data-point i from node j.
        must be of shape (num_data x num_nodes) 

    Returns
    -------
    negative log likelihood: float 
    """
    return -jnp.mean(jnp.log(likelihood @ weights))


@jax.jit
def update_weights(weights, grad):
    """
    Updates weights according to multiplicative gradient algorithm.
    NOTE: this update is positive and sums to 1 without normalization.

    Parameters
    ----------
    weights : jax.Array
        weights of the nodes
    grad : jax.Array
        gradient of weights, same shape as weights

    Returns
    -------
    updated weights: jax.Array  
    """
    return weights*grad


def update_info(weights, likelihood):
    """
    For computing info/diagnostics of weights during iterations of mult. grad.
    NOTE: For now, just loss computed, but could add other things here.

    Parameters
    ----------
    weights : jax.Array
         weights of the nodes
    likelihood : jax.Array
        likelihood of data-point i from node j.
        must be of shape (num_data x num_nodes) 

    Returns
    -------
    loss: scalar
    other diagnostics to track...
    """
    # TODO: other jit-complied stats will go in here
    loss = compute_loss(weights, likelihood)
    return loss


@jax.jit
def compute_scaled_grad_variance(grad, weights, scale):
    """
    Find variance gradient vec, and rescale.
    This is an alternate gap criteria.

    Parameters
    ----------
    grad : jax.Array
        gradient of weights, same shape as weights
    weights : jax.Array
        weights of the nodes
    scale : float
        scaling factor, so that the gap at initial iterate is 1. 

    Returns
    -------
    scaled gap: float
    """
    grad = jnp.where(weights > 0, grad, 0)
    return jnp.sum(weights*(grad - 1)**2) / scale



@jax.jit
def compute_scaled_gap(grad, weights, scale):
    """
    Find maximum index of gradient vec, only at nonzero indices of weights, and rescale.
    This gap is a proxy for convergence, common for convex objectives.

    Parameters
    ----------
    grad : jax.Array
        gradient of weights, same shape as weights
    weights : jax.Array
        weights of the nodes
    scale : float
        scaling factor, so that the gap at initial iterate is 1. 

    Returns
    -------
    scaled gap: float
    """
    grad = jnp.where(weights > 0, grad, 0)
    return (jnp.amax(grad) - 1) / scale


# TODO: behavior for train_test versus gradient gap
def multiplicative_gradient(
    log_likelihood,
    tol=1e-2,
    max_iterations=10000,
    weights_frequency=0,
    train_test_key=None,
    train_test=False,
    verbose=False,
    diagnostic=False,
    scale=True
):
    """
    optimizes the weights with the multiplicative gradient method.

    Parameters
    ----------
    log_likelihood: jax.Array
        log-likelihood of generating data point i from node j.
    tol: float
        tolerance for the stopping criteria
    max_iterations: int
        max iterations if stopping criteria isn't met
    weights_frequency: int
        if larger than 0, weights are saved at every weights_frequency iterations
    train_test_key: jax.PRNGKey
        key for splitting into train, split for train test procedure
    train_test: bool
        If true, a stopping index based on train test procedure will be picked,
        then compared with the gap stopping criteria
    verbose: bool
        if true, some print statements will happen every info_frequency iterations
    diagnostic: bool
        if true, method will go to max_iterations, returning max iteration weights.
        This can be used to diagnose how overfit the max iterations are compared to the 
        weights from train_test or the gap tolerance.
    scale: True
        if true, will scale the first iteration of the gradient gap.
    Returns
    -------
    weights: jax.Array 
    """

    num_data, num_nodes = log_likelihood.shape

    # Initialize weights
    weights = (1/num_nodes)*jnp.ones(num_nodes)

    # Convert log likelihood to likelihood via "soft-max"-ish operation
    likelihood = normalize_log_likeli_to_likeli(log_likelihood)

  
    # Initialize info tracked
    info = {"losses": [], "gaps": [], "grad_variances": [], "weights": []}
    
    # Initialize scaling for gap stopping criteria
    grad_init = compute_grad(weights, likelihood)
    
    if scale: 
        gap_scale = compute_scaled_gap(grad_init, weights, scale=1.0)
        var_scale = compute_scaled_grad_variance(grad_init, weights, scale=1.0)
    else:
        gap_scale = 1.0
        var_scale = 1.0
    
    # Initialize stopping criteria checks.
    # particularly if not doing train_test, treat this as reached already 
    reached_gap = False
    reached_train_test = not train_test

    # Do train test index picking
    if train_test:
        if verbose:
            print("Getting train test stopping index")
        train_test_idx = multiplicative_gradient_train_test(
            train_test_key,
            log_likelihood,
            wait_time=2,
            max_iterations=max_iterations,
            )
        info["train_test_idx"] = train_test_idx
        if verbose: 
            print(f"Validation loss increases at idx: {train_test_idx}")
    
    for k in range(max_iterations):
        # Update info
        loss = update_info(weights, likelihood)
        info["losses"].append(loss)
        # info["your_favorite_stat"].append(...)

        # Check if saving weights
        if weights_frequency > 0 and k % weights_frequency == 0:
            info["weights"].append(weights)

        # Update grad
        grad = compute_grad(weights, likelihood)

        # Check stopping criterions
        gap = compute_scaled_gap(grad, weights, gap_scale)
        grad_variance = compute_scaled_grad_variance(grad, weights, var_scale)
        info["gaps"].append(gap)
        info["grad_variances"].append(grad_variance)

        # Check current gap against tolerance
        if not reached_gap and gap < tol:
            info["gap_idx"] = k
            info["weights_gap"] = weights
            reached_gap = True
            if verbose:
                print(f"reached gap tolerance, at idx: {k}")
                print(f"gap: {gap}")
         
        # Check current index against the train_test stopping index
        if train_test:
            if k == train_test_idx:
                info["weights_train_test"] = weights
                reached_train_test = True

        # Check if all stopping criteria met
        if reached_train_test and reached_gap and not diagnostic:
            if verbose:
                print(f"exiting! At iteration: {k}")
            break

        # Update weights
        weights = update_weights(weights, grad)

    # Collect info in array format, and save weights and corresponding indices if requested
    info["final_idx"] = k
    info["losses"] = jnp.stack(info["losses"])
    info["gaps"] = jnp.stack(info["gaps"])
    info["grad_variances"] = jnp.stack(info["grad_variances"])
    if weights_frequency > 0:
        info["weights"] = jnp.stack(info["weights"])
        info["weights_idx"] = jnp.arange(len(info["weights"]))*weights_frequency

    if not reached_gap:
        print("Terminated at max iters: ")
        print("Returned weights & 'info[weights_gap']' are weights at max_iterations")
        info["weights_gap"] = weights
        info["gap_idx"] = k
    return weights, info


def multiplicative_gradient_train_test(
    key, 
    log_likelihood,
    wait_time=2,
    max_iterations=10000,
    train_pct=0.8,
    smooth_val=0.3
):
    """
    Rudimentary train test split for finding a stopping index 
    of multiplicative gradient.

    Parameters
    ----------
    key: jax.PRNGKey
        key for splitting into train, split for train test procedure
    log_likelihood: jax.Array
        log-likelihood of generating data point i from node j
    wait_time: int
        how many increases in validation loss before stopping
    max_iterations: int
        max iterations if stopping criteria isn't met
    train_pct: float
        percentage of dataset used for training data
    smooth_val: float
        smoothing parameter for exponential smoothing of validation loss
    Returns
    -------
    stopping_idx: int
    """
    num_data, num_nodes = log_likelihood.shape
    
    log_likelihood_train, log_likelihood_test, _, _ = train_test_split(key,
                                                                      log_likelihood,
                                                                      train_pct)

    likelihood_train = normalize_log_likeli_to_likeli(log_likelihood_train)
    likelihood_test = normalize_log_likeli_to_likeli(log_likelihood_test)

    # Initialize weights
    weights = (1/num_nodes)*jnp.ones(num_nodes)

    # Initialize train test procedure
    count = 0
    smoothed_val_loss = compute_loss(weights, likelihood_test)
    for k in range(max_iterations):

        # Update grad
        grad = compute_grad(weights, likelihood_train)

        # Update weights
        weights_new = update_weights(weights, grad)

        # Update smoothed_loss
        val_loss_new = compute_loss(weights_new, likelihood_test)

        # Smooth, if iterated past the soft assignment weights
        if k > 1:
            smoothed_val_loss_new = (smooth_val)*val_loss_new + (1-smooth_val)*smoothed_val_loss
        else:
            smoothed_val_loss_new = val_loss_new 

        # Check stopping criterion: increase in (smoothed) validation loss
        val_losses_diff = smoothed_val_loss_new - smoothed_val_loss
        if val_losses_diff > 0:
            count += 1
        if count >= wait_time:
            break
        weights = weights_new
        smoothed_val_loss = smoothed_val_loss_new 
    
    if k==max_iterations-1: 
        print("NOTE: Train-test stopping criterion not reached.")
    return k

