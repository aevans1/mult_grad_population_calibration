import jax
import jax.numpy as jnp


@jax.jit
def grad_log_prob(weights, likelihood):
    """
    Evaluate the gradient of the log-likelihood of the data given the weights.

    This computes the "probabilistic model" for the image prob density with weights w
    - \sum_m p(y_i |x_m) w_m  
    And then computes the gradient of \sum_i (1/num_images)*log (\sum_m p(y_i|x_m) w_m)
    - (1/num_images)*p(y_i|x_m) / \sum_m p(y_i | x_m) w_m

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

    model = jnp.sum(likelihood*weights, axis=1)
    grad = jnp.mean(likelihood/model[:, jnp.newaxis], axis=0)
    return grad


@jax.jit
def update_weights(weights, grad):
    weights = weights*grad
    return weights


@jax.jit
def update_info(
    weights,
    likelihood,
    reference_weights=None
):
    # TODO: other stats will go in here
    loss = -jnp.mean(jnp.log(likelihood*weights), axis=1)
    return loss


def multiplicative_gradient(
    log_likelihood,
    tol=1e-2,
    max_iterations=100000,
    info_frequency=100,
    VERBOSE=False,
    TRACK_WEIGHTS=False
):
    """
    Optimizes the weights with the multiplicative gradient method.

    Parameters
    ----------
    log_likelihood: jax.Array
        Log-likelihood of generating image i from conformation j.
    tol: float
        Tolerance for the stopping criteria
    max_iterations: int
        Max iterations if stopping criteria isn't met
    info_frequency: int
        Stats are computed at every (stats frequency) iterations
    VERBOSE: bool
        If TRUE, some print statements will happen every info_frequency iterations
    TRACK_WEIGHTS: bool
        If TRUE, the weights at every info_frequency iterations will be saved

    Returns
    -------
    weights: jax.Array 
    """

    num_images, num_structures = log_likelihood.shape

    # Initialize Weights
    weights = (1/num_structures)*jnp.ones(num_structures)

    # Subtracting the largest entry from each row of likelihood
    # The gradient is invariant to row scaling of likelihood, so this is valid
    # With this, we avoid working in log space for the grad and loss
    log_likelihood = log_likelihood - jnp.max(log_likelihood, 1)[:, jnp.newaxis]

    # NOTE: we cannot exponentiate this if previous step hasn't happened!
    likelihood = jnp.exp(log_likelihood)

    # Initialize info tracked
    info = {}
    info["losses"] = []
    info["idx"] = []
    info["weights"] = []
    for k in range(max_iterations):

        # Update info
        if k % info_frequency == 0:
            loss = update_info(weights, likelihood)
            info["losses"].append(loss)
            info["idx"].append(k)
            # info["your_favorite_stat"].append(...)

            if TRACK_WEIGHTS:
                info["weights"].append(weights)

            if VERBOSE:
                print(f"#iterations: {k}")
                print(f"loss: {loss}")
                print("\n")

        # Update weights
        grad = grad_log_prob(weights, likelihood)
        weights = weights*grad

        # Check stopping criterion
        # TODO: swap out?
        gap = jnp.max(grad) - 1
        if gap < tol:
            print("exiting!")
            print(f"#iterations at exit: {k}")
            break

    # Collect stats in array format
    info["losses"] = jnp.stack(info["losses"])
    info["idx"] = jnp.stack(info["idx"])
    if TRACK_WEIGHTS:
        info["weights"] = jnp.stack(info["weights"])
    return weights, info


@jax.jit
def grad_log_prob_in_log_space(weights, log_likelihood):
    """
    Evaluate the gradient of the log-likelihood of the data given the weights.

    NOTE: this function should output the same as grad_log_prob, but keeping this older version in here just in case

    Parameters
    ----------
    log_weights: jax.Array
        Log of the weights of the clusters.
    log_likelihood: jax.Array
        Log-likelihood of generating image i from cluster j.

    Returns
    -------
    grad: jax.array

    """
    num_images, num_structures = log_likelihood.shape

    log_density_at_weights = jax.scipy.special.logsumexp(a=log_likelihood, b=weights, axis=1)
    aux = log_likelihood - log_density_at_weights.reshape(num_images, 1)
    grad = (1/num_images)*(jnp.exp(jax.scipy.special.logsumexp(aux, axis=0)))
    return grad
