import jax.numpy as jnp
import jax


def train_test_split(key, log_likelihood, train_pct=0.8):
    """
    Splits log likelihood into two sets, based on rows (images / data points)

    Parameters
    ----------
    key : jax.Array
        key from jax.random.PRNGKey(seed)
    log_likelihood : jax.Array
        log_likelihood of data-point i from node j.
        must be of shape (num_data x num_nodes) 
    train_pct : float, optional
        split percentage of data by rows, by default 0.8

    Returns
    -------
    train : jax.Array
        training split 
    test : jax.Array
        test split
    train_idx : jax.Array
        indices of original array used for train split
    test_idx : jax.Array
        indices of original array used for test split
    """
    num_data = log_likelihood.shape[0]
    split_size = int(jnp.ceil(train_pct*log_likelihood.shape[0]))
    train_idx = jax.random.choice(
        key,
        num_data,
        (split_size,),
        replace=False)
    test_idx = jnp.setdiff1d(jnp.arange(num_data), train_idx)
    train = log_likelihood[train_idx, :]
    test = log_likelihood[test_idx, :]
    return train, test, train_idx, test_idx


@jax.jit
def normalize_log_likeli_to_likeli(log_likelihood):
    """
    Subtracts the largest entry from each row of  the log likelihood.
    This is for stability, before transforming to likelihood (like in a soft-max) 
    The gradient is invariant to row scaling of likelihood, so this is valid.
    With this normalizing, we avoid working in log space for the grad and loss.

    Parameters
    ----------
    log_likelihood : jax.Array
        log_likelihood of data-point i from node j.
        must be of shape (num_data x num_nodes) 

    Returns
    -------
    likelihood: jax.Array
    """
    log_likelihood -= jnp.amax(log_likelihood, axis=1)[:, None]
    likelihood = jnp.exp(log_likelihood)
    return likelihood

