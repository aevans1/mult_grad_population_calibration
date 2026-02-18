import jax.numpy as jnp
import jax

def cross_val_split(key, log_likelihood, train_pct=0.8):
    """
    Splits log likelihood into two sets, based on rows (images)
    """
    num_images = log_likelihood.shape[0]
    split_size = int(jnp.ceil(train_pct*log_likelihood.shape[0]))
    train_idx = jax.random.choice(key, num_images, (split_size,), replace=False) 
    test_idx = jnp.setdiff1d(jnp.arange(num_images), train_idx)
    train = jnp.copy(log_likelihood[train_idx, :])
    test = jnp.copy(log_likelihood[test_idx, :])
    return train, test, train_idx, test_idx

def find_increase(losses_diff):
    """
    Finds where a loss curve increases `lag' times in a row.
    This is used for rudimentary cross-validation 
    """
    lag = 2
    count = 0
    idx = 0
    my_array = jnp.where(losses_diff > 0, 1, 0)
    while count < lag:
        count += my_array[idx] 
        idx +=1
    return idx