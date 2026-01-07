import jax.numpy as jnp
import jax

def cross_val_split(key, log_likelihood, train_pct=0.8):
    num_images = log_likelihood.shape[0]
    split_size = int(jnp.ceil(train_pct*log_likelihood.shape[0]))
    train_idx = jax.random.choice(key, num_images, (split_size,), replace=False) 
    test_idx = jnp.setdiff1d(jnp.arange(num_images), train_idx)
    train = jnp.copy(log_likelihood[train_idx, :])
    test = jnp.copy(log_likelihood[test_idx, :])
    return train, test, train_idx, test_idx

