import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import mult_grad_population_calibration.optimize_weights as opt

def main():

    # Set up directories
    main_dir = "."
    fig_dir = f"{main_dir}/figures/1d_mixture"
    data_dir = f"{main_dir}/data/likelihoods"


    # Set up gaussian mixture model
    seed = 0
    weights = [0.3, 0.7]
    means = [-1.0, 1.0]
    stds = [0.5, 0.5]
    key = jax.random.key(seed)
    
    # Set up dataset parameters 
    num_samples = 100000
    num_nodes = 100
    noise_std_dev = 0.5

    # Simulate clean data 
    clean_data = sample_gaussian_mixture_1d(key, weights, means, stds, num_samples) 

    print(key)
    key, subkey = jax.random.split(key)
    
    # NOTE: very strange behavior depending on whether key or subkey is used. What is going on?
    # See these plots of the noisy data, one is always different than the other 
    #noise1 = clean_data + noise_std_dev*jax.random.normal(subkey, shape=clean_data.shape)
    #noise2 = clean_data + noise_std_dev*jax.random.normal(key, shape=clean_data.shape)
    #print(noise1.shape)
    #print(noise2.shape) 
    #plt.figure()
    #plt.hist(noise1, bins=100)
    #plt.figure() 
    #plt.hist(noise2, bins=100)
    #plt.show()
    #return

    # Add noise
    # NOTE: consistently across seeds, subkey always gets stranger behavior than key???
    key, subkey = jax.random.split(key)
    data = clean_data + noise_std_dev*jax.random.normal(key, shape=clean_data.shape)
    
    # Choosing nodes: for now, just picking evenly spaced ones
    nodes = jnp.linspace(-4, 4, num_nodes)
    num_nodes = nodes.shape[0]

    # Evaluating prob to set up `true weights` at nodes 
    eval_log_prob = jax.vmap(lambda x : eval_mixture_list(x, weights, means, stds))
    true_weights = jnp.exp(eval_log_prob(nodes))
    true_weights /= true_weights.sum() 

    plt.figure()
    plt.plot(nodes, true_weights/(nodes[1] - nodes[0]))
    plt.hist(clean_data, bins=nodes,density=True)
    plt.figure()
    plt.hist(data)
    
    # Compute log likelihood matrix
    log_likelihood = -1*(data[:, None] - nodes[None, :])**2 / (2*noise_std_dev**2)

    # Compute weights
    ensemble_weights, info = opt.multiplicative_gradient(log_likelihood, max_iterations = 10000, weights_frequency=1, VERBOSE=True)
    losses = info["losses"]
    gaps = info["gaps"]
    weights_gap = info["weights_gap"]
    weights_cross_val = info["weights_cross_val"]
    weights = info["weights"]
    gap_idx = info["gap_idx"]
    cross_val_idx = info["cross_val_idx"] 
   
    plt.figure()
    plt.plot(nodes, true_weights)
    plt.plot(nodes, weights_gap, label="weights_gap")
    plt.plot(nodes, weights_cross_val, label="weights_cross_val")

    iterations = jnp.arange(losses.shape[0])
    plt.figure()
    plt.semilogx(iterations+1, losses)
    plt.figure()
    plt.loglog(iterations+1, gaps)
    plt.show()
    
@jax.jit
def eval_mixture_list(node, weights,means, stds):
    output = jnp.zeros(len(means))
    weights = jnp.array(weights)
    for idx in range(len(means)):
        val = -1*(node - means[idx])**2 / (2*(stds[idx])**2) - jnp.log(2*jnp.pi*stds[idx])
        output = output.at[idx].set(val)
    return jax.scipy.special.logsumexp(a=output, b=weights)


def sample_gaussian_mixture_1d(key, weights, means, stds, num_samples):
    """ 1d gaussian mixture model sampling, for simple tests

    Parameters
    ----------
    key: jax.PRNGkey
    weights : list
    means : list
    stds : list
    num_samples : int
    """

    # Convert to array
    weights = jnp.array(weights)
    means = jnp.array(means)
    stds = jnp.array(stds)

    # Sample component labels
    labels = jax.random.choice(key, len(weights), (num_samples,), p=weights)
    # Sample from gaussians at the component labels
    key, subkey = jax.random.split(key)
    means_at_labels = means[labels] 
    stds_at_labels = stds[labels]
    standard_normals = jax.random.normal(subkey, shape=(num_samples,))
    samples = standard_normals*stds_at_labels + means_at_labels  
    return samples


if __name__ == "__main__":
    main()