import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib as mpl

import mult_grad_population_calibration.optimize_weights as opt

def main():

    # Set up directories
    main_dir = "."
    fig_dir = f"{main_dir}/figures/1d_mixture"
    data_dir = f"{main_dir}/data/likelihoods"

    # Set up pretty plots 
    plt.style.use(f"my_style.mplstyle") # Use stylefile defined
    plt.style.use("seaborn-v0_8-colorblind") # Use colorscheme from colorblind seaborn
    mpl.rcParams['text.usetex'] = True  # Uncomment for latex font in plots
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color'] # Save color list for reference

    # Set up gaussian mixture model
    seed = 1234
    weights = [0.3, 0.7]
    means = [-1.0, 1.0]
    stds = [0.5, 0.5]
    key = jax.random.key(seed)

    # Set up dataset parameters 
    num_samples = 100000
    num_nodes = 100
    noise_std_dev = 0.5

    # Simulate clean data 
    key, subkey = jax.random.split(key)
    clean_data = sample_gaussian_mixture_1d(key, subkey, weights, means, stds, num_samples) 

    # Add noise
    key, subkey = jax.random.split(key)
    data = clean_data + noise_std_dev*jax.random.normal(subkey, shape=clean_data.shape)
    
    # Choosing nodes: for now, just picking evenly spaced ones
    nodes = jnp.linspace(-4, 4, num_nodes)
    num_nodes = nodes.shape[0]

    # Evaluating prob to set up `true weights` at nodes 
    eval_log_prob = jax.vmap(lambda x : eval_mixture_list(x, weights, means, stds))
    true_weights = jnp.exp(eval_log_prob(nodes))
    true_weights /= true_weights.sum() 

    # Plot noisy and clean data (1d problem, we can do this)
    plot_histogram_data(nodes, clean_data, data, true_weights)

    # Compute log likelihood matrix
    log_likelihood = -1*(data[:, None] - nodes[None, :])**2 / (2*noise_std_dev**2)

    # Compute weights
    key, subkey = jax.random.split(key)
    weights, info = opt.multiplicative_gradient(log_likelihood, 
                                                max_iterations=10000, 
                                                weights_frequency=1, 
                                                VERBOSE=True, 
                                                cross_val_key=key, 
                                                CROSS_VALIDATE=True)
    
    plot_weights_and_info(nodes, info, true_weights)
    plt.show()

def plot_histogram_data(nodes, clean_data, data, true_weights):
    plt.figure()
    plt.plot(nodes, true_weights/(nodes[1] - nodes[0]), label="true density", color="k")
    plt.hist(clean_data, bins=nodes,density=True, label="hist, clean data", color="C0")
    plt.ylabel("Probability")
    plt.xlabel("x")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(nodes, true_weights/(nodes[1] - nodes[0]), label="true density", color="k")
    plt.hist(data, bins=nodes, density=True, color="C1", label="hist, noisy data")
    plt.ylabel("Probability")
    plt.xlabel("x")
    plt.legend()
    plt.tight_layout()


def plot_weights_and_info(nodes, info, true_weights):

    # Read in info from optimization 
    losses = info["losses"]
    gaps = info["gaps"]
    weights_gap = info["weights_gap"]
    weights_cross_val = info["weights_cross_val"]
    weights = info["weights"]
    gap_idx = info["gap_idx"]
    cross_val_idx = info["cross_val_idx"] 


    iterations = jnp.arange(0, len(losses), 1)

    # Plot final weights
    plt.figure()
    plt.plot(nodes, true_weights, label='true', color="C0", marker='.')
    plt.plot(nodes, weights_gap, label='weights, gap', color="C1", marker='.')
    plt.plot(nodes, weights_cross_val, label='weights, cross-val', color="C2", marker='.')
    plt.xlabel('x')
    plt.ylabel('Probability') 
    plt.legend(loc="upper right", fontsize=16)
    #plt.savefig(f"{fig_dir}/weight_comparison.png", dpi=300)

    # Plot losses
    plt.figure()
    # Here plotting a semilog plot, and shifting indices so 0 doesn't show up
    plt.semilogx(iterations+1, losses, label='losses', c='k')
    plt.vlines(gap_idx, ymin=jnp.min(losses), ymax=jnp.max(losses), colors='C1', linestyles="-.", label="gap idx")
    plt.vlines(cross_val_idx, ymin=jnp.min(losses), ymax=jnp.max(losses), colors='C2', linestyles="-.", label="cross-val idx")
    plt.xlabel('iterations')
    plt.ylabel('Loss') 
    plt.legend(fontsize=16)
    #plt.savefig(f"{fig_dir}/weight_comparison.png", dpi=300)

    # Plot gaps
    plt.figure()
    # Here plotting a semilog plot, and shifting indices so 0 doesn't show up
    plt.semilogx(iterations+1, gaps, label='gaps', c='k')
    plt.vlines(gap_idx, ymin=jnp.min(gaps), ymax=jnp.max(gaps), colors='C1', linestyles="-.", label="gap idx")
    plt.vlines(cross_val_idx, ymin=jnp.min(gaps), ymax=jnp.max(gaps), colors='C2', linestyles="-.", label="cross-val idx")
    plt.xlabel('iterations')
    plt.ylabel('Gap') 
    plt.legend(fontsize=16)
    #plt.savefig(f"{fig_dir}/weight_comparison.png", dpi=300)


@jax.jit
def eval_mixture_list(node, weights,means, stds):
    output = jnp.zeros(len(means))
    weights = jnp.array(weights)
    for idx in range(len(means)):
        val = -1*(node - means[idx])**2 / (2*(stds[idx])**2) - jnp.log(2*jnp.pi*stds[idx])
        output = output.at[idx].set(val)
    return jax.scipy.special.logsumexp(a=output, b=weights)


def sample_gaussian_mixture_1d(key_labels, key_noise, weights, means, stds, num_samples):
    """ 1d gaussian mixture model sampling, for simple tests

    Parameters
    ----------
    key_labels: jax.PRNGkey
    key_noise: jax.PRNGkey
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
    labels = jax.random.choice(key_labels, len(weights), (num_samples,), p=weights)
    # Sample from gaussians at the component labels
    means_at_labels = means[labels] 
    stds_at_labels = stds[labels]
    standard_normals = jax.random.normal(key_noise, shape=(num_samples,))
    samples = standard_normals*stds_at_labels + means_at_labels  
    return samples


if __name__ == "__main__":
    main()