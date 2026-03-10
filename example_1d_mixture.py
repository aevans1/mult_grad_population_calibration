import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import mult_grad_population_calibration.optimize_weights as opt

def main():

    # Set up RNG keys: first for generating the example, second for cross-validation split
    seed_mixture = 1
    seed_train_test = 2

    # Set up directories (not needed here but can be used if wanting saved figs)
    main_dir = "."
    fig_dir = f"{main_dir}/figures/1d_mixture"
    data_dir = f"{main_dir}/data/likelihoods"

    # Set up pretty plots 
    plt.style.use("my_style.mplstyle") # Use stylefile defined
    plt.style.use("seaborn-v0_8-colorblind") # Use colorscheme from colorblind seaborn

    # Set up gaussian mixture model
    weights = [0.3, 0.7]
    means = [-1.0, 1.0]
    stds = [0.5, 0.5]
    key = jax.random.key(seed_mixture)

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

    # Compute weights:
    # "weights_frequency" set to save every 1 iterations, for retrieving later
    # TRAIN_TEST set to TRUE so that two sets of stopping-criteria are used:
    #  - stopping when the gradient gap is at tol (weights_gap)
    #  - stopping based on a train_test split (weights_train_test)

    key = jax.random.key(seed_train_test)
    weights, info = opt.multiplicative_gradient(log_likelihood, 
                                                max_iterations=10000, 
                                                weights_frequency=1,
                                                tol = 1e-2, 
                                                verbose=True, 
                                                train_test_key=key, 
                                                train_test=True)
    
    # Plot 
    plot_initial=True
    plot_weights_and_info(nodes, info, true_weights, plot_initial=plot_initial)

    # if wanting to see trends easier, set plot_initial=False, 
    # it drops first iterate (initial weights) from plotting x-axis

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


def plot_weights_and_info(nodes, info, true_weights, plot_initial=True):

    # Read in info from optimization 
    losses = info["losses"]
    gaps = info["gaps"]
    weights_gap = info["weights_gap"]
    weights_train_test = info["weights_train_test"]
    weights = info["weights"]
    gap_idx = info["gap_idx"]
    train_test_idx = info["train_test_idx"] 


    
    # Not including stats at initial weights, skews the plots.
    plot_initial = False
    if plot_initial:
        iterations = jnp.arange(0, len(losses), 1) + 1
        gaps_plot = gaps
        losses_plot = losses
        print("NOTE: Plotting with initial loss and gap at iterations=0, may need to plot later iterates to see trends")
        print("to do this, set plot_initial=False")
        print("iterations are shifted to start at iterations=1 for log-plot on x-axis")

    else:
        iterations = jnp.arange(1, len(losses), 1)
        gaps_plot = gaps[1:]
        losses_plot = losses[1:]
        print("NOTE: Plotting without initial loss or gap, to show trends easier")

    # Plot final weights
    plt.figure()
    plt.plot(nodes, true_weights, label='true', color="C0", marker='.')
    plt.plot(nodes, weights_gap, label='weights, gap', color="C1", marker='.')
    plt.plot(nodes, weights_train_test, label='weights, train-test', color="C2", marker='.')
    plt.xlabel('x')
    plt.ylabel('Probability') 
    plt.legend(loc="upper right", fontsize=16)
    #plt.savefig(f"{fig_dir}/weight_comparison.png", dpi=300)

    # Plot losses
    plt.figure()
    # Here plotting a semilog plot, and shifting indices so 0 doesn't show up
    plt.semilogx(iterations, losses_plot, label='losses', c='k')
    plt.vlines(gap_idx, ymin=jnp.min(losses_plot), ymax=jnp.max(losses_plot), colors='C1', linestyles="-.", label="gap idx")
    plt.vlines(train_test_idx, ymin=jnp.min(losses_plot), ymax=jnp.max(losses_plot), colors='C2', linestyles="-.", label="cross-val idx")
    plt.xlabel('iterations')
    plt.ylabel('Loss') 
    plt.legend(fontsize=16)
    #plt.savefig(f"{fig_dir}/weight_comparison.png", dpi=300)

    # Plot gaps
    plt.figure()
    # Here plotting a semilog plot, and shifting indices so 0 doesn't show up
    plt.semilogx(iterations, gaps_plot, label='gaps', c='k')
    plt.vlines(gap_idx, ymin=jnp.min(gaps_plot), ymax=jnp.max(gaps_plot), colors='C1', linestyles="-.", label="gap idx")
    plt.vlines(train_test_idx, ymin=jnp.min(gaps_plot), ymax=jnp.max(gaps_plot), colors='C2', linestyles="-.", label="train-test idx")
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