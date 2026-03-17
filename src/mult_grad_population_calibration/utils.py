import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

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


def plot_weights_and_info_1d(nodes, info, true_weights=None, final_weights=None, plot_initial=True, fig_dir=None):
    """Basic plot for problems where the weights can be indexed in one dimension.
    Practically: this routine includes a plt.plot(weights) call. By `indexed in 1 dimension', 
    this means that for your problem, a plt.plot(weights) should be meaningful compared to plt.bar(weights).
    
    Parameters
    ----------
    nodes : jax.Array
        indices corresponding to the weights
    info : 
        _description_
    true_weights : jax.Array, optional
        true weights for synthetic problems with a ground truth, by default None
    final_weights : jax.Array, optional
        if not None, will plot these weights as `max_iterations' weights for comparing with early stopped weights, by default None
    plot_initial : bool, optional
        if False, drops 0 index from plots for easier visualization of trends, by default True
    fig_dir : _type_, optional
        if not None, figures will be saved to fig_dir, by default None
    """
    # Read in info from optimization 
    losses = info["losses"]
    gaps = info["gaps"]
    weights_gap = info["weights_gap"]
    weights_train_test = info["weights_train_test"]
    weights = info["weights"]
    gap_idx = info["gap_idx"]
    train_test_idx = info["train_test_idx"] 


    if plot_initial:
        # keep all indices for plotting
        iterations = jnp.arange(0, len(losses), 1) + 1
        gaps_plot = gaps
        losses_plot = losses
        print("NOTE: Plotting with initial loss and gap at iterations=0, may need to plot later iterates to see trends")
        print("to do this, set plot_initial=False")
        print("iterations are shifted to start at iterations=1 for log-plot on x-axis")

    else:
        # drop 0 index, shift all plotted quantities by 1 index
        iterations = jnp.arange(1, len(losses), 1)
        gaps_plot = gaps[1:]
        losses_plot = losses[1:]
        gap_idx += 1
        train_test_idx += 1
        print("NOTE: Plotting without initial loss or gap, to show trends easier")

    # Plot final weights
    plt.figure()
    if true_weights is not None:
        plt.plot(nodes, true_weights, label='true', color="C0", marker='.')
    plt.plot(nodes, weights_gap, label='weights, gap', color="C1", marker='.')
    plt.plot(nodes, weights_train_test, label='weights, train-test', color="C2", marker='.')
    if final_weights is not None:
        plt.plot(nodes, final_weights, label='weights, max iters.', color="C3", marker='.')
    plt.xlabel('x')
    plt.ylabel('Probability') 
    plt.legend(loc="upper right", fontsize=16)
    plt.tight_layout()
    if fig_dir is not None:
        plt.savefig(f"{fig_dir}/weight_comparison.png", dpi=300)

    # Plot losses
    plt.figure()
    plt.semilogx(iterations, losses_plot, label='losses', c='k')
    plt.vlines(gap_idx, ymin=jnp.min(losses_plot), ymax=jnp.max(losses_plot), colors='C1', linestyles="-.", label="gap idx")
    plt.vlines(train_test_idx, ymin=jnp.min(losses_plot), ymax=jnp.max(losses_plot), colors='C2', linestyles="-.", label="cross-val idx")
    plt.xlabel('iterations')
    plt.ylabel('Loss') 
    plt.legend(fontsize=16)
    plt.tight_layout()
    if fig_dir is not None:
        plt.savefig(f"{fig_dir}/losses.png", dpi=300)

    # Plot gaps
    plt.figure()
    # Here plotting a semilog plot, and shifting indices so 0 doesn't show up
    plt.semilogx(iterations, gaps_plot, label='gaps', c='k')
    plt.vlines(gap_idx, ymin=jnp.min(gaps_plot), ymax=jnp.max(gaps_plot), colors='C1', linestyles="-.", label="gap idx")
    plt.vlines(train_test_idx, ymin=jnp.min(gaps_plot), ymax=jnp.max(gaps_plot), colors='C2', linestyles="-.", label="train-test idx")
    plt.xlabel('iterations')
    plt.ylabel('Gap') 
    plt.legend(fontsize=16)
    plt.tight_layout()
    if fig_dir is not None:
        plt.savefig(f"{fig_dir}/gaps.png", dpi=300)

