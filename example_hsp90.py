import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import mult_grad_population_calibration.optimize_weights as opt
import mult_grad_population_calibration.utils as utils

def main():

    # Set up RNG keys
    seed_train_test = 1

    # Set up directories (not needed here but can be used if wanting saved figs)
    main_dir = "."
    fig_dir = f"{main_dir}/figures/hsp90"
    data_dir = f"{main_dir}/data/"

    # Set up pretty plots 
    plt.style.use("my_style.mplstyle") # Use stylefile defined
    plt.style.use("seaborn-v0_8-colorblind") # Use colorscheme from colorblind seaborn

    # Load likelihood matrix
    log_likelihood = jnp.load(f"{data_dir}/likelihoods/hsp90/log_likelihood_cryojax.npy")
    num_data, num_nodes = jnp.shape(log_likelihood)


    nodes = jnp.arange(0, num_nodes, 1)

    true_weights = jnp.load(f"{data_dir}/hsp90_true_weights.npy")

    # Compute weights:
    # "weights_frequency" set to save every 1 iterations, for retrieving later
    # train_test set to TRUE so that two sets of stopping-criteria are used:
    #  - stopping when the gradient gap is at tol (weights_gap)
    #  - stopping based on a train_test split (weights_train_test)
    key = jax.random.key(seed_train_test)
    weights, info = opt.multiplicative_gradient(log_likelihood, 
                                                max_iterations=1000, 
                                                weights_frequency=1,
                                                tol=1e-2, 
                                                verbose=True, 
                                                train_test_key=key, 
                                                train_test=True)
    
    # Plot 
    # if wanting to see trends easier, set plot_initial=False, 
    # it drops first iterate (initial weights) from plotting x-axis
    plot_initial=True
    utils.plot_weights_and_info_1d(nodes, info, true_weights=true_weights, plot_initial=plot_initial)
    
    # for saving figures, use:
    #utils.plot_weights_and_info_1d(nodes, info, true_weights=true_weights, plot_initial=plot_initial, fig_dir=fig_dir)

    # for comparing weights at max_iterations, to early stopped weights:
    #  - set diagnostic=True in multiplicative_gradient()
    #  - set final_weights=weights in plot_weights_and_info_1d()
    # example:
        # #this code will save early stopped weights, but run to max_iterations=1000, because diagnostic=True
        # weights, info = opt.multiplicative_gradient(log_likelihood, 
        #                                            max_iterations=1000, 
        #                                            weights_frequency=1,
        #                                            tol = 1e-2, 
        #                                            verbose=True, 
        #                                            train_test_key=key, 
        #                                            train_test=True,
        #                                            diagnostic=True)
        # #this code will plot `weights` returned above as the max iteration weights in the plots: 
        # utils.plot_weights_and_info_1d(nodes, info, true_weights=true_weights, final_weights=weights, plot_initial=plot_initial)
    plt.show()


if __name__ == "__main__":
    main()