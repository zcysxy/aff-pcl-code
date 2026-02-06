# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import multivariate_normal
import pickle
import datetime
import os
np.random.seed(42)

# %%
# Configuration
class Config:
    """Stores all parameters for the numerical experiment."""
    backup_dir = "bkup"
    runs = 10
    n = 20
    d = 5
    t = 60
    alpha = 0.01
    base_scale_a = 4
    delta_a = 0.1
    delta_b = 0.1
    noise_b = 0.5
    noise_a = 1
    noise_a_list = [0.0, 0.5, 1.0, 2.0]
    # Basic
    # heterogeneity_settings = {
    #     'homogeneous': (0.0, 0.0),
    #     'low': (0.05, 0.05),
    #     'medium': (0.2, 0.2),
    #     'high': (0.5, 0.5),
    # }
    # Exhaustive
    # heterogeneity_settings = {}
    # for kernel_het in np.linspace(0.0, 0.9, 10):
    #     for reward_het in np.linspace(0.0, 0.9, 10):
    #         key = f'({kernel_het},{reward_het})'
    #         heterogeneity_settings[key] = (kernel_het, reward_het)
    # Pareto
    n_list = np.array([2,3,4,5,6,7,8,10,15,20,35,50])
    delta_list = 1 / n_list

# %%
# Data Generation for Heterogeneous Systems
def generate_synthetic_data(config: Config):
    """
    Generates synthetic data for a multi-agent heterogeneous linear system
    based on the paper's setup.
    """
    print("Generating synthetic data...")

    # Create heterogeneous environment distributions (mu^i)
    # Multivariate distribution (feature space)
    # Each agent's data distribution is a Gaussian with a different mean.
    means = []
    for i in range(config.n):
        rand_vec = np.random.randn(config.d)
        normed_vec = rand_vec / np.linalg.norm(rand_vec)
        # Center the first agent
        if i == 0:
            normed_vec = np.zeros(config.d)
        mean = config.delta_a * config.base_scale_a * normed_vec
        means.append(mean)
    cov = np.eye(config.d)
    distributions = [multivariate_normal(mean=m, cov=cov) for m in means]

    # Define shared feature embedding A(s) and Phi(s)
    # We use multiplicative noise
    # A_bar_base is a shared, underlying positive definite matrix.
    _temp_A = np.random.rand(config.d, config.d)
    A_bar_base = _temp_A.T @ _temp_A + config.d * np.eye(config.d) # nice conditioning

    _temp_Phi = np.random.randn(config.d, config.d)
    Phi_bar_base = _temp_Phi.T @ _temp_Phi + config.d * np.eye(config.d) # nice conditioning


    def A_func(s):
        # The stochastic feature matrix A(s) depends on the sample s.
        # This ensures that E_{s~mu_i}[A(s)] is different for each agent.
        return (config.noise_a * np.outer(s, s) + np.eye(config.d)) @ A_bar_base

    def Phi_func(s):
        # The stochastic feature matrix Phi(s) depends on the sample s.
        # This ensures that E_{s~mu_i}[Phi(s)] is different for each agent.
        return (config.noise_b * np.outer(s, s) + np.eye(config.d)) @ Phi_bar_base

    def b_func(s, theta_star):
        # The stochastic label b^i(s) follows a linear structure.
        # return Phi @ theta_star
        return Phi_func(s) @ theta_star

    # Create heterogeneous true reward parameters (theta_star_i)
    theta_star_base = np.random.randn(config.d)
    theta_star_base = theta_star_base / np.linalg.norm(theta_star_base)
    thetas_star = []
    for i in range(config.n):
        rand_vec = np.random.randn(config.d)
        norm = np.linalg.norm(rand_vec)
        if norm == 0:
            norm = 1  # avoid division by zero
        rand_vec_normalized = rand_vec / norm
        # Center the first agent
        if i == 0:
            rand_vec_normalized = np.zeros(config.d)
        theta_star = theta_star_base + config.delta_b * rand_vec_normalized
        thetas_star.append(theta_star)

    # Calculate ground truth solutions x_star_i via Monte Carlo
    # The true solution x_star_i = inv(A_bar_i) @ b_bar_i, where the bars
    # denote expectation over mu_i. We approximate this with sampling.
    print("Calculating ground truth solutions via Monte Carlo...")
    n_samples_mc = 5000
    x_stars = []
    for i in range(config.n):
        samples = distributions[i].rvs(size=n_samples_mc)
        A_bar_i = np.mean([A_func(s) for s in samples], axis=0)
        b_bar_i = np.mean([b_func(s,thetas_star[i]) for s in samples], axis=0)
        x_star_i = np.linalg.solve(A_bar_i, b_bar_i)
        x_stars.append(x_star_i)

    # Define density ratio function rho^i(s)
    # rho^i(s) = mu^i(s) / mu^0(s), where mu^0 = (1/n) * sum(mu^j)
    def rho_func(s, i):
        mu_i_pdf = distributions[i].pdf(s)
        mu_0_pdf = np.mean([dist.pdf(s) for dist in distributions])
        return mu_i_pdf / (mu_0_pdf + 1e-9) # Add epsilon for stability

    data = {
        'distributions': distributions,
        'A_func': A_func,
        'b_func': b_func,
        'Phi_func': Phi_func,
        'thetas_star': thetas_star,
        'x_stars': x_stars,
        'rho_func': rho_func,
    }
    print("Data generation complete.")
    return data

# %%
## Algorithms
def run_independent_learning(data: dict, config: Config):
    """Baseline 1: Each agent learns entirely on its own."""
    print("Running IL...")
    x = [np.zeros(config.d) for _ in range(config.n)]
    errors = np.zeros((config.t, config.n))

    for t in range(config.t):
        for i in range(config.n):
            s_t_i = data['distributions'][i].rvs()
            # Local gradient: g_t^i(x_t^i) = A(s_t^i)x_t^i - b^i(s_t^i)
            g_t_i = data['A_func'](s_t_i) @ x[i] - data['b_func'](s_t_i, data['thetas_star'][i])
            x[i] -= config.alpha * g_t_i
            errors[t, i] = np.linalg.norm(x[i] - data['x_stars'][i])**2
            
    return np.mean(errors, axis=1)

# %%
def run_federated_averaging(data: dict, config: Config):
    """Baseline 2: All agents learn a single, unified model."""
    print("Running FL...")
    x_0 = np.zeros(config.d)  # Single central model
    errors = np.zeros((config.t, config.n))

    for t in range(config.t):
        grad_agg = np.zeros(config.d)
        for i in range(config.n):
            s_t_i = data['distributions'][i].rvs()
            # Each agent computes a gradient at the central model x_0
            g_t_i = data['A_func'](s_t_i) @ x_0 - data['b_func'](s_t_i, data['thetas_star'][i])
            grad_agg += g_t_i
        
        x_0 -= config.alpha * (grad_agg / config.n)
        
        #NOTE: Measure error of the single model against each agent's personal optimum
        for i in range(config.n):
            errors[t, i] = np.linalg.norm(x_0 - data['x_stars'][i])**2
            
    return np.mean(errors, axis=1)

# %%
def run_personalized_collaborative(data: dict, config: Config):
    """Proposed Method: Personalized Collaborative Learning."""
    print("Running PCL...")
    # Personalized models for each agent
    x = [np.zeros(config.d) for _ in range(config.n)]
    # Central variables maintained on the server
    x_c = np.zeros(config.d)
    theta_c = np.zeros(config.d) # For learning the central reward
    errors = np.zeros((config.t, config.n))

    for t in range(config.t):
        # In each step, every agent draws a fresh sample
        samples = [dist.rvs() for dist in data['distributions']]

        # Central learning: Update central reward (theta_c) and central decision variable (x_c)
        grad_agg_b = np.zeros(config.d)
        grad_agg_c = np.zeros(config.d)
        
        for j in range(config.n):
            s_t_j = samples[j]
            # Gradient for central reward learning
            grad_agg_b += data['Phi_func'](s_t_j) @ theta_c - data['b_func'](s_t_j, data['thetas_star'][j])
            # Gradient for central model learning
            # NOTE: use learned reward
            # grad_agg_c += data['A_func'](s_t_j) @ x_c - data['b_func'](s_t_j, theta_c)
            grad_agg_c += data['A_func'](s_t_j) @ x_c - data['b_func'](s_t_j, data['thetas_star'][j]) 
        
        theta_c_temp = theta_c.copy()
        b_hat_c_t = lambda s: data['Phi_func'](s) @ theta_c_temp # Learned central reward at step t
        theta_c -= config.alpha * (grad_agg_b / config.n)

        x_c_temp = x_c.copy()
        x_c -= config.alpha * (grad_agg_c / config.n)

        # Local learning: Update personalized models x_i
        for i in range(config.n):
            s_t_i = samples[i]
            
            # 1. Local gradient: g_t^i(x_t^i)
            g_t_i = data['A_func'](s_t_i) @ x[i] - data['b_func'](s_t_i, data['thetas_star'][i])
            
            # 2. Importance-corrected central gradient: (rho^i circ g_t^0)(x_t^c)
            g_rho_corr = np.zeros(config.d)
            for j in range(config.n):
                s_t_j = samples[j]
                g_c_arrow_j = data['A_func'](s_t_j) @ x_c_temp - b_hat_c_t(s_t_j)
                rho_i_j = data['rho_func'](s_t_j, i)
                g_rho_corr += rho_i_j * g_c_arrow_j
            g_rho_corr /= config.n
            
            # 3. Bias correction term: g_t^{c->i}(x_t^c)
            g_bias_corr = data['A_func'](s_t_i) @ x_c_temp - b_hat_c_t(s_t_i)
            
            # Full personalized update direction (Eq. 6 from the paper)
            g_tilde_i = g_t_i + g_rho_corr - g_bias_corr
            
            x[i] -= config.alpha * g_tilde_i
            errors[t, i] = np.linalg.norm(x[i] - data['x_stars'][i])**2

    return np.mean(errors, axis=1)
    # return errors

# %%
# Wrapper for experiments with varying noise_a and fixed heterogeneity
def run_experiments_with_noise(config):
    runs = config.runs
    n_iter = config.t
    methods = {
        'ind': run_independent_learning,
        'fedavg': run_federated_averaging,
        'pcl': run_personalized_collaborative,
        'pcl_i': run_personalized_collaborative,
    }
    noise_a_list = config.noise_a_list

    # Results: noise -> method -> (mean, std)
    results = {}
    for noise_a in noise_a_list:
        print(f"\n=== Running for noise_a={noise_a}")
        config.noise_a = noise_a
        # Regularize learning rate by exp(-noise/2)
        # config.alpha = 0.01 * np.exp(-noise_a)
        errors = {method: np.zeros((runs, n_iter)) for method in methods}

        def run_all_methods(data, config, run_idx):
            for method_key, method_func in methods.items():
                if method_key == 'pcl':
                    _temp_pcl = method_func(data, config)
                    errors[method_key][run_idx] = np.mean(_temp_pcl, axis=1)
                elif method_key == 'pcl_i':
                    errors[method_key][run_idx] = _temp_pcl[:,0]
                else:
                    errors[method_key][run_idx] = method_func(data, config)

        for run in range(runs):
            print(f"Run {run+1}/{runs}")
            data = generate_synthetic_data(config)
            run_all_methods(data, config, run)

        # Compute mean and std
        results[noise_a] = {
            method: (
                errors[method].mean(axis=0),
                errors[method].std(axis=0)
            )
            for method in methods
        }
    return results

# %%
# Wrapper for experiments with multiple repeats and heterogeneity settings
def run_experiments_with_repeats(config):
    runs = config.runs
    n_iter = config.t
    heterogeneity_settings = config.heterogeneity_settings
    
    methods = {
        'ind': run_independent_learning,
        'fedavg': run_federated_averaging,
        'pcl': run_personalized_collaborative,
        'pcl_i': run_personalized_collaborative,
    }

    # Initialize error arrays
    errors = {
        het: {method: np.zeros((runs, n_iter)) for method in methods}
        for het in heterogeneity_settings
    }

    def run_all_methods(data, config, run_idx, het_key):
        for method_key, method_func in methods.items():
            if method_key == 'pcl':
                _temp_pcl = method_func(data, config)
                errors[het_key][method_key][run_idx] = np.mean(_temp_pcl, axis=1)
            elif method_key == 'pcl_i':
                errors[het_key][method_key][run_idx] = _temp_pcl[:,0]
            else:
                errors[het_key][method_key][run_idx] = method_func(data, config)

    for run in range(runs):
        for het_key, (kernel_het, reward_het) in heterogeneity_settings.items():
            print(f"Run {run+1}/{runs} - {het_key.capitalize()} Heterogeneity")
            config.delta_a = kernel_het
            config.delta_b = reward_het
            data = generate_synthetic_data(config)
            run_all_methods(data, config, run, het_key)

    # Compute mean and std
    results = {
        het: {
            method: (
                errors[het][method].mean(axis=0),
                errors[het][method].std(axis=0)
            )
            for method in methods
        }
        for het in heterogeneity_settings
    }
    return results

# %%

def run_pareto_experiments(config: Config):
    """
    Runs experiments by varying the number of agents (n) and the
    heterogeneity level (delta) to generate data for the Pareto-like plot.
    """
    n_list = config.n_list
    delta_list = config.delta_list
    # Store the final MSE for each setting
    results_mse = np.zeros((len(n_list), len(delta_list)))

    for i, n_val in enumerate(n_list):
        for j, delta_val in enumerate(delta_list):
            print(f"\n=== Running for n={n_val}, delta={delta_val:.2f}")
            # Update config for the current run
            config.n = n_val
            config.delta_a = delta_val
            config.delta_b = delta_val

            run_errors = []
            for run in range(config.runs):
                print(f"  Run {run+1}/{config.runs}")
                data = generate_synthetic_data(config)
                # We only need the PCL algorithm for this experiment
                pcl_errors_over_time = run_personalized_collaborative(data, config)
                # Get the mean error over all agents over last 10 iterations
                final_mse = np.mean(pcl_errors_over_time[-10:])
                run_errors.append(final_mse)

            # Average the final MSE over all runs
            results_mse[i, j] = np.mean(run_errors)
            
    return results_mse

# %%

# Main Execution and Variance Plotting
config = Config()

# Run
# results = run_experiments_with_noise(config)
# results = run_experiments_with_repeats(config)
# results = run_pareto_experiments(config)

# Load
backup_files = [f for f in os.listdir(config.backup_dir) if f.endswith("pareto.pkl")]
latest_file = max(backup_files, key=lambda x: x.split(".")[0])
with open(os.path.join(config.backup_dir, latest_file), "rb") as f:
    results = pickle.load(f)

# Save
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# with open(f"bkup/{timestamp}.pkl", "wb") as f:
#     pickle.dump(results, f)


# %%

def plot_results_on_axis(ax, results_dict, config, title):
    x = np.arange(config.t)
    for key, label, marker, color in [
        ('ind', 'Independent', 'o', 'C0'),
        ('fedavg', 'Federated', '^', 'C1'),  # triangle marker
        ('pcl', 'Personalized', 'D', 'C2'),
        ('pcl_i', 'Agent-Specific', 's', 'C3'),  # Changed marker to square ('s') for matplotlib
        ]:
        mean, std = results_dict[key]
        ax.plot(x, mean, label=label, marker=marker, color=color, markevery=10, markersize=7, markerfacecolor='none')
        ax.fill_between(x, mean-1.64*std/np.sqrt(config.runs), mean+1.64*std/np.sqrt(config.runs), color=color, alpha=0.2)
    ax.set_title(title, fontsize=14)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_aspect(1./ax.get_data_ratio())
    # ax.grid(True, which="both", ls="--", alpha=0.6)  # grid removed

# fig, axs = plt.subplots(1, 4, figsize=(12, 4))
# number of rows is number of results divided by 4, rounded up
# fig, axs = plt.subplots(len(results) // 4 + (len(results) % 4 > 0), 4, figsize=(12, 4 * (len(results) // 4 + (len(results) % 4 > 0))), squeeze=False)

# plot_results_on_axis(axs[0], results['homogeneous'], config, 'Homogeneous')
# plot_results_on_axis(axs[1], results['low'], config, 'Low Heterogeneity')
# plot_results_on_axis(axs[2], results['medium'], config, 'Medium Heterogeneity')
# plot_results_on_axis(axs[3], results['high'], config, 'High Heterogeneity')
# results_dict = {'low': 'Low Heterogeneity', 'medium': 'Medium Heterogeneity', 'high': 'High Heterogeneity'}
# Noise levels
results_dict = {}
# results_dict = {0.0: 'No Noise', 0.5: 'Low Noise', 1.0: 'Medium Noise', 5.0: 'High Noise'}
# for noise in config.noise_a_list:
#     results_dict[noise] = f'Noise std: {noise}'
# for het_key in results.keys():
#     kernel_het, reward_het = config.heterogeneity_settings[het_key]
#     results_dict[het_key] = f'({round(kernel_het, 1)}, {round(reward_het, 1)})'
#
# for i, (key, label) in enumerate(results_dict.items()):
#     plot_results_on_axis(axs[i//4, i%4], results[key], config, label)
#
# # fig.suptitle('Comparison of Learning Algorithms under Different Heterogeneity Levels', fontsize=18)
# fig.supxlabel('# Samples', fontsize=14, y=0.12)
# fig.supylabel('Mean Squared Error', fontsize=14)
# handles, labels = axs[0,0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', ncol=len(results), fontsize=14, frameon=False)
# plt.tight_layout(rect=[0, 0.03, 1, 0.94])
# plt.show()
# fig.savefig("fig/all.png", dpi=300)


# %%

def plot_pareto_front(results_mse, config: Config):
    """  
    Plots the L-shaped iso-performance contours from the simulation results.
    Smooths the contour plot using Gaussian filtering.
    Uses log labels on the colorbar, but does not use LogNorm for plotting.
    Rotates the canvas by 45 degrees.
    """
    from scipy.ndimage import gaussian_filter
    from matplotlib.transforms import Affine2D
    import numpy as np

    delta_list = config.delta_list
    n_inv_list = 1 / np.array(config.n_list)

    # Smooth the results using a Gaussian filter
    results_mse_smooth = gaussian_filter(results_mse, sigma=1)

    fig, ax = plt.subplots(figsize=(5, 5))

    # Apply a 45 degree rotation to the axes
    trans = Affine2D().rotate_deg(45) + ax.transData

    # Use the reversed colormap
    contour = ax.contourf(delta_list, n_inv_list, results_mse_smooth, levels=8, cmap='YlGnBu_r', transform=trans)
    # ! Solid contour lines
    lw = 2
    ax.contour(delta_list, n_inv_list, results_mse_smooth, levels=contour.levels, colors='black', linewidths=lw, alpha=0.8, transform=trans)

    # Add a colorbar with log labels
    # cbar = fig.colorbar(contour)
    # cbar.set_label('MSE')
    # # Set custom ticks for colorbar, including 5e-2
    # ticks = [1e-3, 5e-3, 1e-2, 2e-2]
    # cbar.set_ticks(ticks)
    # cbar.ax.yaxis.set_major_formatter(mticker.LogFormatter())
    # cbar.ax.set_yticklabels([np.format_float_scientific(tick,exp_digits=1,trim='-') for tick in ticks])

    # ax.set_xlabel('$\\delta$', fontsize=12)
    # ax.set_ylabel('$n^{-1}$', fontsize=12)
    ax.set_aspect(1./ax.get_data_ratio())
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # ! No box borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_linewidth(lw)
    # ax.spines['left'].set_linewidth(lw)
    # ax.plot(delta_list[0], n_inv_list[-1], ">k", clip_on=False, markersize=10, transform=trans)
    # ax.plot(delta_list[-1], n_inv_list[0], "^k", clip_on=False, markersize=10, transform=trans)

    # Fix: Compute rotated coordinates for the arrow annotation
    theta = np.deg2rad(45)
    def rotate_point(x, y, theta):
        x_rot = x * np.cos(theta) - y * np.sin(theta)
        y_rot = x * np.sin(theta) + y * np.cos(theta)
        return x_rot, y_rot
    # Start and end points
    x0, y0 = delta_list[-1], n_inv_list[-1]
    x1, y1 = delta_list[-1], n_inv_list[0]+0.02
    x0r, y0r = rotate_point(x0, y0, theta)
    x1r, y1r = rotate_point(x1, y1, theta)
    ax.annotate('', xy=(x1r, y1r), xytext=(x0r, y0r),
                arrowprops=dict(arrowstyle="->", color='black', lw=lw),
                annotation_clip=False)
    x0, y0 = delta_list[-1], n_inv_list[-1]
    x1, y1 = delta_list[0]+0.02, n_inv_list[-1]
    x0r, y0r = rotate_point(x0, y0, theta)
    x1r, y1r = rotate_point(x1, y1, theta)
    ax.annotate('', xy=(x1r, y1r), xytext=(x0r, y0r),
                arrowprops=dict(arrowstyle="->", color='black', lw=lw),
                annotation_clip=False)

    plt.tight_layout()
    plt.show()
    
    return fig

fig = plot_pareto_front(results, config)
# fig.savefig("fig/pareto_front.png", dpi=300)

