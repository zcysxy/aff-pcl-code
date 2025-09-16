# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# %%
# --- 1. Configuration ---
class Config:
    """Stores all parameters for the numerical experiment."""
    n_agents = 20
    dim = 5
    n_iterations = 50
    learning_rate = 0.01
    kernel_heterogeneity = 0.5
    # Controls the spread of true reward vectors (theta_star_i)
    reward_heterogeneity = 0.5
    # Standard deviation for stochastic noise in A(s) and b(s)
    noise_std = 1
    n_runs = 20

# %%

# %%
# --- 2. Data Generation for Heterogeneous Systems ---
def generate_synthetic_data(config: Config):
    """
    Generates synthetic data for a multi-agent heterogeneous linear system
    based on the paper's setup.
    """
    print("Generating synthetic data...")

    # --- Create heterogeneous covariate distributions (mu_i) ---
    # Each agent's data distribution is a Gaussian with a different mean.
    means = [
        config.kernel_heterogeneity * np.random.randn(config.dim) # around zero
        for _ in range(config.n_agents)
    ]
    cov = np.eye(config.dim)
    distributions = [multivariate_normal(mean=m, cov=cov) for m in means]

    # --- Define shared stochastic feature A(s) and reward feature map Phi ---
    # A_bar_base is a shared, underlying positive definite matrix.
    _temp_A = np.random.rand(config.dim, config.dim)
    A_bar_base = _temp_A.T @ _temp_A + config.dim * np.eye(config.dim)

    # For simplicity, Phi is a fixed matrix, not dependent on the sample s.
    Phi = np.random.randn(config.dim, config.dim)
    Phi = Phi.T @ Phi

    def A_func(s):
        # The stochastic feature matrix A(s) depends on the sample s.
        # This ensures that E_{s~mu_i}[A(s)] is different for each agent.
        return (config.noise_std * np.outer(s, s) + np.eye(config.dim)) @ A_bar_base

    def b_func(s, theta_star):
        # The stochastic label b^i(s) follows a linear structure.
        return Phi @ theta_star # + config.noise_std * np.random.randn(config.dim)

    # --- Create heterogeneous true reward parameters (theta_star_i) ---
    theta_star_base = np.random.randn(config.dim)
    thetas_star = [
        theta_star_base + config.reward_heterogeneity * np.random.randn(config.dim)
        for _ in range(config.n_agents)
    ]

    # --- Calculate ground truth solutions x_star_i via Monte Carlo ---
    # The true solution x_star_i = inv(A_bar_i) @ b_bar_i, where the bars
    # denote expectation over mu_i. We approximate this with sampling.
    print("Calculating ground truth solutions via Monte Carlo...")
    n_samples_mc = 5000
    x_stars = []
    for i in range(config.n_agents):
        samples = distributions[i].rvs(size=n_samples_mc)
        A_bar_i = np.mean([A_func(s) for s in samples], axis=0)
        b_bar_i = np.mean([Phi @ thetas_star[i] for s in samples], axis=0)
        x_star_i = np.linalg.solve(A_bar_i, b_bar_i)
        x_stars.append(x_star_i)

    # --- Define density ratio function rho^i(s) ---
    # rho^i(s) = mu^i(s) / mu^0(s), where mu^0 = (1/n) * sum(mu^j)
    def rho_func(s, i):
        mu_i_pdf = distributions[i].pdf(s)
        mu_0_pdf = np.mean([dist.pdf(s) for dist in distributions])
        return mu_i_pdf / (mu_0_pdf + 1e-9) # Add epsilon for stability

    data = {
        'distributions': distributions,
        'A_func': A_func,
        'b_func': b_func,
        'Phi': Phi,
        'thetas_star': thetas_star,
        'x_stars': x_stars,
        'rho_func': rho_func,
    }
    print("Data generation complete.")
    return data

# %% [markdown]
"""
--- 3. Algorithms ---
"""


# %%
def run_independent_learning(data: dict, config: Config):
    """Baseline 1: Each agent learns entirely on its own."""
    print("Running Independent Learning...")
    x = [np.zeros(config.dim) for _ in range(config.n_agents)]
    errors = np.zeros((config.n_iterations, config.n_agents))

    for t in range(config.n_iterations):
        for i in range(config.n_agents):
            s_t_i = data['distributions'][i].rvs()
            # Local gradient: g_t^i(x_t^i) = A(s_t^i)x_t^i - b^i(s_t^i)
            g_t_i = data['A_func'](s_t_i) @ x[i] - data['b_func'](s_t_i, data['thetas_star'][i])
            x[i] -= config.learning_rate * g_t_i
            errors[t, i] = np.linalg.norm(x[i] - data['x_stars'][i])**2
            
    return np.mean(errors, axis=1)

# %%
def run_federated_averaging(data: dict, config: Config):
    """Baseline 2: All agents learn a single, unified model."""
    print("Running Federated Averaging...")
    x_0 = np.zeros(config.dim)  # Single central model
    errors = np.zeros((config.n_iterations, config.n_agents))

    for t in range(config.n_iterations):
        grad_agg = np.zeros(config.dim)
        for i in range(config.n_agents):
            s_t_i = data['distributions'][i].rvs()
            # Each agent computes a gradient at the central model x_0
            g_t_i = data['A_func'](s_t_i) @ x_0 - data['b_func'](s_t_i, data['thetas_star'][i])
            grad_agg += g_t_i
        
        x_0 -= config.learning_rate * (grad_agg / config.n_agents)
        
        # Measure error of the single model against each agent's personal optimum
        for i in range(config.n_agents):
            errors[t, i] = np.linalg.norm(x_0 - data['x_stars'][i])**2
            
    return np.mean(errors, axis=1)

# %%
def run_personalized_collaborative(data: dict, config: Config):
    """Proposed Method: Personalized Collaborative Learning with affinity-based variance reduction."""
    print("Running Personalized Collaborative Learning...")
    # Personalized models for each agent
    x = [np.zeros(config.dim) for _ in range(config.n_agents)]
    # Central variables maintained on the server
    x_c = np.zeros(config.dim)
    theta_c = np.zeros(config.dim) # For learning the central reward
    errors = np.zeros((config.n_iterations, config.n_agents))

    for t in range(config.n_iterations):
        # In each step, every agent draws a fresh sample
        samples = [dist.rvs() for dist in data['distributions']]

        # --- Server-side: Update central reward (theta_c) and central model (x_c) ---
        grad_agg_b = np.zeros(config.dim)
        grad_agg_c = np.zeros(config.dim)
        
        b_hat_c_t = lambda s: data['Phi'] @ theta_c # Learned central reward at step t

        for j in range(config.n_agents):
            s_t_j = samples[j]
            # Gradient for central reward learning
            grad_agg_b += data['Phi'] @ theta_c - data['b_func'](s_t_j, data['thetas_star'][j])
            # Gradient for central model learning
            grad_agg_c += data['A_func'](s_t_j) @ x_c - b_hat_c_t(s_t_j)
        
        theta_c -= config.learning_rate * (grad_agg_b / config.n_agents)
        x_c -= config.learning_rate * (grad_agg_c / config.n_agents)

        # --- Client-side: Update personalized models x_i ---
        for i in range(config.n_agents):
            s_t_i = samples[i]
            
            # 1. Local gradient: g_t^i(x_t^i)
            g_t_i = data['A_func'](s_t_i) @ x[i] - data['b_func'](s_t_i, data['thetas_star'][i])
            
            # 2. Importance-corrected central gradient: (rho^i circ g_t^0)(x_t^c)
            g_rho_corr = np.zeros(config.dim)
            for j in range(config.n_agents):
                s_t_j = samples[j]
                g_c_arrow_j = data['A_func'](s_t_j) @ x_c - b_hat_c_t(s_t_j)
                rho_i_j = data['rho_func'](s_t_j, i)
                g_rho_corr += rho_i_j * g_c_arrow_j
            g_rho_corr /= config.n_agents
            
            # 3. Bias correction term: g_t^{c->i}(x_t^c)
            g_bias_corr = data['A_func'](s_t_i) @ x_c - b_hat_c_t(s_t_i)
            
            # Full personalized update direction (Eq. 6 from the paper)
            g_tilde_i = g_t_i + g_rho_corr - g_bias_corr
            
            x[i] -= config.learning_rate * g_tilde_i
            errors[t, i] = np.linalg.norm(x[i] - data['x_stars'][i])**2

    return np.mean(errors, axis=1)

# %%
# --- Wrapper for Multiple Runs and Variance Plotting ---
def run_experiments_with_repeats(config):
    n_runs = config.n_runs
    n_iter = config.n_iterations
    
    # Arrays to store errors for each run
    errors_ind_homogeneous = np.zeros((n_runs, n_iter))
    errors_fedavg_homogeneous = np.zeros((n_runs, n_iter))
    errors_pcl_homogeneous = np.zeros((n_runs, n_iter))
    errors_ind_low = np.zeros((n_runs, n_iter))
    errors_fedavg_low = np.zeros((n_runs, n_iter))
    errors_pcl_low = np.zeros((n_runs, n_iter))
    errors_ind_medium = np.zeros((n_runs, n_iter))
    errors_fedavg_medium = np.zeros((n_runs, n_iter))
    errors_pcl_medium = np.zeros((n_runs, n_iter))
    errors_ind_high = np.zeros((n_runs, n_iter))
    errors_fedavg_high = np.zeros((n_runs, n_iter))
    errors_pcl_high = np.zeros((n_runs, n_iter))
    
    for run in range(n_runs):
        # Homogeneous
        print(f"Run {run+1}/{n_runs} - Homogeneous")
        config.kernel_heterogeneity = 0.0
        config.reward_heterogeneity = 0.0
        data_homogeneous = generate_synthetic_data(config)
        errors_ind_homogeneous[run] = run_independent_learning(data_homogeneous, config)
        errors_fedavg_homogeneous[run] = run_federated_averaging(data_homogeneous, config)
        errors_pcl_homogeneous[run] = run_personalized_collaborative(data_homogeneous, config)

        # Low Heterogeneity
        print(f"Run {run+1}/{n_runs} - Low Heterogeneity")
        config.kernel_heterogeneity = 0.2
        config.reward_heterogeneity = 0.2
        data_low_het = generate_synthetic_data(config)
        errors_ind_low[run] = run_independent_learning(data_low_het, config)
        errors_fedavg_low[run] = run_federated_averaging(data_low_het, config)
        errors_pcl_low[run] = run_personalized_collaborative(data_low_het, config)

        # Medium Heterogeneity
        print(f"Run {run+1}/{n_runs} - Medium Heterogeneity")
        config.kernel_heterogeneity = 0.6
        config.reward_heterogeneity = 0.6
        data_medium_het = generate_synthetic_data(config)
        errors_ind_medium[run] = run_independent_learning(data_medium_het, config)
        errors_fedavg_medium[run] = run_federated_averaging(data_medium_het, config)
        errors_pcl_medium[run] = run_personalized_collaborative(data_medium_het, config)

        # High Heterogeneity
        print(f"Run {run+1}/{n_runs} - High Heterogeneity")
        config.kernel_heterogeneity = 1
        config.reward_heterogeneity = 1
        data_high_het = generate_synthetic_data(config)
        errors_ind_high[run] = run_independent_learning(data_high_het, config)
        errors_fedavg_high[run] = run_federated_averaging(data_high_het, config)
        errors_pcl_high[run] = run_personalized_collaborative(data_high_het, config)

    
    # Compute mean and std
    results = {
        'homogeneous': {
            'ind': (errors_ind_homogeneous.mean(axis=0), errors_ind_homogeneous.std(axis=0)),
            'fedavg': (errors_fedavg_homogeneous.mean(axis=0), errors_fedavg_homogeneous.std(axis=0)),
            'pcl': (errors_pcl_homogeneous.mean(axis=0), errors_pcl_homogeneous.std(axis=0)),
        },
        'low': {
            'ind': (errors_ind_low.mean(axis=0), errors_ind_low.std(axis=0)),
            'fedavg': (errors_fedavg_low.mean(axis=0), errors_fedavg_low.std(axis=0)),
            'pcl': (errors_pcl_low.mean(axis=0), errors_pcl_low.std(axis=0)),
        },
        'medium': {
            'ind': (errors_ind_medium.mean(axis=0), errors_ind_medium.std(axis=0)),
            'fedavg': (errors_fedavg_medium.mean(axis=0), errors_fedavg_medium.std(axis=0)),
            'pcl': (errors_pcl_medium.mean(axis=0), errors_pcl_medium.std(axis=0)),
        },
        'high': {
            'ind': (errors_ind_high.mean(axis=0), errors_ind_high.std(axis=0)),
            'fedavg': (errors_fedavg_high.mean(axis=0), errors_fedavg_high.std(axis=0)),
            'pcl': (errors_pcl_high.mean(axis=0), errors_pcl_high.std(axis=0)),
        }
    }
    return results


# %%

# Main Execution and Variance Plotting ---
config = Config()

# results = run_experiments_with_repeats(config)

import pickle
import datetime
import os
backup_dir = "bkup"
backup_files = [f for f in os.listdir(backup_dir) if f.endswith(".pkl")]
latest_file = max(backup_files, key=lambda x: x.split(".")[0])
with open(os.path.join(backup_dir, latest_file), "rb") as f:
    results = pickle.load(f)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f"bkup/{timestamp}.pkl", "wb") as f:
    pickle.dump(results, f)


# %%

def plot_results_on_axis(ax, results_dict, config, title):
    x = np.arange(config.n_iterations)
    for key, label, marker, color in [
        ('ind', 'Independent Learning', 'o', 'C0'),
        ('fedavg', 'Federated Averaging', '^', 'C1'),  # triangle marker
        ('pcl', 'Personalized Collaborative', 'D', 'C2'),
        ]:
        mean, std = results_dict[key]
        ax.plot(x, mean, label=label, marker=marker, color=color, markevery=10, markersize=8)
        ax.fill_between(x, mean-1.64*std/np.sqrt(config.n_runs), mean+1.64*std/np.sqrt(config.n_runs), color=color, alpha=0.2)
    ax.set_title(title, fontsize=14)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_aspect(1./ax.get_data_ratio())
    # ax.grid(True, which="both", ls="--", alpha=0.6)  # grid removed

fig, axs = plt.subplots(1, 4, figsize=(12, 4))

plot_results_on_axis(axs[0], results['homogeneous'], config, 'Homogeneous')
plot_results_on_axis(axs[1], results['low'], config, 'Low Heterogeneity')
plot_results_on_axis(axs[2], results['medium'], config, 'Medium Heterogeneity')
plot_results_on_axis(axs[3], results['high'], config, 'High Heterogeneity')

# fig.suptitle('Comparison of Learning Algorithms under Different Heterogeneity Levels', fontsize=18)
fig.supxlabel('# Samples', fontsize=14, y=0.12)
fig.supylabel('Mean Squared Error', fontsize=14)
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=14, frameon=False)
plt.tight_layout(rect=[0, 0.03, 1, 0.94])
plt.show()
