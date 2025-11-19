# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import multivariate_normal
import pickle
import datetime
import os

# %%
# Configuration
class Config:
    """Stores all parameters for the numerical experiment."""
    backup_dir = "bkup"
    runs = 3
    n = 20
    d = 5
    t = 60
    alpha = 0.01
    base_scale_a = 4
    delta_a = 0.1
    delta_b = 0.1
    noise_b = 0.5
    noise_a = 1
    lamda = 15.0
    num_clusters = 10
    init_cluster = 1e-2
    exact_assign = True
    pretrain_ratio = 0.5
    noise_a_list = [0.0, 0.5, 1.0, 2.0]
    # Basic
    heterogeneity_settings = {
        'homogeneous': (0.0, 0.0),
        # 'low': (0.05, 0.05),
        'medium': (0.3, 0.3),
        'high': (0.8, 0.8),
    }
    # Exhaustive
    # heterogeneity_settings = {}
    # for kernel_het in np.linspace(0.0, 0.9, 10):
    #     for reward_het in np.linspace(0.0, 0.9, 10):
    #         key = f'({kernel_het},{reward_het})'
    #         heterogeneity_settings[key] = (kernel_het, reward_het)

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
    A_bar_list = []
    b_bar_list = []
    for i in range(config.n):
        samples = distributions[i].rvs(size=n_samples_mc)
        A_bar_i = np.mean([A_func(s) for s in samples], axis=0)
        b_bar_i = np.mean([b_func(s,thetas_star[i]) for s in samples], axis=0)
        x_star_i = np.linalg.solve(A_bar_i, b_bar_i)
        x_stars.append(x_star_i)
        A_bar_list.append(A_bar_i)
        b_bar_list.append(b_bar_i)

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
        'A_bar_list': A_bar_list,
        'b_bar_list': b_bar_list,
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

    # return np.mean(errors, axis=1)
    return errors

# %%
def run_scaffold(data: dict, config: Config):
    """Baseline 3: SCAFFOLD."""
    print("Running SCAFFOLD...")
    x = np.zeros(config.d)  # Server model
    c = np.zeros(config.d)  # Server control variate
    
    # Client-side state
    ci = [np.zeros(config.d) for _ in range(config.n)] # Client control variates
    
    errors = np.zeros((config.t, config.n))
    
    K = 1 # Number of local steps, matching other algorithms
    eta_l = config.alpha # Local learning rate

    for t in range(config.t):
        x_t = x.copy()
        
        delta_y_agg = np.zeros(config.d)
        delta_c_agg = np.zeros(config.d)

        for i in range(config.n):
            s_t_i = data['distributions'][i].rvs()
            
            # Client update
            y_i = x_t.copy()
            
            # Local step(s). K=1 for this implementation.
            grad_i = data['A_func'](s_t_i) @ y_i - data['b_func'](s_t_i, data['thetas_star'][i])
            y_i -= eta_l * (grad_i - ci[i] + c)

            # Update client control variate (Option II from paper)
            c_new_i = ci[i] - c + (x_t - y_i) / (K * eta_l)

            # Deltas for aggregation
            delta_y_i = y_i - x_t
            delta_c_i = c_new_i - ci[i]
            
            delta_y_agg += delta_y_i
            delta_c_agg += delta_c_i
            
            # Update client state for next round
            ci[i] = c_new_i

        # Server update (ηg = 1 as per paper's experiments)
        x += delta_y_agg / config.n
        c += delta_c_agg / config.n
        
        # NOTE: Measure error of the single GLOBAL model against each agent's personal optimum
        for i in range(config.n):
            errors[t, i] = np.linalg.norm(x - data['x_stars'][i])**2
            
    return np.mean(errors, axis=1)

# %%
def run_pfedme(data: dict, config: Config):
    """Baseline 4: pFedMe."""
    print("Running pFedMe...")
    # Personalized models for each agent
    x = [np.zeros(config.d) for _ in range(config.n)]
    # Global model
    x_global = np.zeros(config.d)
    
    errors = np.zeros((config.t, config.n))

    for t in range(config.t):
        x_global_t = x_global.copy()
        
        # Local client updates
        for i in range(config.n):
            s_t_i = data['distributions'][i].rvs()
            
            # Gradient at the client's current personalized model
            grad_i = data['A_func'](s_t_i) @ x[i] - data['b_func'](s_t_i, data['thetas_star'][i])
            
            # pFedMe update rule
            regularization_term = config.lamda * (x[i] - x_global_t)
            x[i] -= config.alpha * (grad_i + regularization_term)
            
            errors[t, i] = np.linalg.norm(x[i] - data['x_stars'][i])**2

        # Update global model by averaging client models
        x_global = np.mean(x, axis=0)

    return errors

# %%
def run_ditto(data: dict, config: Config):
    """Baseline 5: Ditto."""
    print("Running Ditto...")
    # Personalized models for each agent
    v = [np.zeros(config.d) for _ in range(config.n)]
    # Global model
    w = np.zeros(config.d)
    
    errors = np.zeros((config.t, config.n))

    for t in range(config.t):
        w_t = w.copy()
        grad_agg = np.zeros(config.d)
        samples = [data['distributions'][i].rvs() for i in range(config.n)]

        # Global model update (one round of FedAvg)
        for i in range(config.n):
            s_t_i = samples[i]
            g_global_i = data['A_func'](s_t_i) @ w_t - data['b_func'](s_t_i, data['thetas_star'][i])
            grad_agg += g_global_i
        w = w_t - config.alpha * (grad_agg / config.n)

        # Personalized model update
        for i in range(config.n):
            s_t_i = samples[i]
            # Gradient at the client's current personalized model
            g_local_i = data['A_func'](s_t_i) @ v[i] - data['b_func'](s_t_i, data['thetas_star'][i])
            
            # Ditto update rule
            regularization_term = config.lamda * (v[i] - w)
            v[i] -= config.alpha * (g_local_i + regularization_term)
            
            errors[t, i] = np.linalg.norm(v[i] - data['x_stars'][i])**2
            
    return errors

# %%
def run_cluster(data: dict, config: Config):
    """Baseline 6: IFCA (Clustering)."""
    print("Running Cluster (IFCA)...")
    K = config.num_clusters
    # Initialize with small random noise to break symmetry
    cluster_models = [np.random.randn(config.d) * config.init_cluster for _ in range(K)]
    errors = np.zeros((config.t, config.n))

    for t in range(config.t):
        samples = [data['distributions'][i].rvs() for i in range(config.n)]
        
        # 1. Cluster Assignment
        client_clusters = np.zeros(config.n, dtype=int)
        if config.exact_assign:
            # Use exact expected loss for assignment
            A_bar_list = data['A_bar_list']
            b_bar_list = data['b_bar_list']
            for i in range(config.n):
                losses = []
                for j in range(K):
                    w_j = cluster_models[j]
                    A_bar_i = A_bar_list[i]
                    b_bar_i = b_bar_list[i]
                    # loss = 0.5 * np.dot(w_j, A_bar_i @ w_j) - np.dot(b_bar_i, w_j)
                    # Use MSE as loss
                    loss = np.linalg.norm(A_bar_i @ w_j - b_bar_i)**2
                    losses.append(loss)
                client_clusters[i] = np.argmin(losses)
        else:
            # Use stochastic loss approximation for assignment
            for i in range(config.n):
                s_t_i = samples[i]
                losses = []
                for j in range(K):
                    w_j = cluster_models[j]
                    loss = 0.5 * np.dot(w_j, data['A_func'](s_t_i) @ w_j) - \
                           np.dot(data['b_func'](s_t_i, data['thetas_star'][i]), w_j)
                losses.append(loss)
            client_clusters[i] = np.argmin(losses)

        # 2. Gradient Calculation and Aggregation
        cluster_grads = [[] for _ in range(K)]
        for i in range(config.n):
            s_t_i = samples[i]
            chosen_cluster_idx = client_clusters[i]
            chosen_model = cluster_models[chosen_cluster_idx]
            
            grad = data['A_func'](s_t_i) @ chosen_model - data['b_func'](s_t_i, data['thetas_star'][i])
            cluster_grads[chosen_cluster_idx].append(grad)
            
        # 3. Cluster Model Update
        for j in range(K):
            if cluster_grads[j]:
                avg_grad = np.mean(cluster_grads[j], axis=0)
                cluster_models[j] -= config.alpha * avg_grad

        # 4. Error Calculation
        for i in range(config.n):
            model_for_client = cluster_models[client_clusters[i]]
            errors[t, i] = np.linalg.norm(model_for_client - data['x_stars'][i])**2
            
    return np.mean(errors, axis=1)

# %%
def run_finetune(data: dict, config: Config):
    """Baseline 7: FedAvg then Fine-tune."""
    print("Running Fine-tune...")
    
    pretrain_steps = int(config.t * config.pretrain_ratio)
    
    fedavg_model = np.zeros(config.d)
    errors = np.zeros((config.t, config.n))

    # Phase 1: FedAvg Pre-training
    for t in range(pretrain_steps):
        grad_agg = np.zeros(config.d)
        for i in range(config.n):
            s_t_i = data['distributions'][i].rvs()
            g_t_i = data['A_func'](s_t_i) @ fedavg_model - data['b_func'](s_t_i, data['thetas_star'][i])
            grad_agg += g_t_i
        
        fedavg_model -= config.alpha * (grad_agg / config.n)
        
        for i in range(config.n):
            errors[t, i] = np.linalg.norm(fedavg_model - data['x_stars'][i])**2

    # Phase 2: Independent Fine-tuning
    personalized_models = [fedavg_model.copy() for _ in range(config.n)]
    for t in range(pretrain_steps, config.t):
        for i in range(config.n):
            s_t_i = data['distributions'][i].rvs()
            grad_i = data['A_func'](s_t_i) @ personalized_models[i] - data['b_func'](s_t_i, data['thetas_star'][i])
            personalized_models[i] -= config.alpha * grad_i
            errors[t, i] = np.linalg.norm(personalized_models[i] - data['x_stars'][i])**2
            
    return np.mean(errors, axis=1)

# %%
# Wrapper for experiments with multiple repeats and heterogeneity settings
def run_experiments_with_repeats(config):
    runs = config.runs
    n_iter = config.t
    heterogeneity_settings = config.heterogeneity_settings
    
    methods = {
        'ind': run_independent_learning,
        'fedavg': run_federated_averaging,
        # 'scaffold': run_scaffold,
        'pcl': run_personalized_collaborative,
        # 'pcl_i': run_personalized_collaborative,
        # 'pfedme': run_pfedme,
        # 'pfedme_i': run_pfedme,
        'ditto': run_ditto,
        # 'ditto_i': run_ditto,
        # 'cluster': run_cluster,
        'finetune': run_finetune,
    }

    # Initialize error arrays
    errors = {
        het: {method: np.zeros((runs, n_iter)) for method in methods}
        for het in heterogeneity_settings
    }

    def run_all_methods(data, config, run_idx, het_key):
        # Cache results for personalized methods to avoid re-running
        _temp_results = {}
        for method_key, method_func in methods.items():
            # For personalized methods that return per-agent errors
            if method_key in ['pcl', 'pfedme', 'ditto']:
                if method_key not in _temp_results:
                    _temp_results[method_key] = method_func(data, config)
                errors[het_key][method_key][run_idx] = np.mean(_temp_results[method_key], axis=1)
            elif method_key in ['pcl_i', 'pfedme_i', 'ditto_i']:
                base_method = method_key.replace('_i', '')
                if base_method not in _temp_results:
                    # Find the corresponding base method function
                    base_method_func = list(set(m for k, m in methods.items() if k.startswith(base_method)))[0]
                    _temp_results[base_method] = base_method_func(data, config)
                errors[het_key][method_key][run_idx] = _temp_results[base_method][:,0]
            # For non-personalized methods
            else:
                if method_key in errors[het_key]:
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
            for method in methods if method in errors[het]
        }
        for het in heterogeneity_settings
    }
    return results

# %%

# Main Execution and Variance Plotting
config = Config()

# Run
# results = run_experiments_with_noise(config)
results = run_experiments_with_repeats(config)

# Load
# backup_files = [f for f in os.listdir(config.backup_dir) if f.endswith("comp.pkl")]
# latest_file = max(backup_files, key=lambda x: x.split(".")[0])
# with open(os.path.join(config.backup_dir, latest_file), "rb") as f:
#     results = pickle.load(f)

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
        ('scaffold', 'SCAFFOLD', '*', 'C4'),
        ('pcl', 'PCL', 'D', 'C2'),
        ('pcl_i', 'Agent-specific PCL', 's', 'C3'),  # Changed marker to square ('s') for matplotlib
        ('pfedme', 'pFedMe', 'P', 'C5'),
        ('pfedme_i', 'Agent-specific pFedMe', 'X', 'C6'),
        ('ditto', 'Ditto', 'v', 'C7'),
        ('ditto_i', 'Agent-specific Ditto', '<', 'C8'),
        ('cluster', 'Clustered FL', '>', 'C9'),
        ('finetune', 'Fine-tune', 'd', 'C0'),
        ]:
        if key in results_dict:
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
fig, axs = plt.subplots(len(results) // 4 + (len(results) % 4 > 0), 4, figsize=(12, 4 * (len(results) // 4 + (len(results) % 4 > 0))), squeeze=False)

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
for het_key in results.keys():
    kernel_het, reward_het = config.heterogeneity_settings[het_key]
    # results_dict[het_key] = f'({round(kernel_het, 1)}, {round(reward_het, 1)})'
    # Use a dictionary mapping instead of match-case for compatibility
    het_map = {
        'homogeneous': 'Homogeneous',
        'low': 'Low Heterogeneity',
        'medium': 'Medium Heterogeneity',
        'high': 'High Heterogeneity',
    }
    results_dict[het_key] = het_map[het_key]


for i, (key, label) in enumerate(results_dict.items()):
    plot_results_on_axis( axs[i//4, i%4], results[key], config, label)

# fig.suptitle('Comparison of Learning Algorithms under Different Heterogeneity Levels', fontsize=18)
fig.supxlabel('# Samples', fontsize=14, y=0.12)
fig.supylabel('Mean Squared Error', fontsize=14)
handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(results), fontsize=14, frameon=False)
plt.tight_layout(rect=[0, 0.03, 1, 0.94])
# plt.show()
fig.savefig("fig/comp.png", dpi=300)

# %%
# Summary table
def compute_summary_table(results, config):
    het_keys = list(config.heterogeneity_settings.keys())
    delta_A_list = sorted(list(set([config.heterogeneity_settings[k][0] for k in het_keys])))
    delta_b_list = sorted(list(set([config.heterogeneity_settings[k][1] for k in het_keys])))
    n_A = len(delta_A_list)
    n_b = len(delta_b_list)

    # Define method pairs for improvement calculation
    method_pairs = [
        ('pcl', 'ind'),      # PCL over IL
        ('pcl', 'fedavg'),  # PCL over FL
        ('pcl_i', 'ind'),   # First agent over IL
        ('pcl_i', 'fedavg') # First agent over FL
    ]
    table = np.zeros((n_A, n_b, len(method_pairs) + 2))  # +2 for delta_A and delta_b

    for het_key in het_keys:
        delta_A, delta_b = config.heterogeneity_settings[het_key]
        row_idx = delta_A_list.index(delta_A)
        col_idx = delta_b_list.index(delta_b)
        res = results[het_key]
        means = {k: res[k][0] for k in res}
        # Compute last 10-step averages for all methods
        last10 = {k: np.mean(means[k][50:60]) for k in means}
        table[row_idx, col_idx, 0:2] = [delta_A, delta_b]
        for idx, (num_key, denom_key) in enumerate(method_pairs):
            denom = last10[denom_key]
            num = last10[num_key]
            imp = 100 * (denom - num) / denom if (denom != 0 and num <= 2*denom) else np.nan
            table[row_idx, col_idx, idx+2] = imp
    return table

# Plot heatmap of improvement of PCL over IL (table[:,:,2])
def plot_heatmap(table, index=2):
    cmp = plt.get_cmap('YlGnBu')
    cmp.set_bad(color='lightgray')  # Color for NaN values
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(table[:,:,index], cmap=cmp, aspect='auto', vmin=0, vmax=100)
    for (i, j), val in np.ndenumerate(table[:,:,index]):
        if np.isnan(val):
            text_color = 'black'
            display_val = "NaN"
        else:
            rgba = cmp(val / 100)  # Normalize val to [0,1] for colormap
            r, g, b, _ = rgba
            # Calculate luminance (perceived brightness)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = 'black' if luminance > 0.5 else 'white'
            display_val = str(round(val))
        ax.text(j, i, display_val, ha='center', va='center', color=text_color, fontsize=8)
    ax.set_xticks(range((table.shape[1])))
    ax.set_yticks(range((table.shape[0])))
    ax.set_xticklabels([round(v,2) for v in table[0,:,1]])
    ax.set_yticklabels([round(v,2) for v in table[:,0,0]])
    ax.set_xlabel('$\\delta_b$', fontsize=12, usetex=True)
    ax.set_ylabel('$\\delta_A$', fontsize=12, usetex=True)
    fig.colorbar(im, ax=ax, label='Improvement (%)')
    ax.invert_yaxis()  # Flip the y axis
    plt.tight_layout()
    plt.show()
    plt.tight_layout()
    return fig

# summary_table = compute_summary_table(results, config)
# # truncated_table = summary_table[:9, :9, :]  # For a 4x4 heatmap
# for index in range(2,4):
#     fig = plot_heatmap(summary_table, index)
#     fig.savefig(f"fig/heatmap_{index}.png", dpi=300)

