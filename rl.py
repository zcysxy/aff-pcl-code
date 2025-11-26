import numpy as np
import matplotlib.pyplot as plt

class Config:
    """Stores all parameters for the RL experiment."""
    n = 10  # Number of agents
    S = 10  # Number of states
    A = 5   # Number of actions
    d = 20  # Feature dimension
    gamma = 0.1  # Discount factor
    alpha = 1e-1  # Learning rate
    temperature = 10 # Softmax temperature
    T = 2500  # Number of steps
    K = 1 # Synchronization period for FedAvg
    
    # Heterogeneity parameters
    eps_r = 0.5 # Reward heterogeneity
    eps_p = 0.5 # Transition heterogeneity

    # AffPCL option
    adaptive_density_ratio: bool = True

    runs = 1 # number of independent runs
    
def generate_mdp_data(config: Config):
    """
    Generates synthetic data for heterogeneous MDPs, using a softmax policy
    for calculating the ground truth theta_star and stationary distributions.
    """
    print("Generating MDP data...")
    
    # Base MDP
    P_base = np.random.rand(config.S, config.A, config.S)
    P_base = P_base / P_base.sum(axis=2, keepdims=True)
    R_base = np.random.rand(config.S, config.A)

    # Heterogeneous MDPs
    agents = []
    for i in range(config.n):
        if i == 0:
            P_i, R_i = P_base, R_base
        else:
            P_noise = np.random.rand(config.S, config.A, config.S)
            P_i = P_base + config.eps_p * P_noise
            P_i = np.maximum(P_i, 0)
            P_i = P_i / P_i.sum(axis=2, keepdims=True)
            R_noise = np.random.rand(config.S, config.A) * 2 - 1
            R_i = R_base + config.eps_r * R_noise
        agents.append({'P': P_i, 'R': R_i})
    
    # Feature representation phi(s, a)
    phi = np.random.randn(config.S, config.A, config.d)
    phi = phi / np.linalg.norm(phi, axis=2, keepdims=True)
    phi_flat = phi.reshape(-1, config.d)

    for i in range(config.n):
        # 1. Find the optimal q_star to define the reference softmax policy
        q_star_opt = np.zeros((config.S, config.A))

        for _ in range(1000): # Value iteration for optimal q
            v_star_opt = np.max(q_star_opt, axis=1)
            q_star_opt = agents[i]['R'] + config.gamma * (agents[i]['P'] @ v_star_opt)
    
        # 2. Define the softmax policy based on q_star_opt
        if config.temperature > 0:
            scaled_q = q_star_opt / config.temperature
            exp_q = np.exp(scaled_q - np.max(scaled_q, axis=1, keepdims=True))
            pi_policy = exp_q / np.sum(exp_q, axis=1, keepdims=True)
        else: # Greedy policy if temperature is zero
            pi_policy = np.zeros_like(q_star_opt)
            greedy_actions = np.argmax(q_star_opt, axis=1)
            pi_policy[np.arange(config.S), greedy_actions] = 1.0


        # 3. Calculate q_pi for this policy by solving the Bellman expectation equation
        P_pi_sa = np.zeros((config.S * config.A, config.S * config.A))
        for s in range(config.S):
            for a in range(config.A):
                for s_next in range(config.S):
                    p_s_next = agents[i]['P'][s, a, s_next]
                    for a_next in range(config.A):
                        P_pi_sa[s * config.A + a, s_next * config.A + a_next] = p_s_next * pi_policy[s_next, a_next]
    
        R_flat = agents[i]['R'].flatten()
        I = np.eye(config.S * config.A)
    
        try:
            q_pi_flat = np.linalg.solve(I - config.gamma * P_pi_sa, R_flat)
        except np.linalg.LinAlgError:
            q_pi_flat = np.linalg.pinv(I - config.gamma * P_pi_sa) @ R_flat
    
        agents[i]['q_pi'] = q_pi_flat.reshape(config.S, config.A)
    
        # 4. Project q_pi to get the ground truth theta_star for the on-policy setting
        agents[i]['theta_star'] = np.linalg.pinv(phi_flat) @ q_pi_flat
    
        # 5. Calculate stationary distribution d_sa for this policy
        P_pi_s = np.einsum('sa,sap->sp', pi_policy, agents[i]['P'])
    
        d_s = np.ones(config.S) / config.S
        for _ in range(1000): # Power iteration
            d_s = d_s @ P_pi_s
        d_s /= d_s.sum()
        d_sa = d_s[:, np.newaxis] * pi_policy
        agents[i]['d_sa'] = d_sa
    
    # Calculate average stationary distribution for rho
    avg_d_sa = np.mean([agent['d_sa'] for agent in agents], axis=0)
    
    for i in range(config.n):
        rho_i = agents[i]['d_sa'] / (avg_d_sa + 1e-9)
        agents[i]['rho_func'] = (lambda r: lambda s, a: r[s, a])(rho_i)
    
    data = {
        'agents': agents,
        'phi': phi,
    }
    
    print("MDP data generation complete.")
    
    return data


def run_independent_sarsa(data: dict, config: Config):
    """Baseline 1: Each agent learns entirely on its own."""
    print("Running Independent SARSA...")
    
    thetas = [np.zeros(config.d) for _ in range(config.n)]
    errors = np.zeros((config.T, config.n))
    
    s = [np.random.randint(config.S) for _ in range(config.n)]
    
    def get_action(q_values, temperature):
        if temperature == 0:
            return np.argmax(q_values)
        scaled_q = q_values / (temperature + 1e-9)
        exp_q = np.exp(scaled_q - np.max(scaled_q))
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(config.A, p=probs)

    for t in range(config.T):
        for i in range(config.n):
            q_s = data['phi'][s[i], :, :] @ thetas[i]
            a = get_action(q_s, config.temperature)
            
            s_next = np.random.choice(config.S, p=data['agents'][i]['P'][s[i], a, :])
            r = data['agents'][i]['R'][s[i], a]
            
            q_s_next = data['phi'][s_next, :, :] @ thetas[i]
            a_next = get_action(q_s_next, config.temperature)
            
            phi_sa = data['phi'][s[i], a, :]
            td_error = r + config.gamma * q_s_next[a_next] - q_s[a]
            thetas[i] += config.alpha * td_error * phi_sa
            
            s[i] = s_next
            
            errors[t, i] = np.linalg.norm(thetas[i] - data['agents'][i]['theta_star'])**2
            
    return np.mean(errors, axis=1)

def run_fedavg_sarsa(data: dict, config: Config):
    """Baseline 2: FedAvg SARSA."""
    print("Running FedAvg SARSA...")
    
    thetas = [np.zeros(config.d) for _ in range(config.n)]
    theta_global = np.zeros(config.d)
    errors = np.zeros((config.T, config.n))
    
    s = [np.random.randint(config.S) for _ in range(config.n)]
    
    def get_action(q_values, temperature):
        if temperature == 0:
            return np.argmax(q_values)
        scaled_q = q_values / (temperature + 1e-9)
        exp_q = np.exp(scaled_q - np.max(scaled_q))
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(config.A, p=probs)

    for t in range(config.T):
        for i in range(config.n):
            q_s = data['phi'][s[i], :, :] @ thetas[i]
            a = get_action(q_s, config.temperature)
            
            s_next = np.random.choice(config.S, p=data['agents'][i]['P'][s[i], a, :])
            r = data['agents'][i]['R'][s[i], a]
            
            q_s_next = data['phi'][s_next, :, :] @ thetas[i]
            a_next = get_action(q_s_next, config.temperature)
            
            phi_sa = data['phi'][s[i], a, :]
            td_error = r + config.gamma * q_s_next[a_next] - q_s[a]
            thetas[i] += config.alpha * td_error * phi_sa
            
            s[i] = s_next
            
            errors[t, i] = np.linalg.norm(theta_global - data['agents'][i]['theta_star'])**2
            
        if (t + 1) % config.K == 0:
            theta_global = np.mean(thetas, axis=0)
            thetas = [theta_global.copy() for _ in range(config.n)]
            
    return np.mean(errors, axis=1)


def run_affpcl_sarsa(data: dict, config: Config):
    """Proposed Method: AffPCL SARSA with optional adaptive density ratio."""
    print(f"Running AffPCL SARSA (Adaptive: {config.adaptive_density_ratio})...")
    
    # Models
    thetas = [np.zeros(config.d) for _ in range(config.n)]
    theta_c = np.zeros(config.d)
    R_c = np.zeros((config.S, config.A))
    N_c = np.zeros((config.S, config.A)) + 1e-9

    # State
    errors = np.zeros((config.T, config.n))
    s = [np.random.randint(config.S) for _ in range(config.n)]

    # Adaptive density estimation state
    d_s_counts = [np.ones(config.S) for _ in range(config.n)]

    def get_action(q_values, temperature):
        if temperature == 0:
            return np.argmax(q_values)
        scaled_q = q_values / (temperature + 1e-9)
        exp_q = np.exp(scaled_q - np.max(scaled_q))
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(config.A, p=probs)

    def get_policy_table(theta, temperature):
        q_table = data['phi'] @ theta
        if temperature == 0:
            pi_table = np.zeros_like(q_table)
            greedy_actions = np.argmax(q_table, axis=1)
            pi_table[np.arange(config.S), greedy_actions] = 1.0
            return pi_table
        scaled_q = q_table / (temperature + 1e-9)
        exp_q = np.exp(scaled_q - np.max(scaled_q, axis=1, keepdims=True))
        return exp_q / np.sum(exp_q, axis=1, keepdims=True)

    for t in range(config.T):
        theta_c_t = theta_c.copy()
        
        samples = []
        for i in range(config.n):
            q_s_i = data['phi'][s[i], :, :] @ thetas[i]
            a_i = get_action(q_s_i, config.temperature)
            r_i = data['agents'][i]['R'][s[i], a_i]
            s_next_i = np.random.choice(config.S, p=data['agents'][i]['P'][s[i], a_i, :])
            samples.append({'s': s[i], 'a': a_i, 'r': r_i, 's_next': s_next_i})

            N_c[s[i], a_i] += 1
            R_c[s[i], a_i] += (r_i - R_c[s[i], a_i]) / N_c[s[i], a_i]
        
        central_gradients = []
        for j in range(config.n):
            sample_j = samples[j]
            s_j, a_j, _, s_next_j = sample_j['s'], sample_j['a'], sample_j['r'], sample_j['s_next']
            
            q_s_j_c = data['phi'][s_j, :, :] @ theta_c_t
            r_c = R_c[s_j, a_j]
            q_s_next_j_c = data['phi'][s_next_j, :, :] @ theta_c_t
            a_next_j_c = get_action(q_s_next_j_c, config.temperature)

            td_error_c = r_c + config.gamma * q_s_next_j_c[a_next_j_c] - q_s_j_c[a_j]
            g_c_j = td_error_c * data['phi'][s_j, a_j, :]
            central_gradients.append(g_c_j)

        grad_c_agg = np.mean(central_gradients, axis=0)
        theta_c += config.alpha * grad_c_agg

        if config.adaptive_density_ratio:
            all_d_hat_sa = []
            for k in range(config.n):
                d_hat_k = d_s_counts[k] / np.sum(d_s_counts[k])
                pi_hat_k = get_policy_table(thetas[k], config.temperature)
                d_hat_sa_k = d_hat_k[:, np.newaxis] * pi_hat_k
                all_d_hat_sa.append(d_hat_sa_k)
            d_hat_sa_0 = np.mean(all_d_hat_sa, axis=0)
        
        for i in range(config.n):
            sample_i = samples[i]
            s_i, a_i, r_i, s_next_i = sample_i['s'], sample_i['a'], sample_i['r'], sample_i['s_next']

            q_s_i = data['phi'][s_i, :, :] @ thetas[i]
            q_s_next_i = data['phi'][s_next_i, :, :] @ thetas[i]
            a_next_i = get_action(q_s_next_i, config.temperature)
            td_error_local = r_i + config.gamma * q_s_next_i[a_next_i] - q_s_i[a_i]
            g_local = td_error_local * data['phi'][s_i, a_i, :]

            g_rho_corr = np.zeros(config.d)
            if config.adaptive_density_ratio:
                rho_hat_i = all_d_hat_sa[i] / (d_hat_sa_0 + 1e-9)
                for j in range(config.n):
                    s_j, a_j = samples[j]['s'], samples[j]['a']
                    g_c_j = central_gradients[j]
                    rho_val = rho_hat_i[s_j, a_j]
                    g_rho_corr += rho_val * g_c_j
            else: 
                for j in range(config.n):
                    s_j, a_j = samples[j]['s'], samples[j]['a']
                    g_c_j = central_gradients[j]
                    rho_val = data['agents'][i]['rho_func'](s_j, a_j)
                    g_rho_corr += rho_val * g_c_j
            g_rho_corr /= config.n

            g_bias = central_gradients[i]
            
            g_tilde = g_local + g_rho_corr - g_bias
            thetas[i] += config.alpha * g_tilde

            s[i] = s_next_i
            d_s_counts[i][s_next_i] += 1
            errors[t, i] = np.linalg.norm(thetas[i] - data['agents'][i]['theta_star'])**2

    return np.mean(errors, axis=1)


def run_experiments(config: Config):
    """Wrapper for experiments with multiple repeats."""
    
    methods = {
        'Independent': run_independent_sarsa,
        'FedAvg': run_fedavg_sarsa,
        'AffPCL': run_affpcl_sarsa,
        'AffPCL w/ DRE': run_affpcl_sarsa,
    }

    results = {method: np.zeros((config.runs, config.T)) for method in methods}

    for run in range(config.runs):
        print(f"Run {run + 1}/{config.runs}")
        data = generate_mdp_data(config)
        for method_key, method_func in methods.items():
            if method_key == 'AffPCL w/ DRE':
                config.adaptive_density_ratio = True
            elif method_key == 'AffPCL':
                config.adaptive_density_ratio = False
            errors = method_func(data, config)
            results[method_key][run, :] = errors

    final_results = {
        method: (
            results[method].mean(axis=0),
            results[method].std(axis=0)
        )
        for method in methods
    }
    return final_results

def plot_results(results, config):
    """Plots the results of the experiments."""
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(config.T)
    
    for method, (mean, std) in results.items():
        ax.plot(x, mean, label=method)
        ax.fill_between(x, mean - std / np.sqrt(config.runs), mean + std / np.sqrt(config.runs), alpha=0.2)
        
    ax.set_title('Comparison of RL Algorithms')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Mean Squared Error')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig('fig/rl_comp.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    config = Config()
    results = run_experiments(config)
    plot_results(results, config)
