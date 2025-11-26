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
    temperature = 100 # Softmax temperature
    T = 2500  # Number of steps
    K = 1 # Synchronization period for FedAvg
    
    # Heterogeneity parameters
    eps_r = 0.9 # Reward heterogeneity
    eps_p = 0.9 # Transition heterogeneity

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
        scaled_q = q_star_opt / config.temperature
        exp_q = np.exp(scaled_q - np.max(scaled_q, axis=1, keepdims=True)) # amax for numerical stability
        pi_policy = exp_q / np.sum(exp_q, axis=1, keepdims=True)

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
        P_pi_s = np.einsum('sa,sap->sp', pi_policy, agents[i]['P']) # S x S transition matrix
    
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
        scaled_q = q_values / temperature
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
        scaled_q = q_values / temperature
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
    """Proposed Method: AffPCL SARSA."""
    print("Running AffPCL SARSA...")
    
    # Models
    thetas = [np.zeros(config.d) for _ in range(config.n)]
    theta_c = np.zeros(config.d)
    R_c = np.zeros((config.S, config.A))
    N_c = np.zeros((config.S, config.A)) # Visit counts for running average

    # State
    errors = np.zeros((config.T, config.n))
    s = [np.random.randint(config.S) for _ in range(config.n)]

    def get_action(q_values, temperature):
        if temperature == 0: # Greedy action
            return np.argmax(q_values)
        scaled_q = q_values / temperature
        exp_q = np.exp(scaled_q - np.max(scaled_q))
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(config.A, p=probs)

    for t in range(config.T):
        theta_c_t = theta_c.copy()
        
        # --- Step 1: Collect samples from all agents and update central reward model ---
        samples = []
        for i in range(config.n):
            q_s_i = data['phi'][s[i], :, :] @ thetas[i]
            a_i = get_action(q_s_i, config.temperature)
            r_i = data['agents'][i]['R'][s[i], a_i]
            s_next_i = np.random.choice(config.S, p=data['agents'][i]['P'][s[i], a_i, :])
            samples.append({'s': s[i], 'a': a_i, 'r': r_i, 's_next': s_next_i})

            # Update central reward model
            N_c[s[i], a_i] += 1
            R_c[s[i], a_i] += (r_i - R_c[s[i], a_i]) / N_c[s[i], a_i]
        
        # --- Step 2: Update central model using collected samples ---
        grad_c_agg = np.zeros(config.d)
        for j in range(config.n):
            sample_j = samples[j]
            s_j, a_j, _, s_next_j = sample_j['s'], sample_j['a'], sample_j['r'], sample_j['s_next']
            
            # Central model's TD error for this sample
            q_s_j_c = data['phi'][s_j, :, :] @ theta_c_t
            a_j_c = get_action(q_s_j_c, config.temperature) # Action central model would have taken
            r_c = R_c[s_j, a_j]

            q_s_next_j_c = data['phi'][s_next_j, :, :] @ theta_c_t
            a_next_j_c = get_action(q_s_next_j_c, config.temperature)

            td_error_c = r_c + config.gamma * q_s_next_j_c[a_next_j_c] - q_s_j_c[a_j_c]
            grad_c_agg += td_error_c * data['phi'][s_j, a_j_c, :]
        
        theta_c += config.alpha * (grad_c_agg / config.n)

        # --- Step 3: Local Personalized Updates ---
        for i in range(config.n):
            sample_i = samples[i]
            s_i, a_i, r_i, s_next_i = sample_i['s'], sample_i['a'], sample_i['r'], sample_i['s_next']

            # 1. Local gradient (based on agent's own action and reward)
            q_s_i = data['phi'][s_i, :, :] @ thetas[i]
            q_s_next_i = data['phi'][s_next_i, :, :] @ thetas[i]
            a_next_i = get_action(q_s_next_i, config.temperature)
            td_error_local = r_i + config.gamma * q_s_next_i[a_next_i] - q_s_i[a_i]
            g_local = td_error_local * data['phi'][s_i, a_i, :]

            # 2. Central gradient for correction terms (based on agent i's sample)
            q_s_i_c = data['phi'][s_i, :, :] @ theta_c_t
            a_i_c = get_action(q_s_i_c, config.temperature)
            r_c = R_c[s_i, a_i]

            q_s_next_i_c = data['phi'][s_next_i, :, :] @ theta_c_t
            a_next_i_c = get_action(q_s_next_i_c, config.temperature)
            
            td_error_c_i = r_c + config.gamma * q_s_next_i_c[a_next_i_c] - q_s_i_c[a_i_c]
            g_c_i = td_error_c_i * data['phi'][s_i, a_i_c, :]

            # 3. Importance correction and Bias correction
            rho_val = data['agents'][i]['rho_func'](s_i, a_i)
            g_rho_corr = rho_val * g_c_i
            g_bias = g_c_i
            
            # Full AffPCL update (gradient ascent)
            g_tilde = g_local + g_rho_corr - g_bias
            thetas[i] += config.alpha * g_tilde

            # Update state for next iteration
            s[i] = s_next_i
            errors[t, i] = np.linalg.norm(thetas[i] - data['agents'][i]['theta_star'])**2

    return np.mean(errors, axis=1)


def run_experiments(config):
    """Wrapper for experiments with multiple repeats."""
    
    methods = {
        'Independent': run_independent_sarsa,
        'FedAvg': run_fedavg_sarsa,
        'AffPCL': run_affpcl_sarsa,
    }

    results = {method: np.zeros((config.runs, config.T)) for method in methods}

    for run in range(config.runs):
        print(f"Run {run + 1}/{config.runs}")
        data = generate_mdp_data(config)
        for method_key, method_func in methods.items():
            errors = method_func(data, config)
            results[method_key][run, :] = errors

    # Compute mean and std
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
