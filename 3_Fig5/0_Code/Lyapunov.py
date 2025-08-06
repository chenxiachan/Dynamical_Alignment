import math
from tqdm import tqdm
from encoding import *


device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")

@jit(nopython=True)
def compute_jacobian_lorenz(x, y, z, sigma=10, beta=2.667, rho=28):
    """Compute Jacobian matrix for the Lorenz system at point (x,y,z)"""
    jacobian = np.zeros((3, 3), dtype=np.float64)

    # dx/dt = sigma * (y - x)
    jacobian[0, 0] = -sigma
    jacobian[0, 1] = sigma
    jacobian[0, 2] = 0

    # dy/dt = x * (rho - z) - y
    jacobian[1, 0] = rho - z
    jacobian[1, 1] = -1
    jacobian[1, 2] = -x

    # dz/dt = x * y - beta * z
    jacobian[2, 0] = y
    jacobian[2, 1] = x
    jacobian[2, 2] = -beta

    return jacobian


@jit(nopython=True)
def compute_jacobian_rossler(x, y, z, a=0.2, b=0.2, c=5.7):
    """Compute Jacobian matrix for the Rössler system at point (x,y,z)"""
    jacobian = np.zeros((3, 3), dtype=np.float64)

    # dx/dt = -y - z
    jacobian[0, 0] = 0
    jacobian[0, 1] = -1
    jacobian[0, 2] = -1

    # dy/dt = x + a*y
    jacobian[1, 0] = 1
    jacobian[1, 1] = a
    jacobian[1, 2] = 0

    # dz/dt = b + z*(x-c)
    jacobian[2, 0] = z
    jacobian[2, 1] = 0
    jacobian[2, 2] = x - c

    return jacobian


@jit(nopython=True)
def compute_jacobian_aizawa(x, y, z, alpha=0.95, beta=0.7, gamma=0.6, delta=3.5, epsilon=0.25, zeta=0.1):
    """Compute Jacobian matrix for the Aizawa system at point (x,y,z)"""
    jacobian = np.zeros((3, 3), dtype=np.float64)

    # dx/dt = (z-beta)*x - delta*y
    jacobian[0, 0] = z - beta
    jacobian[0, 1] = -delta
    jacobian[0, 2] = x

    # dy/dt = delta*x + (z-beta)*y
    jacobian[1, 0] = delta
    jacobian[1, 1] = z - beta
    jacobian[1, 2] = y

    # dz/dt = gamma + alpha*z - z^3/3 - (x^2+y^2)*(1+epsilon*z) + zeta*z*x^3
    jacobian[2, 0] = -2 * x * (1 + epsilon * z) + 3 * zeta * z * x * x
    jacobian[2, 1] = -2 * y * (1 + epsilon * z)
    jacobian[2, 2] = alpha - z * z - epsilon * (x * x + y * y) + zeta * x * x * x

    return jacobian


@jit(nopython=True)
def compute_jacobian_nose_hoover(x, y, z, alpha=1.0):
    """Compute Jacobian matrix for the Nosé-Hoover system at point (x,y,z)"""
    jacobian = np.zeros((3, 3), dtype=np.float64)

    # dx/dt = y
    jacobian[0, 0] = 0
    jacobian[0, 1] = 1
    jacobian[0, 2] = 0

    # dy/dt = -x - y*z
    jacobian[1, 0] = -1
    jacobian[1, 1] = -z
    jacobian[1, 2] = -y

    # dz/dt = y^2 - alpha
    jacobian[2, 0] = 0
    jacobian[2, 1] = 2 * y
    jacobian[2, 2] = 0

    return jacobian


@jit(nopython=True)
def compute_jacobian_sprott(x, y, z, a=3.0):
    """Compute Jacobian matrix for the Sprott Case C system at point (x,y,z)"""
    jacobian = np.zeros((3, 3), dtype=np.float64)

    # dx/dt = y*z
    jacobian[0, 0] = 0
    jacobian[0, 1] = z
    jacobian[0, 2] = y

    # dy/dt = x - y
    jacobian[1, 0] = 1
    jacobian[1, 1] = -1
    jacobian[1, 2] = 0

    # dz/dt = 1 - a*x^2
    jacobian[2, 0] = -2 * a * x
    jacobian[2, 1] = 0
    jacobian[2, 2] = 0

    return jacobian


@jit(nopython=True)
def compute_jacobian_chua(x, y, z, alpha=15.6, beta=28.58, gamma=0.0):
    """Compute Jacobian matrix for the Chua circuit system at point (x,y,z)"""
    jacobian = np.zeros((3, 3), dtype=np.float64)

    # Chua diode parameters
    m0 = -1.143
    m1 = -0.714

    # Calculate h(x) derivative
    if x > 1:
        h_x_prime = m1
    elif x < -1:
        h_x_prime = m1
    else:
        h_x_prime = m0

    # dx/dt = alpha * (y - x - h(x))
    jacobian[0, 0] = -alpha * (1 + h_x_prime)
    jacobian[0, 1] = alpha
    jacobian[0, 2] = 0

    # dy/dt = x - y + z
    jacobian[1, 0] = 1
    jacobian[1, 1] = -1
    jacobian[1, 2] = 1

    # dz/dt = -beta*y - gamma*z
    jacobian[2, 0] = 0
    jacobian[2, 1] = -beta
    jacobian[2, 2] = -gamma

    return jacobian


@jit(nopython=True)
def compute_jacobian_mixed_oscillator(x, y, z, alpha=1.0, beta=0.2, delta=0.1, gamma=0.1, omega=1.0):
    jacobian = np.zeros((3, 3), dtype=np.float64)

    # dx/dt = y
    jacobian[0, 0] = 0
    jacobian[0, 1] = 1
    jacobian[0, 2] = 0

    # dy/dt = -alpha*x - beta*x^3 - delta*y + gamma*z
    jacobian[1, 0] = -alpha - 3 * beta * x * x
    jacobian[1, 1] = -delta
    jacobian[1, 2] = gamma

    # dz/dt = -omega*x - delta*z + gamma*x*y
    jacobian[2, 0] = -omega + gamma * y
    jacobian[2, 1] = gamma * x
    jacobian[2, 2] = -delta

    return jacobian


def compute_lyapunov_exponent(attractor_type, steps=200, dt=0.01, transient=0,
                              initial_conditions=(0.1, 0.1, 0.1, 0.1), attractor_params=None,
                              qr_interval=5):
    """
    Compute Lyapunov exponents for the specified attractor type using improved numerical methods

    Parameters:
        attractor_type: Type of attractor ('lorenz', 'rossler', etc.)
        steps: Number of steps for computation
        dt: Time step size
        transient: Initial transient steps to discard
        initial_conditions: Initial conditions (x0, y0, z0, w0)
        attractor_params: Dictionary of attractor parameters
        qr_interval: Interval for QR decomposition

    Returns:
        lyapunov_exponents: Array of all Lyapunov exponents sorted from largest to smallest
    """
    # Select the appropriate Jacobian computation function
    if attractor_type == 'lorenz':
        compute_jacobian = compute_jacobian_lorenz
        n_dims = 3
    elif attractor_type == 'rossler':
        compute_jacobian = compute_jacobian_rossler
        n_dims = 3
    elif attractor_type == 'aizawa':
        compute_jacobian = compute_jacobian_aizawa
        n_dims = 3
    elif attractor_type == 'nose_hoover':
        compute_jacobian = compute_jacobian_nose_hoover
        n_dims = 3
    elif attractor_type == 'sprott':
        compute_jacobian = compute_jacobian_sprott
        n_dims = 3
    elif attractor_type == 'chua':
        compute_jacobian = compute_jacobian_chua
        n_dims = 3
    elif attractor_type == 'mixed_oscillator':
        compute_jacobian = compute_jacobian_mixed_oscillator
        n_dims = 3
    else:
        raise ValueError(f"Unknown attractor type: {attractor_type}")

    # Set parameters with defaults
    params = {}
    if attractor_params:
        params.update(attractor_params)

    # Get initial conditions based on dimensionality
    if n_dims == 2:
        x0, y0 = initial_conditions[:2]
    elif n_dims == 3:
        x0, y0, z0 = initial_conditions[:3]
    elif n_dims == 4:
        x0, y0, z0, w0 = initial_conditions[:4]

    # Run through initial transient
    if attractor_type == 'lorenz':
        trajectory = lorenz_transformer_vectorized(x0, y0, z0,
                                                   sigma=params.get('sigma', 10),
                                                   beta=params.get('beta', 2.667),
                                                   rho=params.get('rho', 28),
                                                   tmax=transient * dt, h=dt)
        x, y, z = trajectory[-1]
        current_state = (x, y, z)
    elif attractor_type == 'rossler':
        trajectory = rossler_transformer_vectorized(x0, y0, z0,
                                                    a=params.get('a', 0.2),
                                                    b=params.get('b', 0.2),
                                                    c=params.get('c', 5.7),
                                                    tmax=transient * dt, h=dt)
        x, y, z = trajectory[-1]
        current_state = (x, y, z)
    elif attractor_type == 'aizawa':
        trajectory = aizawa_transformer_vectorized(x0, y0, z0,
                                                   alpha=params.get('alpha', 0.95),
                                                   beta=params.get('beta', 0.7),
                                                   gamma=params.get('gamma', 0.6),
                                                   delta=params.get('delta', 3.5),
                                                   epsilon=params.get('epsilon', 0.25),
                                                   zeta=params.get('zeta', 0.1),
                                                   tmax=transient * dt, h=dt)
        x, y, z = trajectory[-1]
        current_state = (x, y, z)
    elif attractor_type == 'nose_hoover':
        trajectory = nose_hoover_transformer_vectorized(x0, y0, z0,
                                                        alpha=params.get('alpha', 1.0),
                                                        tmax=transient * dt, h=dt)
        x, y, z = trajectory[-1]
        current_state = (x, y, z)
    elif attractor_type == 'sprott':
        trajectory = sprott_case_c_transformer_vectorized(x0, y0, z0,
                                                          a=params.get('a', 3.0),
                                                          tmax=transient * dt, h=dt)
        x, y, z = trajectory[-1]
        current_state = (x, y, z)
    elif attractor_type == 'chua':
        trajectory = chua_circuit_transformer_vectorized(x0, y0, z0,
                                                         alpha=params.get('alpha', 15.6),
                                                         beta=params.get('beta', 28.58),
                                                         gamma=params.get('gamma', 0.0),
                                                         tmax=transient * dt, h=dt)
        x, y, z = trajectory[-1]
        current_state = (x, y, z)
    elif attractor_type == 'mixed_oscillator':
        trajectory = mixed_oscillator_transformer_vectorized(
            x0, y0, z0,
            alpha=params.get('alpha', 1.0),
            beta=params.get('beta', 0.2),
            delta=params.get('delta', 0.1),
            gamma=params.get('gamma', 0.1),
            omega=params.get('omega', 1.0),
            drive=params.get('drive', 0.0),
            tmax=transient * dt,
            h=dt
        )
        x, y, z = trajectory[-1]
        current_state = (x, y, z)

    # Initialize orthogonal basis and cumulative vector
    Q = np.eye(n_dims)
    cum_sum = np.zeros(n_dims)

    # Count actual QR decompositions for accurate normalization
    qr_count = 0

    # Main loop - compute Lyapunov exponents
    for i in range(steps):
        # Compute Jacobian matrix at current point
        if attractor_type == 'lorenz':
            x, y, z = current_state
            J = compute_jacobian(x, y, z,
                                 sigma=params.get('sigma', 10),
                                 beta=params.get('beta', 2.667),
                                 rho=params.get('rho', 28))
        elif attractor_type == 'rossler':
            x, y, z = current_state
            J = compute_jacobian(x, y, z,
                                 a=params.get('a', 0.2),
                                 b=params.get('b', 0.2),
                                 c=params.get('c', 5.7))
        elif attractor_type == 'aizawa':
            x, y, z = current_state
            J = compute_jacobian(x, y, z,
                                 alpha=params.get('alpha', 0.95),
                                 beta=params.get('beta', 0.7),
                                 gamma=params.get('gamma', 0.6),
                                 delta=params.get('delta', 3.5),
                                 epsilon=params.get('epsilon', 0.25),
                                 zeta=params.get('zeta', 0.1))
        elif attractor_type == 'nose_hoover':
            x, y, z = current_state
            J = compute_jacobian(x, y, z,
                                 alpha=params.get('alpha', 1.0))
        elif attractor_type == 'sprott':
            x, y, z = current_state
            J = compute_jacobian(x, y, z,
                                 a=params.get('a', 3.0))
        elif attractor_type == 'chua':
            x, y, z = current_state
            J = compute_jacobian(x, y, z,
                                 alpha=params.get('alpha', 15.6),
                                 beta=params.get('beta', 28.58),
                                 gamma=params.get('gamma', 0.0))
        elif attractor_type == 'mixed_oscillator':
            x, y, z = current_state
            J = compute_jacobian(x, y, z,
                                 alpha=params.get('alpha', 1.0),
                                 beta=params.get('beta', 0.2),
                                 delta=params.get('delta', 0.1),
                                 gamma=params.get('gamma', 0.1),
                                 omega=params.get('omega', 1.0))

        # One step integration to update current state
        if attractor_type == 'lorenz':
            x, y, z = current_state
            x_new, y_new, z_new = lorenz_transformer_vectorized(x, y, z,
                                                                sigma=params.get('sigma', 10),
                                                                beta=params.get('beta', 2.667),
                                                                rho=params.get('rho', 28),
                                                                tmax=dt, h=dt)[-1]
            current_state = (x_new, y_new, z_new)
        elif attractor_type == 'rossler':
            x, y, z = current_state
            x_new, y_new, z_new = rossler_transformer_vectorized(x, y, z,
                                                                 a=params.get('a', 0.2),
                                                                 b=params.get('b', 0.2),
                                                                 c=params.get('c', 5.7),
                                                                 tmax=dt, h=dt)[-1]
            current_state = (x_new, y_new, z_new)
        elif attractor_type == 'aizawa':
            x, y, z = current_state
            x_new, y_new, z_new = aizawa_transformer_vectorized(x, y, z,
                                                                alpha=params.get('alpha', 0.95),
                                                                beta=params.get('beta', 0.7),
                                                                gamma=params.get('gamma', 0.6),
                                                                delta=params.get('delta', 3.5),
                                                                epsilon=params.get('epsilon', 0.25),
                                                                zeta=params.get('zeta', 0.1),
                                                                tmax=dt, h=dt)[-1]
            current_state = (x_new, y_new, z_new)
        elif attractor_type == 'nose_hoover':
            x, y, z = current_state
            x_new, y_new, z_new = nose_hoover_transformer_vectorized(x, y, z,
                                                                     alpha=params.get('alpha', 1.0),
                                                                     tmax=dt, h=dt)[-1]
            current_state = (x_new, y_new, z_new)
        elif attractor_type == 'sprott':
            x, y, z = current_state
            x_new, y_new, z_new = sprott_case_c_transformer_vectorized(x, y, z,
                                                                       a=params.get('a', 3.0),
                                                                       tmax=dt, h=dt)[-1]
            current_state = (x_new, y_new, z_new)
        elif attractor_type == 'chua':
            x, y, z = current_state
            x_new, y_new, z_new = chua_circuit_transformer_vectorized(x, y, z,
                                                                      alpha=params.get('alpha', 15.6),
                                                                      beta=params.get('beta', 28.58),
                                                                      gamma=params.get('gamma', 0.0),
                                                                      tmax=dt, h=dt)[-1]
            current_state = (x_new, y_new, z_new)
        elif attractor_type == 'mixed_oscillator':
            x, y, z = current_state
            x_new, y_new, z_new = mixed_oscillator_transformer_vectorized(
                x, y, z,
                alpha=params.get('alpha', 1.0),
                beta=params.get('beta', 0.2),
                delta=params.get('delta', 0.1),
                gamma=params.get('gamma', 0.1),
                omega=params.get('omega', 1.0),
                drive=params.get('drive', 0.0),
                tmax=dt, h=dt
            )[-1]
            current_state = (x_new, y_new, z_new)

        # Improve Q update using a better approximation of exp(J*dt)
        # Instead of simple Euler method (I + dt*J), we use a 4th order approximation
        # exp(J*dt) ≈ I + dt*J + (dt*J)^2/2! + (dt*J)^3/3! + (dt*J)^4/4!

        # Start with identity and multiply by J
        J_dt = J * dt
        Q_update = np.eye(n_dims)
        J_power = np.eye(n_dims)

        # Add higher order terms: I + dt*J + (dt*J)^2/2! + ...
        for order in range(1, 5):  # Up to 4th order
            J_power = np.dot(J_power, J_dt)
            Q_update += J_power / math.factorial(order)

        # Apply the update to Q
        Q = np.dot(Q_update, Q)

        # Perform QR decomposition at specified intervals
        if (i + 1) % qr_interval == 0:
            Q, R = np.linalg.qr(Q)
            qr_count += 1

            # Update cumulative sum of log(diagonal(R))
            for j in range(n_dims):
                cum_sum[j] += np.log(abs(R[j, j]))

    # Calculate Lyapunov exponents with correct normalization
    # Divide by the total integration time actually used for QR decompositions
    lyapunov_exponents = cum_sum / (qr_count * dt * qr_interval)

    # Sort Lyapunov exponents from largest to smallest
    lyapunov_exponents = np.sort(lyapunov_exponents)[::-1]

    return lyapunov_exponents


def analyze_lyapunov_exponents(config, results_dir='lyapunov_results'):
    """
    Analyze Lyapunov exponent characteristics for different attractors

    Parameters:
        config: Configuration object
        results_dir: Directory to save results

    Returns:
        DataFrame with results
    """
    os.makedirs(results_dir, exist_ok=True)

    # Attractors to analyze
    attractors = ['lorenz', 'rossler', 'aizawa', 'nose_hoover', 'sprott', 'chua']

    # Store results
    results = {
        'attractor': [],
        'largest_lyapunov': [],
        'all_lyapunov': [],
        'lyapunov_sum': [],
    }

    print("Analyzing Lyapunov exponents for different attractors...")
    for attractor in tqdm(attractors, desc="Computing Lyapunov exponents"):
        try:
            print(f"\nProcessing {attractor} attractor...")
            print('Tmax:', int(config.tmax*100))
            # Compute Lyapunov exponents
            if attractor == 'lorenz':
                lyapunov_exponents = compute_lyapunov_exponent(
                    attractor,
                    steps=int(config.tmax*100),
                    dt=0.01,
                    transient=0,
                    initial_conditions=(0.1, 0.1, 0.1),
                    attractor_params={'sigma': 10, 'beta': 2.667, 'rho': 28},
                    qr_interval=5
                )
            elif attractor == 'rossler':
                lyapunov_exponents = compute_lyapunov_exponent(
                    attractor,
                    steps=int(config.tmax*100),
                    dt=0.01,
                    transient=0,
                    initial_conditions=(0.1, 0.1, 0.1),
                    attractor_params={'a': 0.2, 'b': 0.2, 'c': 5.7},
                    qr_interval=5
                )
            elif attractor == 'aizawa':
                lyapunov_exponents = compute_lyapunov_exponent(
                    attractor,
                    steps=int(config.tmax*100),
                    dt=0.01,
                    transient=0,
                    initial_conditions=(0.1, 0.1, 0.1),
                    qr_interval=5
                )
            elif attractor == 'nose_hoover':
                lyapunov_exponents = compute_lyapunov_exponent(
                    attractor,
                    steps=int(config.tmax*100),
                    dt=0.01,
                    transient=0,
                    initial_conditions=(0.1, 0.1, 0.1),
                    attractor_params={'alpha': 1.0},
                    qr_interval=5
                )
            elif attractor == 'sprott':
                lyapunov_exponents = compute_lyapunov_exponent(
                    attractor,
                    steps=int(config.tmax*100),
                    dt=0.01,
                    transient=0,
                    initial_conditions=(0.1, 0.1, 0.1),
                    attractor_params={'a': 3.0},
                    qr_interval=5
                )
            elif attractor == 'chua':
                lyapunov_exponents = compute_lyapunov_exponent(
                    attractor,
                    steps=int(config.tmax*100),
                    dt=0.01,
                    transient=0,
                    initial_conditions=(0.1, 0.1, 0.1),
                    attractor_params={'alpha': 15.6, 'beta': 28.58, 'gamma': 0.0},
                    qr_interval=5
                )

            results['attractor'].append(attractor)
            results['largest_lyapunov'].append(lyapunov_exponents[0])
            results['all_lyapunov'].append(lyapunov_exponents)
            results['lyapunov_sum'].append(np.sum(lyapunov_exponents))

            print(f"Lyapunov exponents: {lyapunov_exponents}")
            print(f"Largest Lyapunov exponent: {lyapunov_exponents[0]}")


        except Exception as e:
            print(f"Error processing {attractor}: {e}")
            # Add placeholder to maintain consistency
            results['attractor'].append(attractor)
            results['largest_lyapunov'].append(np.nan)
            results['all_lyapunov'].append([np.nan, np.nan, np.nan])
            results['lyapunov_sum'].append(np.nan)


    # Convert to DataFrame
    df = pd.DataFrame({
        'attractor': results['attractor'],
        'largest_lyapunov': results['largest_lyapunov'],
        'lyapunov_sum': results['lyapunov_sum'],
    })

    # Store additional columns as strings, as DataFrame cannot directly store arrays
    df['all_lyapunov'] = [', '.join(map(str, lya)) for lya in results['all_lyapunov']]

    # Save as CSV
    csv_path = os.path.join(results_dir, 'lyapunov_exponents.csv')
    df.to_csv(csv_path, index=False)
    print(f"Lyapunov exponent results saved to {csv_path}")

    # Visualize results
    visualize_lyapunov_exponents(df, results_dir)

    return df


def visualize_lyapunov_exponents(df, results_dir):
    """
    Visualize Lyapunov exponents for different attractors

    Parameters:
        df: DataFrame containing Lyapunov exponents
        results_dir: Directory to save results
    """
    # Replace NaN values with 0 for visualization
    df_clean = df.fillna(0)

    # 1. Compare largest Lyapunov exponents across attractors
    plt.figure(figsize=(12, 8))
    bars = plt.bar(df_clean['attractor'], df_clean['largest_lyapunov'])

    # Set color based on value
    for i, bar in enumerate(bars):
        if df_clean['largest_lyapunov'].iloc[i] > 0:
            bar.set_color('red')  # Positive values indicate chaos
        else:
            bar.set_color('blue')  # Negative or zero values indicate non-chaos

    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Attractor Type')
    plt.ylabel('Largest Lyapunov Exponent')
    plt.title('Comparison of Largest Lyapunov Exponents Across Attractors')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', alpha=0.3)

    # Add value labels
    for i, value in enumerate(df_clean['largest_lyapunov']):
        plt.text(i, value + 0.01 * np.sign(value),
                 f"{value:.4f}", ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'largest_lyapunov.png'), dpi=300)
    plt.close()


def compare_lyapunov_with_encoding_performance(config, results_dir='lyapunov_comparison'):
    """
    Compare the relationship between Lyapunov exponents and encoding performance

    Parameters:
        config: Configuration object
        results_dir: Directory to save results
    """
    os.makedirs(results_dir, exist_ok=True)

    # Load Lyapunov exponent data
    lyapunov_csv = os.path.join('lyapunov_results', 'lyapunov_exponents.csv')

    if not os.path.exists(lyapunov_csv):
        print("Lyapunov exponent data not found, running analysis...")
        lyapunov_df = analyze_lyapunov_exponents(config)
    else:
        lyapunov_df = pd.read_csv(lyapunov_csv)

    # Load encoding performance data
    performance_csv = os.path.join('statistical_results_attractors', 'final_results', 'detailed_results.csv')

    if not os.path.exists(performance_csv):
        print("Encoding performance data not found, please run encoding performance experiments first")
        return

    performance_df = pd.read_csv(performance_csv)

    # Extract SNN model data
    snn_perf = performance_df[performance_df['Model'] == 'SNN']

    # Extract attractor type from experiment name
    snn_perf['attractor'] = snn_perf['Experiment'].str.split('-').str[0].str.lower()

    # Group by attractor type and calculate averages
    avg_perf = snn_perf.groupby('attractor').agg({
        'Best Accuracy': 'mean',
        'Convergence Epoch': 'mean'
    }).reset_index()

    # Merge Lyapunov exponents with performance data
    merged_df = pd.merge(lyapunov_df, avg_perf, on='attractor', how='inner')

    if merged_df.empty:
        print("No matching data after merge, please check attractor names")
        return

    # Save merged data
    merged_csv = os.path.join(results_dir, 'lyapunov_vs_performance.csv')
    merged_df.to_csv(merged_csv, index=False)
    print(f"Lyapunov vs performance comparison data saved to {merged_csv}")

    # Visualize comparisons
    plt.figure(figsize=(12, 10))

    # 1. Largest Lyapunov exponent vs classification accuracy
    plt.subplot(2, 1, 1)
    plt.scatter(merged_df['largest_lyapunov'], merged_df['Best Accuracy'],
                s=100, alpha=0.7)

    # Add trend line

    if len(merged_df) > 1:
        try:
            z = np.polyfit(merged_df['largest_lyapunov'], merged_df['Best Accuracy'], 1)
            p = np.poly1d(z)

            x_trend = np.linspace(merged_df['largest_lyapunov'].min(), merged_df['largest_lyapunov'].max(), 100)
            plt.plot(x_trend, p(x_trend), "r--", alpha=0.7)
        except:
            print('no trend!')

        # Calculate correlation coefficient
        corr = np.corrcoef(merged_df['largest_lyapunov'], merged_df['Best Accuracy'])[0, 1]
        plt.text(0.05, 0.95, f"Correlation: {corr:.2f}", transform=plt.gca().transAxes)


    # Add labels
    for i, txt in enumerate(merged_df['attractor']):
        plt.annotate(txt, (merged_df['largest_lyapunov'].iloc[i], merged_df['Best Accuracy'].iloc[i]),
                     xytext=(5, 5), textcoords='offset points')

    plt.xlabel('Largest Lyapunov Exponent')
    plt.ylabel('Average Classification Accuracy (%)')
    plt.title('Relationship Between Lyapunov Exponents and Classification Accuracy')
    plt.grid(True, alpha=0.3)

    # 2. Largest Lyapunov exponent vs convergence speed
    plt.subplot(2, 1, 2)
    plt.scatter(merged_df['largest_lyapunov'], merged_df['Convergence Epoch'],
                s=100, alpha=0.7)


    # Add trend line
    if len(merged_df) > 1:
        try:
            z = np.polyfit(merged_df['largest_lyapunov'], merged_df['Convergence Epoch'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(merged_df['largest_lyapunov'].min(), merged_df['largest_lyapunov'].max(), 100)
            plt.plot(x_trend, p(x_trend), "r--", alpha=0.7)
        except:
            print('no trend!')
        # Calculate correlation coefficient
        corr = np.corrcoef(merged_df['largest_lyapunov'], merged_df['Convergence Epoch'])[0, 1]
        plt.text(0.05, 0.95, f"Correlation: {corr:.2f}", transform=plt.gca().transAxes)

    # Add labels
    for i, txt in enumerate(merged_df['attractor']):
        plt.annotate(txt, (merged_df['largest_lyapunov'].iloc[i], merged_df['Convergence Epoch'].iloc[i]),
                     xytext=(5, 5), textcoords='offset points')


    plt.xlabel('Largest Lyapunov Exponent')
    plt.ylabel('Average Convergence Epoch')
    plt.title('Relationship Between Lyapunov Exponents and Convergence Speed')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'lyapunov_vs_performance.png'), dpi=300)
    plt.close()

    def calculate_active_info_storage(time_series, embedding_dim=2, tau=1, bins=10):
        """
        计算时间序列的主动信息存储 (更加稳健的版本)

        参数:
        - time_series: 一维时间序列
        - embedding_dim: 嵌入维度
        - tau: 时间延迟
        - bins: 概率分布估计的bin数量

        返回:
        - ais: 主动信息存储值
        """
        try:
            # 标准化数据
            ts = time_series.copy()
            if np.std(ts) > 1e-10:
                ts = (ts - np.mean(ts)) / np.std(ts)

            # 检查是否有无限值或NaN
            if np.any(np.isnan(ts)) or np.any(np.isinf(ts)):
                return np.nan

            # 为0和1阶嵌入创建时间序列
            x_t = ts[(embedding_dim * tau):]  # 当前状态

            # 创建过去状态矩阵
            x_past = np.zeros((len(x_t), embedding_dim))
            for i in range(embedding_dim):
                x_past[:, i] = ts[(embedding_dim - i - 1) * tau: len(ts) - i * tau - 1]

            # 限制数据范围，防止极端值
            x_t = np.clip(x_t, -5, 5)
            x_past = np.clip(x_past, -5, 5)

            # 计算联合概率和边缘概率
            # 添加一个小的平滑因子
            epsilon = 1e-10

            # 创建联合直方图 - 当前与过去
            h_joint, edges = np.histogramdd(
                np.column_stack([x_t, x_past]),
                bins=[bins] + [bins] * embedding_dim
            )

            # 归一化为概率分布
            p_joint = h_joint / np.sum(h_joint) + epsilon

            # 计算边缘概率
            # 当前状态概率
            h_curr, _ = np.histogram(x_t, bins=edges[0])
            p_curr = h_curr / np.sum(h_curr) + epsilon

            # 过去状态概率
            h_past = np.histogramdd(x_past, bins=edges[1:])[0]
            p_past = h_past / np.sum(h_past) + epsilon

            # 计算互信息
            mutual_info = 0
            it = np.nditer(p_joint, flags=['multi_index'])
            while not it.finished:
                idx = it.multi_index
                p_xy = p_joint[idx]
                if p_xy > epsilon:
                    p_x = p_curr[idx[0]]
                    past_idx = idx[1:]
                    p_y = p_past[past_idx]
                    mutual_info += p_xy * np.log2(p_xy / (p_x * p_y))
                it.iternext()

            return max(0, mutual_info)  # 确保不为负值

        except Exception as e:
            print(f"  AIS计算错误: {str(e)}")
            return np.nan


def calculate_active_info_storage(time_series, embedding_dim=2, tau=1, bins=10):
    """
    Calculates Active Information Storage for a time series (more robust version).

    Parameters:
    - time_series: A 1D time series.
    - embedding_dim: The embedding dimension.
    - tau: The time delay.
    - bins: The number of bins for probability distribution estimation.

    Returns:
    - ais: The Active Information Storage value.
    """
    try:
        # Standardize the data
        ts = time_series.copy()
        if np.std(ts) > 1e-10:
            ts = (ts - np.mean(ts)) / np.std(ts)

        # Check for infinite or NaN values
        if np.any(np.isnan(ts)) or np.any(np.isinf(ts)):
            return np.nan

        # Create time series for 0th and 1st order embedding
        x_t = ts[(embedding_dim * tau):]  # Current state

        # Create the past state matrix
        x_past = np.zeros((len(x_t), embedding_dim))
        for i in range(embedding_dim):
            x_past[:, i] = ts[(embedding_dim - i - 1) * tau: len(ts) - i * tau - 1]

        # Clip data range to prevent extreme values
        x_t = np.clip(x_t, -5, 5)
        x_past = np.clip(x_past, -5, 5)

        # Calculate joint and marginal probabilities
        # Add a small smoothing factor
        epsilon = 1e-10

        # Create joint histogram - current vs. past
        h_joint, edges = np.histogramdd(
            np.column_stack([x_t, x_past]),
            bins=[bins] + [bins] * embedding_dim
        )

        # Normalize to a probability distribution
        p_joint = h_joint / np.sum(h_joint) + epsilon

        # Calculate marginal probabilities
        # Current state probability
        h_curr, _ = np.histogram(x_t, bins=edges[0])
        p_curr = h_curr / np.sum(h_curr) + epsilon

        # Past state probability
        h_past = np.histogramdd(x_past, bins=edges[1:])[0]
        p_past = h_past / np.sum(h_past) + epsilon

        # Calculate mutual information
        mutual_info = 0
        it = np.nditer(p_joint, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            p_xy = p_joint[idx]
            if p_xy > epsilon:
                p_x = p_curr[idx[0]]
                past_idx = idx[1:]
                p_y = p_past[past_idx]
                mutual_info += p_xy * np.log2(p_xy / (p_x * p_y))
            it.iternext()

        return max(0, mutual_info)  # Ensure the value is not negative

    except Exception as e:
        print(f"  AIS calculation error: {str(e)}")
        return np.nan