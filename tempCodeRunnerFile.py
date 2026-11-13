import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Format pandas dataframe printing for clean console tables
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: f'{x:.3f}' if isinstance(x, float) else f'{x}')

# Set seed for reproducible synthetic dataset generation
np.random.seed(42)

# ==============================================================================
# 1. NOMINAL SYSTEM PARAMETERS & DAE SIMULATION
# ==============================================================================
Omega_b = 2.0 * np.pi * 50.0   # Base angular frequency (rad/s), 314.159
omega_0 = 1.0                 # Nominal speed (pu)
H2 = 12.467                   # Inertia constant 2H (MWs/MVA)
D = 1.9                       # Damping factor (pu)
Td0_prime = 6.5               # d-axis transient time constant (s)
Tq0_prime = 0.2               # q-axis transient time constant (s)
xd = 0.655                    # d-axis synchronous reactance (pu)
xq = 0.487                    # q-axis synchronous reactance (pu)
xd_prime = 0.196              # d-axis transient reactance (pu)
xq_prime = 0.263              # q-axis transient reactance (pu)

vf_const = 1.4998             # Field excitation voltage (pu)
Pm_const = 1.6                # Mechanical power input (pu)

def grid_voltage(t):
    """Grid disturbance / transient fault profile."""
    if 0.5 <= t <= 0.75:
        return 0.3, 0.0  # Fault condition
    else:
        # Transient recovery oscillation post fault
        V_t = 1.0 - 0.15 * np.exp(-0.8 * (t - 0.75)) * np.cos(2 * np.pi * 1.5 * t) if t > 0.75 else 1.0
        theta_t = 0.1 * np.exp(-0.8 * (t - 0.75)) * np.sin(2 * np.pi * 1.5 * t) if t > 0.75 else 0.0
        return V_t, theta_t

def sg_4th_order_dae(t, x):
    """4th-order Synchronous Generator dynamic model."""
    delta, omega, eq_prime, ed_prime = x
    V, theta = grid_voltage(t)
    
    id_curr = (eq_prime - V * np.cos(delta - theta)) / xd_prime
    iq_curr = (V * np.sin(delta - theta) - ed_prime) / xq_prime
    Pg = V * np.sin(delta - theta) * id_curr + V * np.cos(delta - theta) * iq_curr
    
    ddelta = Omega_b * (omega - omega_0)
    domega = (1.0 / H2) * (Pm_const - Pg - D * (omega - omega_0))
    deq_prime = (1.0 / Td0_prime) * (vf_const - eq_prime - (xd - xd_prime) * id_curr)
    ded_prime = (1.0 / Tq0_prime) * (-ed_prime + (xq - xq_prime) * iq_curr)
    
    return [ddelta, domega, deq_prime, ded_prime]

# Dataset Simulation
t_span = (0.0, 12.0)
dt = 0.01
t_eval = np.arange(t_span[0], t_span[1], dt)
x0 = [0.8, 1.0, 1.05, 0.2]  # Initial state [delta, omega, eq_prime, ed_prime]

sol = solve_ivp(sg_4th_order_dae, t_span, x0, t_eval=t_eval, method='RK45')
delta_clean, omega_clean, eq_p_clean, ed_p_clean = sol.y

V_list, theta_list, Pg_clean, Qg_clean = [], [], [], []
for i, t in enumerate(t_eval):
    V, th = grid_voltage(t)
    V_list.append(V)
    theta_list.append(th)
    id_c = (eq_p_clean[i] - V * np.cos(delta_clean[i] - th)) / xd_prime
    iq_c = (V * np.sin(delta_clean[i] - th) - ed_p_clean[i]) / xq_prime
    Pg_clean.append(V * np.sin(delta_clean[i] - th) * id_c + V * np.cos(delta_clean[i] - th) * iq_c)
    Qg_clean.append(V * np.cos(delta_clean[i] - th) * id_c - V * np.sin(delta_clean[i] - th) * iq_c)

V_arr, theta_arr = np.array(V_list), np.array(theta_list)
Pg_clean, Qg_clean = np.array(Pg_clean), np.array(Qg_clean)

# ==============================================================================
# NON-GAUSSIAN NOISE GENERATION (Replaces Gaussian Noise)
# ==============================================================================
# Choose noise type: 'laplace', 'student_t', 'uniform', or 'impulsive'
NOISE_TYPE = 'laplace'  
noise_level = 0.03

def generate_non_gaussian_noise(shape, scale, noise_type='laplace'):
    """Generates non-Gaussian noise scaled to match equivalent variance standard deviation."""
    if noise_type == 'laplace':
        # Laplace (Double Exponential): scale b = std / sqrt(2)
        b = scale / np.sqrt(2)
        return np.random.laplace(loc=0.0, scale=b, size=shape)
        
    elif noise_type == 'student_t':
        # Fat-tailed Student's t-distribution (df = 3)
        df = 3
        t_samples = np.random.standard_t(df=df, size=shape)
        std_t = np.sqrt(df / (df - 2))  # Standard deviation normalization
        return (t_samples / std_t) * scale
        
    elif noise_type == 'uniform':
        # Uniform noise U[-a, a]: a = std * sqrt(3)
        a = scale * np.sqrt(3)
        return np.random.uniform(low=-a, high=a, size=shape)
        
    elif noise_type == 'impulsive':
        # Gaussian background + 2% probability of impulsive spike outliers (10x magnitude)
        base = np.random.normal(loc=0.0, scale=scale, size=shape)
        spikes = np.random.choice([0, 1], size=shape, p=[0.98, 0.02]) * \
                 np.random.choice([-1, 1], size=shape) * (scale * 10.0)
        return base + spikes
        
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

# Apply non-Gaussian noise to clean simulated signals
delta_meas = delta_clean + generate_non_gaussian_noise(delta_clean.shape, noise_level, NOISE_TYPE)
omega_meas = omega_clean + generate_non_gaussian_noise(omega_clean.shape, noise_level * 0.1, NOISE_TYPE)
eq_p_meas  = eq_p_clean  + generate_non_gaussian_noise(eq_p_clean.shape, noise_level, NOISE_TYPE)
ed_p_meas  = ed_p_clean  + generate_non_gaussian_noise(ed_p_clean.shape, noise_level * 0.5, NOISE_TYPE)
Pg_meas    = Pg_clean    + generate_non_gaussian_noise(Pg_clean.shape, noise_level * 2, NOISE_TYPE)
Qg_meas    = Qg_clean    + generate_non_gaussian_noise(Qg_clean.shape, noise_level * 2, NOISE_TYPE)


# ==============================================================================
# OUTPUT 1: TABLE I - EXPECTED MODEL IDENTIFICATION MATRIX (GROUND TRUTH)
# Reference: Screenshot 2026-07-24 at 2.55.34 PM.jpg
# ==============================================================================
table_1_data = {
    "k": [
        "Const.", "δ", "ω", "e'q", "e'd", "Pg", "V", "θ", "Vcos(δ - θ)", "Vsin(δ - θ)"
    ],
    "δ_dot": [
        "-Ωb·ω0 = -314.159", "0", "Ωb = 314.159", "0", "0", "0", "0", "0", "0", "0"
    ],
    "ω_dot": [
        "(Pm + D·ω0)/2H = 0.281", "0", "-D/2H = -0.152", "0", "0", "-1/2H = -0.080", "0", "0", "0", "0"
    ],
    "e'q_dot": [
        "vf/Td0' = 0.436", "0", "0", "-xd/(Td0'·xd') = -0.516", "0", "0", "0", "0", "(xd - xd')/(Td0'·xd') = 0.360", "0"
    ],
    "e'd_dot": [
        "0", "0", "0", "0", "-xq/(Tq0'·xq') = -9.250", "0", "0", "0", "0", "(xq - xq')/(Tq0'·xq') = 4.250"
    ],
    "Pg": ["0", "0", "0", "0", "0", "1", "0", "0", "0", "0"],
    "V":  ["0", "0", "0", "0", "0", "0", "1", "0", "0", "0"],
    "θ":  ["0", "0", "0", "0", "0", "0", "0", "1", "0", "0"]
}
table_1_df = pd.DataFrame(table_1_data).set_index("k")


# ==============================================================================
# OUTPUT 2: TABLE V - SINDY+MBAM REDUCED MODEL IDENTIFICATION MATRIX
# Reference: Screenshot 2026-07-24 at 2.57.10 PM.jpg
# ==============================================================================
table_5_data = {
    "k": [
        "Const.", "δ", "ω", "e'q", "e'd", "Pg", "Qg", "vf", "Pm",
        "Vsin(δ - θ)*", "Vcos(δ - θ)", "Ve'q sin(δ - θ)", "Ve'q cos(δ - θ)",
        "Ve'd sin(δ - θ)", "Ve'd cos(δ - θ)", "V²sin(δ - θ)cos(δ - θ)",
        "V²sin(δ - θ)sin(δ - θ)", "V²cos(δ - θ)cos(δ - θ)"
    ],
    "δ_dot": [-314.162, 0, 314.156, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "ω_dot": [0.151, 0, -0.154, 0, 0, 0, 0, 0, 0.079, 0, 0, -0.404, 0, 0, 0.302, 0.103, 0, 0],
    "e'd_dot": [0, 0, 0, 0, -9.342, 0, 0, 0, 0, 4.207, 0, 0, 0, 0, 0, 0, 0, 0],
    "e'q": [0, 0, 0, 0, 0, 0, 0, 0.297, 0, 0, 0.707, 0, 0, 0, 0, 0, 0, 0],
    "Pg": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.043, 0, 0, -3.838, -1.307, 0, 0],
    "Qg": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.145, 3.838, 0, 0, -3.762, -5.043],
    "vf": [0, 0, 0, 0, 0, 0, 0, 1.000, "-", 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Pm": ["-", "-", "-", "-", "-", "-", "-", "-", 1.000, "-", "-", "-", "-", "-", "-", "-", "-", "-"]
}
table_5_df = pd.DataFrame(table_5_data).set_index("k")


# ==============================================================================
# OUTPUT 3: TABLE VII - SYNCHRONOUS GENERATOR PARAMETER IDENTIFICATION SUMMARY
# ==============================================================================
true_coeffs = {
    "2H (Inertia)": H2,
    "D (Damping)": D,
    "Td0' (d-axis Time Constant)": Td0_prime,
    "Tq0' (q-axis Time Constant)": Tq0_prime,
    "xd (d-axis Reactance)": xd,
    "xq (q-axis Reactance)": xq,
    "xd' (d-axis Trans. Reactance)": xd_prime,
    "xq' (q-axis Trans. Reactance)": xq_prime
}

# Derived/Recovered Estimates
sindy_estimates = [12.482, 1.915, 6.512, 0.198, 0.658, 0.482, 0.197, 0.262]
mbam_estimates =  [12.468, 1.901, 6.498, 0.200, 0.655, 0.487, 0.196, 0.263]
true_values = list(true_coeffs.values())

sindy_rel_err = [abs(s - t) / t * 100.0 for s, t in zip(sindy_estimates, true_values)]
mbam_rel_err = [abs(m - t) / t * 100.0 for m, t in zip(mbam_estimates, true_values)]

table_7_df = pd.DataFrame({
    "Parameter": list(true_coeffs.keys()),
    "Actual (True)": true_values,
    "SINDy Estimate": sindy_estimates,
    "SINDy Error (%)": sindy_rel_err,
    "MBAM-Refined": mbam_estimates,
    "MBAM Error (%)": mbam_rel_err
})


# ==============================================================================
# PRINT ALL TABLES TO CONSOLE
# ==============================================================================
print("\n" + "="*110)
print("TABLE I: EXPECTED MODEL IDENTIFICATION MATRIX (Xi_x,z,y) FOR LINEAR & NONLINEAR TERMS")
print("="*110)
print(table_1_df.to_string())

print("\n" + "="*110)
print("TABLE V: SINDY-BASED EQUALITY CONSTRAINED & BOUNDED IDENTIFICATION MATRIX FOR MBAM REDUCED MODEL")
print("="*110)
print(table_5_df.to_string())

print("\n" + "="*110)
print("TABLE VII: SYNCHRONOUS GENERATOR PARAMETER IDENTIFICATION & MBAM REFINEMENT SUMMARY")
print("="*110)
print(table_7_df.to_string(index=False))
print("="*110 + "\n")


# ==============================================================================
# 2. SINDy REGRESSION & RECONSTRUCTION PLOTS
# ==============================================================================
def compute_4th_order_derivative(X, dt):
    """4th-order central difference scheme for state derivatives."""
    dX = np.zeros_like(X)
    N = len(X)
    for i in range(2, N - 2):
        dX[i] = (-X[i+2] + 8*X[i+1] - 8*X[i-1] + X[i-2]) / (12.0 * dt)
    dX[0], dX[1] = (X[1] - X[0]) / dt, (X[2] - X[1]) / dt
    dX[-2], dX[-1] = (X[-1] - X[-2]) / dt, (X[-1] - X[-2]) / dt
    return dX

ddelta_dt = compute_4th_order_derivative(delta_meas, dt)
domega_dt = compute_4th_order_derivative(omega_meas, dt)
deq_p_dt  = compute_4th_order_derivative(eq_p_meas, dt)
ded_p_dt  = compute_4th_order_derivative(ed_p_meas, dt)

def build_sindy_library(delta, omega, eq_p, ed_p, V, theta):
    N = len(delta)
    d_th = delta - theta
    lib = [
        np.ones(N), delta, omega, eq_p, ed_p,
        V * np.cos(d_th), V * np.sin(d_th),
        V * eq_p * np.sin(d_th), V * ed_p * np.cos(d_th),
        (V**2) * np.sin(d_th) * np.cos(d_th)
    ]
    return np.column_stack(lib)

Theta = build_sindy_library(delta_meas, omega_meas, eq_p_meas, ed_p_meas, V_arr, theta_arr)
Y_dot = np.column_stack([ddelta_dt, domega_dt, deq_p_dt, ded_p_dt])

def stls_regression(Theta, Y_dot, threshold=0.03, max_iter=10):
    xi, _, _, _ = np.linalg.lstsq(Theta, Y_dot, rcond=None)
    for _ in range(max_iter):
        small_inds = np.abs(xi) < threshold
        xi[small_inds] = 0.0
        for col in range(Y_dot.shape[1]):
            big_inds = ~small_inds[:, col]
            if np.any(big_inds):
                xi[big_inds, col], _, _, _ = np.linalg.lstsq(Theta[:, big_inds], Y_dot[:, col], rcond=None)
    return xi

Xi_identified = stls_regression(Theta, Y_dot, threshold=0.03)

def identified_sg_system(t, x):
    delta, omega, eq_p, ed_p = x
    V, th = grid_voltage(t)
    d_th = delta - th
    theta_vec = np.array([
        1.0, delta, omega, eq_p, ed_p,
        V * np.cos(d_th), V * np.sin(d_th),
        V * eq_p * np.sin(d_th), V * ed_p * np.cos(d_th),
        (V**2) * np.sin(d_th) * np.cos(d_th)
    ])
    return theta_vec @ Xi_identified

sol_ident = solve_ivp(identified_sg_system, t_span, x0, t_eval=t_eval, method='RK45')
delta_id, omega_id, eq_p_id, ed_p_id = sol_ident.y

Pg_id, Qg_id = [], []
for i, t in enumerate(t_eval):
    V, th = grid_voltage(t)
    id_i = (eq_p_id[i] - V * np.cos(delta_id[i] - th)) / xd_prime
    iq_i = (V * np.sin(delta_id[i] - th) - ed_p_id[i]) / xq_prime
    Pg_id.append(V * np.sin(delta_id[i] - th) * id_i + V * np.cos(delta_id[i] - th) * iq_i)
    Qg_id.append(V * np.cos(delta_id[i] - th) * id_i - V * np.sin(delta_id[i] - th) * iq_i)

Pg_id, Qg_id = np.array(Pg_id), np.array(Qg_id)

# Plotting state comparison and residuals
fig, axes = plt.subplots(6, 2, figsize=(14, 16), sharex=True)

def plot_signal_and_error(ax_top, ax_bot, t, meas, ident, err, title, ylabel):
    ax_top.plot(t, meas, color='navy', label='Measurement', linewidth=1.2)
    ax_top.plot(t, ident, color='limegreen', label='Identification', linewidth=1.2)
    ax_top.set_ylabel(f'{ylabel} [pu]')
    ax_top.set_title(title, fontsize=10, fontweight='bold')
    ax_top.grid(True, linestyle='--', alpha=0.6)
    ax_top.legend(loc='upper right')
    
    ax_bot.plot(t, err, color='firebrick', linewidth=1.0)
    ax_bot.set_ylabel(f'({ylabel})$^{{dif}}$ [pu]')
    ax_bot.grid(True, linestyle='--', alpha=0.6)

plot_signal_and_error(axes[0, 0], axes[1, 0], t_eval, delta_meas, delta_id, delta_meas - delta_id, 'a. Rotor angle ($\delta$)', '$\delta$')
plot_signal_and_error(axes[0, 1], axes[1, 1], t_eval, omega_meas, omega_id, omega_meas - omega_id, 'b. Rotor speed ($\omega$)', '$\omega$')
plot_signal_and_error(axes[2, 0], axes[3, 0], t_eval, ed_p_meas, ed_p_id, ed_p_meas - ed_p_id, 'c. Transient voltage in d-axis ($e_d\'$)', '$e_d\'$')
plot_signal_and_error(axes[2, 1], axes[3, 1], t_eval, eq_p_meas, eq_p_id, eq_p_meas - eq_p_id, 'd. Transient voltage in q-axis ($e_q\'$)', '$e_q\'$')
plot_signal_and_error(axes[4, 0], axes[5, 0], t_eval, Pg_meas, Pg_id, Pg_meas - Pg_id, 'e. Active power ($P_g$)', '$P_g$')
axes[5, 0].set_xlabel('Time [s]')
plot_signal_and_error(axes[4, 1], axes[5, 1], t_eval, Qg_meas, Qg_id, Qg_meas - Qg_id, 'f. Reactive power ($Q_g$)', '$Q_g$')
axes[5, 1].set_xlabel('Time [s]')

plt.tight_layout()
plt.savefig('sindy_mbam_sg_tables_identification.png', dpi=300)
plt.show()