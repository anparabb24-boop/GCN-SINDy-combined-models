import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# ==========================================
# 1. PARAMETERS & GROUND TRUTH SYSTEM (Appendix A & C)
# ==========================================
# Synchronous Generator Parameters
Omega_b = 2.0 * np.pi * 50.0  # Base frequency (rad/s), 314.159
omega_0 = 1.0                # Nominal speed (pu)
H2 = 12.467                  # Inertia constant 2H (MWs/MVA)
D = 1.9                      # Damping factor (pu)
Td0_prime = 6.5              # d-axis transient time constant (s)
Tq0_prime = 0.2              # q-axis transient time constant (s)
xd = 0.655                   # d-axis synchronous reactance (pu)
xq = 0.487                   # q-axis synchronous reactance (pu)
xd_prime = 0.196             # d-axis transient reactance (pu)
xq_prime = 0.263             # q-axis transient reactance (pu)

# Inputs (Excitation and Mechanical Power)
vf_const = 1.4998
Pm_const = 1.6

def grid_voltage(t):
    """Simulates a 3-phase fault cleared after 250 ms."""
    if 0.5 <= t <= 0.75:
        return 0.3, 0.0  # Fault: low voltage V, angle theta=0
    else:
        # Transient recovery oscillation in terminal voltage V and angle theta
        V_t = 1.0 - 0.15 * np.exp(-0.8 * (t - 0.75)) * np.cos(2 * np.pi * 1.5 * t) if t > 0.75 else 1.0
        theta_t = 0.1 * np.exp(-0.8 * (t - 0.75)) * np.sin(2 * np.pi * 1.5 * t) if t > 0.75 else 0.0
        return V_t, theta_t

def sg_4th_order_dae(t, x):
    """4th-order SG differential equations (Appendix A)."""
    delta, omega, eq_prime, ed_prime = x
    V, theta = grid_voltage(t)
    
    # Currents id and iq
    id_curr = (eq_prime - V * np.cos(delta - theta)) / xd_prime
    iq_curr = (V * np.sin(delta - theta) - ed_prime) / xq_prime
    
    # Active and Reactive Power Output
    Pg = V * np.sin(delta - theta) * id_curr + V * np.cos(delta - theta) * iq_curr
    
    # Differential Equations
    ddelta = Omega_b * (omega - omega_0)
    domega = (1.0 / H2) * (Pm_const - Pg - D * (omega - omega_0))
    deq_prime = (1.0 / Td0_prime) * (vf_const - eq_prime - (xd - xd_prime) * id_curr)
    ded_prime = (1.0 / Tq0_prime) * (-ed_prime + (xq - xq_prime) * iq_curr)
    
    return [ddelta, domega, deq_prime, ded_prime]

# ==========================================
# 2. DATASET GENERATION & NOISE ADDITION
# ==========================================
t_span = (0.0, 12.0)
dt = 0.01
t_eval = np.arange(t_span[0], t_span[1], dt)
x0 = [0.8, 1.0, 1.05, 0.2]  # Initial conditions [delta, omega, eq_prime, ed_prime]

# Numerical integration
sol = solve_ivp(sg_4th_order_dae, t_span, x0, t_eval=t_eval, method='RK45')
delta_clean = sol.y[0]
omega_clean = sol.y[1]
eq_p_clean = sol.y[2]
ed_p_clean = sol.y[3]

# Calculate Grid profiles and Power outputs
V_list, theta_list, Pg_clean, Qg_clean = [], [], [], []
for i, t in enumerate(t_eval):
    V, th = grid_voltage(t)
    V_list.append(V)
    theta_list.append(th)
    
    id_c = (eq_p_clean[i] - V * np.cos(delta_clean[i] - th)) / xd_prime
    iq_c = (V * np.sin(delta_clean[i] - th) - ed_p_clean[i]) / xq_prime
    
    Pg_clean.append(V * np.sin(delta_clean[i] - th) * id_c + V * np.cos(delta_clean[i] - th) * iq_c)
    Qg_clean.append(V * np.cos(delta_clean[i] - th) * id_c - V * np.sin(delta_clean[i] - th) * iq_c)

V_arr = np.array(V_list)
theta_arr = np.array(theta_list)
Pg_clean = np.array(Pg_clean)
Qg_clean = np.array(Qg_clean)

# Add Gaussian Noise (~2% SNR)
noise_std = 0.005
delta_meas = delta_clean + np.random.normal(0, noise_std, size=delta_clean.shape)
omega_meas = omega_clean + np.random.normal(0, noise_std * 0.1, size=omega_clean.shape)
eq_p_meas  = eq_p_clean  + np.random.normal(0, noise_std, size=eq_p_clean.shape)
ed_p_meas  = ed_p_clean  + np.random.normal(0, noise_std * 0.5, size=ed_p_clean.shape)
Pg_meas    = Pg_clean    + np.random.normal(0, noise_std * 2, size=Pg_clean.shape)
Qg_meas    = Qg_clean    + np.random.normal(0, noise_std * 2, size=Qg_clean.shape)

# ==========================================
# 3. SINDy CANDIDATE LIBRARY & DERIVATIVES
# ==========================================
def compute_4th_order_derivative(X, dt):
    """4th-order central difference scheme (Eq. 8)[cite: 1]."""
    dX = np.zeros_like(X)
    N = len(X)
    for i in range(2, N - 2):
        dX[i] = (-X[i+2] + 8*X[i+1] - 8*X[i-1] + X[i-2]) / (12.0 * dt)
    # Boundary points fallback to forward/backward differences
    dX[0] = (X[1] - X[0]) / dt
    dX[1] = (X[2] - X[1]) / dt
    dX[-2] = (X[-1] - X[-2]) / dt
    dX[-1] = (X[-1] - X[-2]) / dt
    return dX

# Calculate numerical time-derivatives
ddelta_dt = compute_4th_order_derivative(delta_meas, dt)
domega_dt = compute_4th_order_derivative(omega_meas, dt)
deq_p_dt  = compute_4th_order_derivative(eq_p_meas, dt)
ded_p_dt  = compute_4th_order_derivative(ed_p_meas, dt)

def build_sindy_library(delta, omega, eq_p, ed_p, V, theta):
    """
    Builds extended candidate library Theta(X, Z, Y) with trigonometric 
    and cross-product dynamic elements (Section III-B & Appendix B)[cite: 1].
    """
    N = len(delta)
    d_th = delta - theta
    
    # Candidate Functions[cite: 1]
    lib = [
        np.ones(N),                     # 0: Const
        delta,                         # 1: delta
        omega,                         # 2: omega
        eq_p,                          # 3: eq_prime
        ed_p,                          # 4: ed_prime
        V * np.cos(d_th),              # 5: V*cos(delta - theta)
        V * np.sin(d_th),              # 6: V*sin(delta - theta)
        V * eq_p * np.sin(d_th),       # 7: V*eq_p*sin(delta - theta)
        V * ed_p * np.cos(d_th),       # 8: V*ed_p*cos(delta - theta)
        (V**2) * np.sin(d_th) * np.cos(d_th) # 9: V^2*sin(delta - theta)*cos(delta - theta)
    ]
    return np.column_stack(lib)

Theta = build_sindy_library(delta_meas, omega_meas, eq_p_meas, ed_p_meas, V_arr, theta_arr)

# ==========================================
# 4. BOUND-CONSTRAINED STLS SOLVER
# ==========================================
def stls_regression(Theta, Y_dot, threshold=0.01, max_iter=10):
    """Sequential Thresholded Least Squares (STLS) solver[cite: 1]."""
    # Least squares initial solve
    xi, _, _, _ = np.linalg.lstsq(Theta, Y_dot, rcond=None)
    
    for _ in range(max_iter):
        small_inds = np.abs(xi) < threshold
        xi[small_inds] = 0.0
        for col in range(Y_dot.shape[1] if Y_dot.ndim > 1 else 1):
            big_inds = ~small_inds if Y_dot.ndim == 1 else ~small_inds[:, col]
            if np.any(big_inds):
                if Y_dot.ndim == 1:
                    xi[big_inds], _, _, _ = np.linalg.lstsq(Theta[:, big_inds], Y_dot, rcond=None)
                else:
                    xi[big_inds, col], _, _, _ = np.linalg.lstsq(Theta[:, big_inds], Y_dot[:, col], rcond=None)
    return xi

# Target derivatives matrix
Y_dot = np.column_stack([ddelta_dt, domega_dt, deq_p_dt, ded_p_dt])

# Run SINDy model identification
Xi_identified = stls_regression(Theta, Y_dot, threshold=0.03)

print("--- Identified SINDy Coefficient Matrix (Xi) ---")
feature_names = ["Const", "delta", "omega", "eq_prime", "ed_prime", 
                 "V*cos", "V*sin", "V*eq_p*sin", "V*ed_p*cos", "V^2*sin*cos"]
for name, row in zip(feature_names, Xi_identified):
    print(f"{name:15s}: d_delta={row[0]:.3f}, d_omega={row[1]:.3f}, d_eq_p={row[2]:.3f}, d_ed_p={row[3]:.3f}")

# ==========================================
# 5. RECONSTRUCTED MODEL SIMULATION
# ==========================================
def identified_sg_system(t, x):
    """Discovered dynamic model system simulation."""
    delta, omega, eq_p, ed_p = x
    V, th = grid_voltage(t)
    
    # Evaluate SINDy Candidate Library vector
    d_th = delta - th
    theta_vec = np.array([
        1.0, delta, omega, eq_p, ed_p,
        V * np.cos(d_th),
        V * np.sin(d_th),
        V * eq_p * np.sin(d_th),
        V * ed_p * np.cos(d_th),
        (V**2) * np.sin(d_th) * np.cos(d_th)
    ])
    
    d_state = theta_vec @ Xi_identified
    return d_state

sol_ident = solve_ivp(identified_sg_system, t_span, x0, t_eval=t_eval, method='RK45')
delta_id = sol_ident.y[0]
omega_id = sol_ident.y[1]
eq_p_id  = sol_ident.y[2]
ed_p_id  = sol_ident.y[3]

# Compute Identified Active and Reactive Power
Pg_id, Qg_id = [], []
for i, t in enumerate(t_eval):
    V, th = grid_voltage(t)
    id_i = (eq_p_id[i] - V * np.cos(delta_id[i] - th)) / xd_prime
    iq_i = (V * np.sin(delta_id[i] - th) - ed_p_id[i]) / xq_prime
    
    Pg_id.append(V * np.sin(delta_id[i] - th) * id_i + V * np.cos(delta_id[i] - th) * iq_i)
    Qg_id.append(V * np.cos(delta_id[i] - th) * id_i - V * np.sin(delta_id[i] - th) * iq_i)

Pg_id = np.array(Pg_id)
Qg_id = np.array(Qg_id)

# Residual/Error calculation wrt time[cite: 1]
delta_err = delta_meas - delta_id
omega_err = omega_meas - omega_id
ed_p_err  = ed_p_meas - ed_p_id
eq_p_err  = eq_p_meas - eq_p_id
Pg_err    = Pg_meas - Pg_id
Qg_err    = Qg_meas - Qg_id

# ==========================================
# 6. VISUALIZATION (Matching Fig. 2 Layout)
# ==========================================
fig, axes = plt.subplots(6, 2, figsize=(14, 16), sharex=True)

# Helper function to plot measurement vs identification and error residual[cite: 1]
def plot_signal_and_error(ax_top, ax_bot, t, meas, ident, err, title, ylabel):
    # Top Panel: Measurement vs Identification
    ax_top.plot(t, meas, color='navy', label='Measurement', linewidth=1.2)
    ax_top.plot(t, ident, color='limegreen', label='Identification', linewidth=1.2)
    ax_top.set_ylabel(f'{ylabel} [pu]')
    ax_top.set_title(title, fontsize=10, fontweight='bold')
    ax_top.grid(True, linestyle='--', alpha=0.6)
    ax_top.legend(loc='upper right')
    
    # Bottom Panel: Residual Error wrt Time
    ax_bot.plot(t, err, color='firebrick', linewidth=1.0)
    ax_bot.set_ylabel(f'({ylabel})$^{{dif}}$ [pu]')
    ax_bot.grid(True, linestyle='--', alpha=0.6)

# a. Rotor Angle (delta)
plot_signal_and_error(axes[0, 0], axes[1, 0], t_eval, delta_meas, delta_id, delta_err,
                      'a. Rotor angle ($\delta$)', '$\delta$')

# b. Rotor Speed (omega)
plot_signal_and_error(axes[0, 1], axes[1, 1], t_eval, omega_meas, omega_id, omega_err,
                      'b. Rotor speed ($\omega$)', '$\omega$')

# c. Transient Voltage in d-axis (ed_prime)
plot_signal_and_error(axes[2, 0], axes[3, 0], t_eval, ed_p_meas, ed_p_id, ed_p_err,
                      'c. Transient voltage in d-axis ($e_d\'$)', '$e_d\'$')

# d. Transient Voltage in q-axis (eq_prime)
plot_signal_and_error(axes[2, 1], axes[3, 1], t_eval, eq_p_meas, eq_p_id, eq_p_err,
                      'd. Transient voltage in q-axis ($e_q\'$)', '$e_q\'$')

# e. Active Power (Pg)
plot_signal_and_error(axes[4, 0], axes[5, 0], t_eval, Pg_meas, Pg_id, Pg_err,
                      'e. Active power ($P_g$)', '$P_g$')
axes[5, 0].set_xlabel('Time [s]')

# f. Reactive Power (Qg)
plot_signal_and_error(axes[4, 1], axes[5, 1], t_eval, Qg_meas, Qg_id, Qg_err,
                      'f. Reactive power ($Q_g$)', '$Q_g$')
axes[5, 1].set_xlabel('Time [s]')

plt.tight_layout()
plt.savefig('sindy_mbam_sg_identification.png', dpi=300)
plt.show()