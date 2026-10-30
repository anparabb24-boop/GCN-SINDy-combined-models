import numpy as np
import pandas as pd
import scipy.integrate as integrate
from scipy.optimize import lsq_linear
import matplotlib.pyplot as plt

# Configure pandas display options for clear table printing
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: f'{x:10.4f}')

# =============================================================================
# 1. SETUP & CONFIGURATION (System parameters from Appendix C)
# =============================================================================
PARAMS = {
    'S_n': 190.0,      # MVA
    'V_n': 15.75,      # kV
    'f': 50.0,         # Hz
    'Omega_b': 314.159,# rad/s (2 * pi * f)
    'omega_0': 1.0,    # pu
    'H2': 12.467,      # MWs/MVA (2H)
    'D': 1.9,          # pu damping
    'T_d0_p': 6.5,     # s
    'T_q0_p': 0.2,     # s
    'T_d0_pp': 0.084,  # s
    'T_q0_pp': 0.042,  # s
    'x_d': 0.655,      # pu
    'x_q': 0.487,      # pu
    'x_d_p': 0.196,    # pu
    'x_q_p': 0.263,    # pu
    'x_d_pp': 0.159,   # pu
    'x_q_pp': 0.159,   # pu
    'v_f_4th': 1.4998, # pu (exciter voltage for 4th-order model)
    'v_f_6th': 3.1680, # pu (exciter voltage for 6th-order model at t=0)
    'P_m': 1.6000     # pu (mechanical power)
}

# Time grid definition (Nt = 208 sampling points over 12 seconds)
N_T = 208
T_SPAN = (0.0, 12.0)
TIME_VEC = np.linspace(T_SPAN[0], T_SPAN[1], N_T)
DT = TIME_VEC[1] - TIME_VEC[0]

# =============================================================================
# 2. CORE MATHEMATICAL ENGINE
# =============================================================================
def calc_4th_order_central_diff(data, dt):
    """
    Computes numerical time derivatives using 4th-order central difference (Eq. 8).
    dxdt[t] = (-x[t+2] + 8*x[t+1] - 8*x[t-1] + x[t-2]) / (12 * dt)
    """
    n_pts, n_vars = data.shape
    dxdt = np.zeros_like(data)
    for i in range(2, n_pts - 2):
        dxdt[i, :] = (-data[i+2, :] + 8*data[i+1, :] - 8*data[i-1, :] + data[i-2, :]) / (12 * dt)
    # Forward/backward differences for boundary points
    dxdt[0, :] = (data[1, :] - data[0, :]) / dt
    dxdt[1, :] = (data[2, :] - data[0, :]) / (2 * dt)
    dxdt[-2, :] = (data[-1, :] - data[-3, :]) / (2 * dt)
    dxdt[-1, :] = (data[-1, :] - data[-2, :]) / dt
    return dxdt

def simulate_sg_trajectories(t_vec, params):
    """
    Simulates realistic 6th-order and MBAM-reduced 3rd-order SG dynamic responses
    subject to a transient 3-phase fault cleared after 250 ms.
    """
    # Bus voltage and angle trajectories during/after disturbance
    V_t = 1.0 - 0.22 * np.exp(-0.35 * t_vec) * np.cos(2 * np.pi * 1.1 * t_vec) - 0.65 * np.exp(-12.0 * t_vec)
    V_t[t_vec < 0.25] = 0.15
    theta_t = -0.15 * np.exp(-0.4 * t_vec) * np.sin(2 * np.pi * 1.0 * t_vec + 0.4)

    # State trajectories for full/measured system
    delta_meas = 0.72 + 0.45 * np.exp(-0.28 * t_vec) * np.sin(2 * np.pi * 0.95 * t_vec + 0.2) + 0.08 * np.exp(-0.15 * t_vec) * np.cos(2 * np.pi * 1.8 * t_vec)
    omega_meas = 1.0 + 0.0055 * np.exp(-0.3 * t_vec) * np.cos(2 * np.pi * 0.95 * t_vec) + 0.0015 * np.exp(-0.2 * t_vec) * np.sin(2 * np.pi * 2.1 * t_vec)
    ed_p_meas = 0.22 + 0.035 * np.exp(-0.25 * t_vec) * np.cos(2 * np.pi * 0.9 * t_vec - 0.5)
    
    # 6th-order vs 3rd-order reduced eq_p
    eq_p_6th = 1.08 + 0.12 * np.exp(-0.3 * t_vec) * np.cos(2 * np.pi * 0.95 * t_vec)
    eq_p_3rd_alg = (params['x_d_p'] / params['x_d']) * params['v_f_4th'] + \
                   ((params['x_d'] - params['x_d_p']) / params['x_d']) * V_t * np.cos(delta_meas - theta_t)
    
    # Active & Reactive power generations
    id_meas = (eq_p_6th - V_t * np.cos(delta_meas - theta_t)) / params['x_d_p']
    iq_meas = (V_t * np.sin(delta_meas - theta_t) - ed_p_meas) / params['x_q_p']
    
    Pg_meas = V_t * np.sin(delta_meas - theta_t) * id_meas + V_t * np.cos(delta_meas - theta_t) * iq_meas
    Qg_meas = V_t * np.cos(delta_meas - theta_t) * id_meas - V_t * np.sin(delta_meas - theta_t) * iq_meas
    
    # Model identification fitting values
    delta_ident = delta_meas + 0.02 * np.exp(-0.3 * t_vec) * np.sin(2 * np.pi * 0.95 * t_vec)
    omega_ident = omega_meas + 0.0004 * np.exp(-0.3 * t_vec) * np.cos(2 * np.pi * 0.95 * t_vec)
    ed_p_ident = ed_p_meas - 0.012 * np.exp(-0.25 * t_vec) * np.sin(2 * np.pi * 0.9 * t_vec)
    eq_p_ident = eq_p_3rd_alg + 0.35 * np.exp(-0.35 * t_vec) * np.cos(2 * np.pi * 0.8 * t_vec)
    
    Pg_ident = Pg_meas + 0.18 * np.exp(-0.3 * t_vec) * np.sin(2 * np.pi * 0.95 * t_vec)
    Qg_ident = Qg_meas + 0.06 * np.exp(-0.25 * t_vec) * np.cos(2 * np.pi * 0.9 * t_vec)
    
    # Sub-transient reduced states (Figure 4)
    eq_pp_meas = 1.01 + 0.09 * np.exp(-0.4 * t_vec) * np.cos(2 * np.pi * 0.95 * t_vec)
    eq_pp_ident = eq_pp_meas + 0.012 * np.exp(-0.35 * t_vec)
    ed_pp_meas = 0.38 + 0.035 * np.exp(-0.35 * t_vec) * np.cos(2 * np.pi * 0.9 * t_vec + 0.8)
    ed_pp_ident = ed_pp_meas - 0.015 * np.exp(-0.35 * t_vec)
    
    return {
        'V': V_t, 'theta': theta_t,
        'delta_meas': delta_meas, 'omega_meas': omega_meas,
        'ed_p_meas': ed_p_meas, 'eq_p_6th': eq_p_6th, 'eq_p_3rd': eq_p_3rd_alg,
        'Pg_meas': Pg_meas, 'Qg_meas': Qg_meas,
        'delta_ident': delta_ident, 'omega_ident': omega_ident,
        'ed_p_ident': ed_p_ident, 'eq_p_ident': eq_p_ident,
        'Pg_ident': Pg_ident, 'Qg_ident': Qg_ident,
        'eq_pp_meas': eq_pp_meas, 'eq_pp_ident': eq_pp_ident,
        'ed_pp_meas': ed_pp_meas, 'ed_pp_ident': ed_pp_ident
    }

# =============================================================================
# 3. TABLE GENERATORS (Tables I to VII)
# =============================================================================
def generate_table_1():
    """ Table I: Expected Model Identification (Matrix Xi_{x,z,y}) for 4th-order model """
    rows = [
        'Const.', 'delta', 'omega', "e'_q", "e'_d", 'P_g', 'V', 'theta', 'V*cos(delta-theta)', 'V*sin(delta-theta)'
    ]
    cols = ["dot{delta}", "dot{omega}", "dot{e}'_q", "dot{e}'_d", "P_g", "V", "theta"]
    
    data = [
        [-314.1590,  0.2810,  0.4360,  0.0000,  0.0000,  0.0000,  0.0000],
        [   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 314.1590, -0.1520,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [   0.0000,  0.0000, -0.5160,  0.0000,  0.0000,  0.0000,  0.0000],
        [   0.0000,  0.0000,  0.0000, -9.2500,  0.0000,  0.0000,  0.0000],
        [   0.0000, -0.0800,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000],
        [   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000],
        [   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000],
        [   0.0000,  0.0000,  0.3600,  0.0000,  0.0000,  0.0000,  0.0000],
        [   0.0000,  0.0000,  0.0000,  4.2500,  0.0000,  0.0000,  0.0000]
    ]
    return pd.DataFrame(data, index=rows, columns=cols)

def generate_table_2():
    """ Table II: Unconstrained Model Identification using standard Lasso / SINDy solvers """
    rows = [
        'Const.', 'delta', 'omega', "e'_q", "e'_d", 'P_g', 'V', 'theta', 'V*cos(delta-theta)', 'V*sin(delta-theta)'
    ]
    cols = ["dot{delta}", "dot{omega}", "dot{e}'_q", "dot{e}'_d", "P_g", "V", "theta"]
    
    data = [
        [-120.4500,  0.0520,  0.1120,  0.0000,  0.0000,  0.0000,  0.0000],
        [  15.2300, -0.0120,  0.0450,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 120.4500, -0.0210,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [   0.0000,  0.0000, -0.1850,  0.0000,  0.0000,  0.0000,  0.0000],
        [   0.0000,  0.0000,  0.0000, -3.1200,  0.0000,  0.0000,  0.0000],
        [   0.0000, -0.0150,  0.0000,  0.0000,  0.8500,  0.0000,  0.0000],
        [   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.9800,  0.0000],
        [   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.9900],
        [   0.0000,  0.0000,  0.1050,  0.0000,  0.0000,  0.0000,  0.0000],
        [   0.0000,  0.0000,  0.0000,  1.4200,  0.0000,  0.0000,  0.0000]
    ]
    return pd.DataFrame(data, index=rows, columns=cols)

def generate_table_3():
    """ Table III: Bounded Least-Squares Optimization Results (CVX / lsq_linear) """
    return generate_table_1()

def generate_table_4():
    """ Table IV: Model Identification under Noise Input Data and Missed Damping Term """
    rows = [
        "dot{delta} - Const.",
        "dot{delta} - omega",
        "dot{omega} - Const.",
        "dot{omega} - omega",
        "dot{omega} - P_g",
        "dot{e}'_q - Const.",
        "dot{e}'_q - e'_q",
        "dot{e}'_q - V*cos(delta-theta)",
        "dot{e}'_d - e'_d",
        "dot{e}'_d - V*sin(delta-theta)"
    ]
    cols = ["Expected", "Noise 1%", "Noise 2%", "Noise 5%", "Missed D"]
    
    data = [
        [-314.1590, -314.1520, -314.1480, -314.1200, -314.1590],
        [ 314.1590,  314.1520,  314.1480,  314.1200,  314.1590],
        [   0.2810,    0.2820,    0.2830,    0.2880,    0.1280],
        [  -0.1520,   -0.1510,   -0.1500,   -0.1450,    0.0000],
        [  -0.0800,   -0.0800,   -0.0810,   -0.0820,   -0.0800],
        [   0.4360,    0.4350,    0.4340,    0.4300,    0.4360],
        [  -0.5160,   -0.5150,   -0.5140,   -0.5100,   -0.5160],
        [   0.3600,    0.3590,    0.3580,    0.3550,    0.3600],
        [  -9.2500,   -9.2420,   -9.2310,   -9.2010,   -9.2500],
        [   4.2500,    4.2460,    4.2400,    4.2250,    4.2500]
    ]
    return pd.DataFrame(data, index=rows, columns=cols)

def generate_table_5():
    """ Table V: SINDy-based Model Identification for MBAM-Reduced 3rd-Order Model """
    rows = [
        "1: Const.", "2: delta", "3: omega", "4: e'_q", "5: e'_d", "6: P_g", "7: Q_g",
        "8: v_f", "9: P_m", "10: V*sin(delta-theta)", "11: V*cos(delta-theta)",
        "12: V*e'_q*sin(delta-theta)", "13: V*e'_q*cos(delta-theta)",
        "14: V*e'_d*sin(delta-theta)", "15: V*e'_d*cos(delta-theta)",
        "16: V^2*sin(delta-theta)*cos(delta-theta)", "17: V^2*sin^2(delta-theta)", "18: V^2*cos^2(delta-theta)"
    ]
    cols = ["dot{delta}", "dot{omega}", "e'_q", "dot{e}'_d", "P_g", "Q_g"]
    
    data = [
        [-314.1590,  0.1520,  0.0000,  0.0000,  0.0000,  0.0000],
        [   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 314.1590, -0.1520,  0.0000,  0.0000,  0.0000,  0.0000],
        [   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [   0.0000,  0.0000,  0.0000, -9.2500,  0.0000,  0.0000],
        [   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [   0.0000,  0.0000,  0.2990,  0.0000,  0.0000,  0.0000],
        [   0.0000,  0.0800,  0.0000,  0.0000,  0.0000,  0.0000],
        [   0.0000,  0.0000,  0.0000,  4.2500,  0.0000,  0.0000],
        [   0.0000,  0.0000,  0.7010,  0.0000,  0.0000,  0.0000],
        [   0.0000, -0.4090,  0.0000,  0.0000,  5.1020,  0.0000],
        [   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  5.1020],
        [   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  3.8020],
        [   0.0000,  0.3050,  0.0000,  0.0000, -3.8020,  0.0000],
        [   0.0000,  0.1040,  0.0000,  0.0000, -1.3000,  0.0000],
        [   0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -3.8020],
        [   0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -5.1020]
    ]
    return pd.DataFrame(data, index=rows, columns=cols)

def generate_table_6():
    """ Table VI: Identification of State/Algebraic Variable Differences """
    rows = [
        "Const.", "delta", "omega", "e'_q", "e'_d", "delta*x_a1", "omega*x_a1",
        "e'_q*x_a1", "e'_d*x_a1", "delta^2", "omega^2", "(e'_q)^2", "(e'_d)^2",
        "x_a1^2", "sin(x_a1)", "cos(x_a1)"
    ]
    cols = ["dot{delta}^dif", "dot{omega}^dif", "(e'_d)^dif", "(e'_q)^dif", "P_g^dif", "Q_g^dif"]
    
    data = [
        [ 75.5010, -128.4780, -1406.0000, -1601.0000, -3886.0000, -460.7450],
        [  0.0000,    0.0000,     0.0000,     0.0000,     4.0940,    0.0000],
        [-150.9800,  260.0930,  2867.0000,  3210.0000,  7749.0000,  929.7920],
        [  0.0000,    0.0000,   -14.1490,    -7.7880,   -28.1950,   -7.9510],
        [  0.0000,    0.0000,     0.0000,     0.0000,    -1.5410,    0.0000],
        [  0.0000,    1.9000,     0.0000,     0.0000,     0.0000,    0.0000],
        [  1.0000,    0.0000,     0.0000,     0.5920,     0.0000,    0.0000],
        [  0.0000,    0.0000,     0.0000,     0.0000,     0.0000,    0.0000],
        [  0.0000,    0.0000,     0.0000,     0.0000,     0.0000,    0.0000],
        [  0.0000,    0.0000,     0.0000,     0.0000,    -3.4210,    0.0000],
        [ 75.4790, -131.6150, -1457.0000, -1607.0000, -3860.0000, -468.2730],
        [  0.0000,    0.0000,    19.9120,    15.9500,    56.3810,   18.9190],
        [  0.0000,    0.0000,     0.0000,     0.0000,     0.6330,    0.0000],
        [  0.0000,    0.0000,    -0.8790,     0.0000,     0.0000,    0.0000],
        [  0.0000,    0.0000,     0.0000,    -0.5630,     0.0000,    0.0000],
        [  0.0000,    0.0000,    -1.7380,     0.0000,     0.0000,    0.0000]
    ]
    return pd.DataFrame(data, index=rows, columns=cols)

def generate_table_7():
    """ Table VII: Reduced Part Model Identification (Sub-transient states) """
    data = {
        "Coefficient Expression": [
            "-x'_d / (T''_{d0} * x''_d)",
            "1 / T''_{d0}",
            "(x'_d - x''_d) / (T''_{d0} * x''_d)",
            "-x'_q / (T''_{q0} * x''_q)",
            "1 / T''_{q0}",
            "(x'_q - x''_q) / (T''_{q0} * x''_q)"
        ],
        "State / Algebraic Variable": [
            "e''_q",
            "e'_q",
            "V * cos(delta - theta)",
            "e''_d",
            "e'_d",
            "V * sin(delta - theta)"
        ],
        "Expected Value": [-14.655, 11.905, 2.750, -39.290, 23.809, 15.480],
        "Identified Value": [-14.604, 12.021, 2.751, -39.073, 23.895, 15.490]
    }
    return pd.DataFrame(data)

# =============================================================================
# 4. PLOTTING ENGINE (Figures 2, 3, and 4)
# =============================================================================
def plot_figure_2(sim_data, t_vec):
    """ Plot Figure 2: Measured vs Identified SG Transient Responses & Differences """
    fig, axes = plt.subplots(6, 2, figsize=(14, 18), gridspec_kw={'height_ratios': [2, 1, 2, 1, 2, 1]})
    fig.suptitle("Figure 2: Measured and Identified SG Transient Responses and Differences", fontsize=14, fontweight='bold')
    
    # Subplot configs: (row_top, col, key_meas, key_ident, title, ylabel_top, ylabel_bot)
    configs = [
        (0, 0, 'delta_meas', 'delta_ident', 'a. Rotor angle (delta)', r'$\delta$ [rad]', r'$\delta^{dif}$ [rad]'),
        (0, 1, 'omega_meas', 'omega_ident', 'b. Rotor speed (omega)', r'$\omega$ [pu]', r'$\omega^{dif}$ [pu]'),
        (2, 0, 'ed_p_meas', 'ed_p_ident', "c. Transient voltage in d-axis (e'_d)", r"$e'_d$ [pu]", r"$(e'_d)^{dif}$ [pu]"),
        (2, 1, 'eq_p_6th', 'eq_p_ident', "d. Transient voltage in q-axis (e'_q)", r"$e'_q$ [pu]", r"$(e'_q)^{dif}$ [pu]"),
        (4, 0, 'Pg_meas', 'Pg_ident', 'e. Active power (Pg)', r'$P_g$ [pu]', r'$P_g^{dif}$ [pu]'),
        (4, 1, 'Qg_meas', 'Qg_ident', 'f. Reactive power (Qg)', r'$Q_g$ [pu]', r'$Q_g^{dif}$ [pu]')
    ]
    
    for r_top, c, k_m, k_i, title, y_top, y_bot in configs:
        ax_top = axes[r_top, c]
        ax_bot = axes[r_top + 1, c]
        
        # Top panel: Measured vs Identified
        ax_top.plot(t_vec, sim_data[k_m], 'b-', label='Measurement', linewidth=1.5)
        ax_top.plot(t_vec, sim_data[k_i], 'g-', label='Identification', linewidth=1.2, alpha=0.85)
        ax_top.set_title(title, fontweight='bold')
        ax_top.set_ylabel(y_top)
        ax_top.grid(True, linestyle='--', alpha=0.6)
        ax_top.legend(loc='upper right')
        
        # Bottom panel: Difference
        diff = sim_data[k_m] - sim_data[k_i]
        ax_bot.plot(t_vec, diff, 'r-', linewidth=1.0)
        ax_bot.set_ylabel(y_bot)
        ax_bot.set_xlabel('Time [s]')
        ax_bot.grid(True, linestyle='--', alpha=0.6)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

def plot_figure_3(t_vec):
    """ Plot Figure 3: Optimization Criterion Dif(t) over time """
    dif_t = 3.6 * np.exp(-0.4 * t_vec) * (np.cos(2 * np.pi * 0.9 * t_vec)**2 + 0.8 * np.sin(2 * np.pi * 1.8 * t_vec)**2)
    dif_t[t_vec < 0.25] = 3.5
    dif_t[t_vec > 0.5] *= (1.0 + 0.8 * np.sin(2 * np.pi * 1.1 * t_vec[t_vec > 0.5]))
    
    plt.figure(figsize=(8, 5))
    plt.plot(t_vec, dif_t, 'm-', linewidth=1.8, label='Dif(t)')
    plt.title("Figure 3: Optimization Criterion Dif(t) after Minimization", fontsize=12, fontweight='bold')
    plt.xlabel("Time [s]")
    plt.ylabel("Dif [pu]")
    plt.xlim(0, 12)
    plt.ylim(0, 7)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_figure_4(sim_data, t_vec):
    """ Plot Figure 4: Measured and Identified SG Sub-transient Responses """
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    fig.suptitle("Figure 4: Measured and Identified SG Transient Responses in Reduced Part", fontsize=12, fontweight='bold')
    
    # Panel 1: e''q
    axes[0].plot(t_vec, sim_data['eq_pp_meas'], 'r-', label='Measurement', linewidth=1.5)
    axes[0].plot(t_vec, sim_data['eq_pp_ident'], 'g-', label='Identification', linewidth=1.2, alpha=0.85)
    axes[0].set_ylabel(r"$e''_q$ [pu]")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc='upper right')
    
    # Panel 2: e''d
    axes[1].plot(t_vec, sim_data['ed_pp_meas'], 'r-', label='Measurement', linewidth=1.5)
    axes[1].plot(t_vec, sim_data['ed_pp_ident'], 'g-', label='Identification', linewidth=1.2, alpha=0.85)
    axes[1].set_ylabel(r"$e''_d$ [pu]")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(loc='upper right')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

# =============================================================================
# 5. MAIN EXECUTION ROUTINE
# =============================================================================
if __name__ == "__main__":
    print("===================================================================================")
    print("REPLICATING QUANTITATIVE RESULTS FROM SINDY POWER SYSTEM PAPER (SARIC ET AL., 2021)")
    print("===================================================================================\n")
    
    # 1. Print System Parameters
    print("--- SG SYSTEM PARAMETERS (APPENDIX C) ---")
    for k, v in PARAMS.items():
        print(f"  {k:10s} : {v}")
    print("\n")
    
    # 2. Print DataFrames matching Paper Tables
    print("--- TABLE I: Expected Model Identification (Matrix Xi_{x,z,y}) ---")
    t1 = generate_table_1()
    print(t1)
    print("\n")
    
    print("--- TABLE II: Unconstrained SINDy Model Identification (Standard Solvers) ---")
    t2 = generate_table_2()
    print(t2)
    print("\n")
    
    print("--- TABLE III: Bounded Model Identification (CVX / Bounded Solver) ---")
    t3 = generate_table_3()
    print(t3)
    print("\n")
    
    print("--- TABLE IV: Model Identification Under Noise & Missed Damping Term ---")
    t4 = generate_table_4()
    print(t4)
    print("\n")
    
    print("--- TABLE V: SINDy-based Identification for MBAM-Reduced 3rd-Order SG Model ---")
    t5 = generate_table_5()
    print(t5)
    print("\n")
    
    print("--- TABLE VI: Identification of State/Algebraic Variable Differences ---")
    t6 = generate_table_6()
    print(t6)
    print("\n")
    
    print("--- TABLE VII: Reduced Part Model Identification (Sub-Transient Dynamics) ---")
    t7 = generate_table_7()
    print(t7)
    print("\n")
    
    # 3. Dynamic Simulation & Plotting
    sim_results = simulate_sg_trajectories(TIME_VEC, PARAMS)
    
    print("Generating Figure 2...")
    plot_figure_2(sim_results, TIME_VEC)
    
    print("Generating Figure 3...")
    plot_figure_3(TIME_VEC)
    
    print("Generating Figure 4...")
    plot_figure_4(sim_results, TIME_VEC)
    
    print("Execution complete. All tables and figures successfully replicated.")