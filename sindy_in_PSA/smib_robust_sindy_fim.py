"""
SMIB robust SINDy + FIM-guided model reduction demo
--------------------------------------------------
Dependencies: numpy pandas scipy matplotlib

Important:
- This is a reproducible synthetic SMIB experiment based on the user's supplied
  4-state equations.
- It replaces the artificial MBAM parameter multipliers with data-driven
  Fisher-information-guided pruning. That is an MBAM-inspired approximation,
  NOT the full geodesic Manifold Boundary Approximation Method.
- Robust Savitzky-Golay smoothing + Huber IRLS regression are used to reduce
  sensitivity to heavy-tailed/impulsive noise.
- Parameter extraction from sparse coefficients is deliberately not presented
  as validated unless the selected library supports those mappings.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter

# --------------------------- CONFIG ---------------------------
SEED = 42
NOISE_TYPE = "laplace"       # gaussian, laplace, student_t, uniform, impulsive
NOISE_LEVEL = 0.07
DT = 0.01
T_END = 12.0
FAULT_START, FAULT_END = 0.5, 0.75
TRAIN_END = 0.45             # fit only pre-fault data; avoid event contamination
SG_WINDOW = 11               # odd number; at dt=0.01 this is 0.11 s
SG_POLY = 3
HUBER_DELTA = 1.5
STLS_THRESHOLD = 0.04         # threshold in normalized coefficient coordinates
RIDGE = 1e-5
FIM_CONDITION_LIMIT = 1e-7
# ---------------------- SMIB parameters -----------------------
OMEGA_B = 2 * np.pi * 50.0
OMEGA_0 = 1.0
H2 = 12.467
D = 1.9
TD0_PRIME = 6.5
TQ0_PRIME = 0.2
XD, XQ = 0.655, 0.487
XD_PRIME, XQ_PRIME = 0.196, 0.263
VF_CONST, PM_CONST = 1.4998, 1.6

rng = np.random.default_rng(SEED)

def grid_voltage(t):
    if FAULT_START <= t <= FAULT_END:
        return 0.3, 0.0
    if t > FAULT_END:
        v = 1.0 - 0.15*np.exp(-0.8*(t-FAULT_END))*np.cos(2*np.pi*1.5*t)
        th = 0.1*np.exp(-0.8*(t-FAULT_END))*np.sin(2*np.pi*1.5*t)
        return v, th
    return 1.0, 0.0

def smib_rhs(t, x):
    delta, omega, eqp, edp = x
    v, th = grid_voltage(t)
    dd = delta - th
    id_ = (eqp - v*np.cos(dd))/XD_PRIME
    iq_ = (v*np.sin(dd) - edp)/XQ_PRIME
    pe = v*np.sin(dd)*id_ + v*np.cos(dd)*iq_
    return [
        OMEGA_B*(omega-OMEGA_0),
        (PM_CONST-pe-D*(omega-OMEGA_0))/H2,
        (VF_CONST-eqp-(XD-XD_PRIME)*id_)/TD0_PRIME,
        (-edp+(XQ-XQ_PRIME)*iq_)/TQ0_PRIME
    ]

def noise(shape, scale, kind):
    if kind == "gaussian":
        return rng.normal(0, scale, shape)
    if kind == "laplace":
        return rng.laplace(0, scale/np.sqrt(2), shape)
    if kind == "student_t":
        df = 3
        return rng.standard_t(df, shape) / np.sqrt(df/(df-2)) * scale
    if kind == "uniform":
        return rng.uniform(-np.sqrt(3)*scale, np.sqrt(3)*scale, shape)
    if kind == "impulsive":
        base = rng.normal(0, scale, shape)
        mask = rng.random(shape) < 0.02
        spikes = rng.choice([-1.0, 1.0], shape) * (10*scale) * mask
        return base + spikes
    raise ValueError(f"Unknown NOISE_TYPE={kind}")

def robust_smooth(x):
    # Savitzky-Golay is a smoother, not a guaranteed robust filter.
    # Clip isolated large innovations before smoothing using a rolling robust scale.
    x = np.asarray(x, float)
    med = pd.Series(x).rolling(9, center=True, min_periods=1).median().to_numpy()
    resid = x - med
    mad = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-10
    clipped = med + np.clip(resid, -5*mad, 5*mad)
    w = min(SG_WINDOW, len(x) if len(x) % 2 else len(x)-1)
    w = max(w, SG_POLY + 2 + ((SG_POLY + 2) % 2 == 0))
    if w >= len(x):
        w = len(x)-1 if len(x) % 2 == 0 else len(x)
    return savgol_filter(clipped, w, min(SG_POLY, w-1), mode="interp")

def library(x, v, th):
    delta, omega, eqp, edp = x.T
    dth = delta - th
    names = [
        "1", "delta", "omega", "eqp", "edp",
        "V*cos(delta-theta)", "V*sin(delta-theta)",
        "V*eqp*sin(delta-theta)", "V*edp*cos(delta-theta)",
        "V^2*sin(delta-theta)*cos(delta-theta)"
    ]
    T = np.column_stack([
        np.ones(len(delta)), delta, omega, eqp, edp,
        v*np.cos(dth), v*np.sin(dth),
        v*eqp*np.sin(dth), v*edp*np.cos(dth),
        v*v*np.sin(dth)*np.cos(dth)
    ])
    return names, T

def huber_irls(A, y, delta=1.5, ridge=1e-5, iterations=30):
    """Huber IRLS with ridge stabilization; assumes columns already standardized."""
    w = np.ones(len(y))
    coef = np.zeros(A.shape[1])
    for _ in range(iterations):
        Aw = A * np.sqrt(w[:, None])
        yw = y * np.sqrt(w)
        coef_new = np.linalg.solve(Aw.T @ Aw + ridge*np.eye(A.shape[1]), Aw.T @ yw)
        r = y - A @ coef_new
        s = 1.4826*np.median(np.abs(r - np.median(r))) + 1e-10
        u = np.abs(r)/(delta*s)
        w_new = np.ones_like(u)
        mask = u > 1
        w_new[mask] = 1/u[mask]
        if np.linalg.norm(coef_new-coef) < 1e-8*(1+np.linalg.norm(coef)):
            coef = coef_new
            break
        coef, w = coef_new, w_new
    return coef

def fit_robust_stlsq(A, Y, threshold=0.04):
    mu = A.mean(axis=0)
    sd = A.std(axis=0)
    valid = sd > 1e-10
    sd[~valid] = 1.0
    Z = (A-mu)/sd
    Z[:, ~valid] = 0.0
    coefs = np.zeros((A.shape[1], Y.shape[1]))
    for j in range(Y.shape[1]):
        ymu = np.median(Y[:, j])
        ys = 1.4826*np.median(np.abs(Y[:, j]-ymu)) + 1e-8
        yn = (Y[:, j]-ymu)/ys
        support = valid.copy()
        for _ in range(15):
            c = huber_irls(Z[:, support], yn, HUBER_DELTA, RIDGE)
            ids = np.flatnonzero(support)
            keep = np.abs(c) >= threshold
            new = np.zeros_like(support)
            new[ids[keep]] = True
            if np.array_equal(new, support):
                break
            support = new
            if not support.any():
                support[np.argmax(np.abs(np.linalg.lstsq(Z, yn, rcond=None)[0]))] = True
        if support.any():
            c = huber_irls(Z[:, support], yn, HUBER_DELTA, RIDGE)
            coefs[np.flatnonzero(support), j] = c * ys / sd[support]
            coefs[:, j] -= 0.0
    # Coefficients above are mapped to original library columns. Centering intercept
    # correction is omitted here because the constant is included as a library term.
    return coefs, Z, sd, valid

def fim_prune(A, Xi, condition_limit=1e-7):
    """FIM-guided pruning heuristic. Not full MBAM."""
    out = Xi.copy()
    for j in range(Xi.shape[1]):
        active = np.flatnonzero(np.abs(out[:, j]) > 1e-10)
        while len(active) > 1:
            F = A[:, active].T @ A[:, active] / len(A)
            eigval, eigvec = np.linalg.eigh(F)
            if eigval[-1] <= 1e-12 or eigval[0]/eigval[-1] >= condition_limit:
                break
            drop_local = np.argmax(np.abs(eigvec[:, 0]))
            drop = active[drop_local]
            out[drop, j] = 0.0
            active = np.flatnonzero(np.abs(out[:, j]) > 1e-10)
            if len(active) <= 1:
                break
            # Refit on remaining support with Huber loss.
            out[active, j] = huber_irls(A[:, active], Y_train[:, j], HUBER_DELTA, RIDGE)
    return out

# Simulate clean system
t = np.arange(0, T_END, DT)
sol = solve_ivp(smib_rhs, (t[0], t[-1]), [0.8, 1.0, 1.05, 0.2],
                t_eval=t, rtol=1e-8, atol=1e-10, max_step=DT)
X_clean = sol.y.T
V = np.array([grid_voltage(ti)[0] for ti in t])
theta = np.array([grid_voltage(ti)[1] for ti in t])
X_noisy = X_clean + np.column_stack([
    noise((len(t),), NOISE_LEVEL, NOISE_TYPE),
    noise((len(t),), NOISE_LEVEL*0.1, NOISE_TYPE),
    noise((len(t),), NOISE_LEVEL, NOISE_TYPE),
    noise((len(t),), NOISE_LEVEL*0.5, NOISE_TYPE)
])
X_smooth = np.column_stack([robust_smooth(X_noisy[:, j]) for j in range(4)])

# Derivatives are estimated from smoothed measurements.
Xdot = np.column_stack([np.gradient(X_smooth[:, j], DT, edge_order=2) for j in range(4)])
names, Theta = library(X_smooth, V, theta)
train = t < TRAIN_END
# Avoid derivative edge samples from the Savitzky-Golay filter.
train[:SG_WINDOW//2] = False
train_idx = np.flatnonzero(train)
if len(train_idx) < 20:
    raise RuntimeError("Too few pre-fault training samples. Increase TRAIN_END or reduce smoothing window.")
Y_train = Xdot[train_idx]
# Normalize feature columns for robust regression.
mu = Theta[train_idx].mean(axis=0)
sd = Theta[train_idx].std(axis=0)
sd[sd < 1e-10] = 1.0
A_all = (Theta-mu)/sd
A_train = A_all[train_idx]
# Fit with robust STLSQ in standardized coordinates; coefficient matrix is standardized-coordinate coefficients.
Xi_full = np.zeros((A_train.shape[1], Y_train.shape[1]))
for j in range(Y_train.shape[1]):
    ys = 1.4826*np.median(np.abs(Y_train[:,j]-np.median(Y_train[:,j]))) + 1e-8
    yn = (Y_train[:,j]-np.median(Y_train[:,j]))/ys
    support = np.ones(A_train.shape[1], dtype=bool)
    for _ in range(15):
        ids = np.flatnonzero(support)
        c = huber_irls(A_train[:, support], yn, HUBER_DELTA, RIDGE)
        keep_local = np.abs(c) >= STLS_THRESHOLD
        if not keep_local.any():
            keep_local[np.argmax(np.abs(c))] = True
        new_support = np.zeros_like(support)
        new_support[ids[keep_local]] = True
        if np.array_equal(new_support, support):
            break
        support = new_support
    if support.any():
        c = huber_irls(A_train[:, support], yn, HUBER_DELTA, RIDGE)
        Xi_full[support,j] = c * ys

# FIM-guided reduction: only use training data; refit after every deletion.
Xi_reduced = Xi_full.copy()
for j in range(Xi_full.shape[1]):
    active = np.flatnonzero(np.abs(Xi_reduced[:,j]) > 1e-10)
    while len(active) > 1:
        F = A_train[:,active].T @ A_train[:,active] / len(A_train)
        eigval, eigvec = np.linalg.eigh(F)
        if eigval[-1] <= 1e-12 or eigval[0]/eigval[-1] >= FIM_CONDITION_LIMIT:
            break
        drop = active[np.argmax(np.abs(eigvec[:,0]))]
        active = active[active != drop]
        Xi_reduced[:,j] = 0
        if len(active):
            # refit normalized response with robust loss
            yj = Y_train[:,j]
            ys = 1.4826*np.median(np.abs(yj-np.median(yj))) + 1e-8
            yn = (yj-np.median(yj))/ys
            Xi_reduced[active,j] = huber_irls(A_train[:,active], yn, HUBER_DELTA, RIDGE) * ys

# Predictions on all samples, and evaluation against clean derivatives.
pred_full = A_all @ Xi_full
pred_red = A_all @ Xi_reduced
clean_dot = np.column_stack([np.gradient(X_clean[:,j], DT, edge_order=2) for j in range(4)])
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
print(f"\nSMIB robust SINDy | noise={NOISE_TYPE}, level={NOISE_LEVEL}")
print(f"Training samples: {len(train_idx)}; library terms: {len(names)}")
print(f"Nonzero coefficients: full={np.count_nonzero(Xi_full)}, reduced={np.count_nonzero(Xi_reduced)}")
print(f"Derivative RMSE vs clean numerical derivative: full={rmse(pred_full,clean_dot):.5g}, reduced={rmse(pred_red,clean_dot):.5g}")
print("\nIdentified standardized-coordinate coefficients (rows=library terms, columns=state derivatives):")
print(pd.DataFrame(Xi_reduced, index=names, columns=["delta_dot","omega_dot","eqp_dot","edp_dot"]).round(5).to_string())

fig, ax = plt.subplots(4,1,figsize=(12,10),sharex=True)
labels = ["delta (rad)", "omega (pu)", "eq' (pu)", "ed' (pu)"]
for j in range(4):
    ax[j].plot(t, X_clean[:,j], label="clean", lw=1.5)
    ax[j].plot(t, X_noisy[:,j], label="noisy", alpha=.35, lw=.7)
    ax[j].plot(t, X_smooth[:,j], label="robust-smoothed", lw=1.0)
    ax[j].axvspan(FAULT_START, FAULT_END, color="red", alpha=.08)
    ax[j].set_ylabel(labels[j]); ax[j].grid(alpha=.25)
    if j == 0: ax[j].legend(ncol=3)
ax[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig("smib_robust_sindy_fim.png", dpi=180)
plt.show()
