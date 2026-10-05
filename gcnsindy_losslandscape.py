"""
Joint GCN + SINDy training with pruning-transient diagnostics  (v2)

What changed vs v1 (all switchable from the CONFIG block):
  1. REFIT before thresholding: at every pruning event Xi is refit in closed form
     (least squares on the current support) before deciding what to prune, and refit
     again on the reduced support. This is the STLS logic your original script had.
     Set refit=False in RUNS to reproduce v1 behaviour (prune the raw GD iterate).
  2. GRADUAL pruning: at most MAX_PRUNE_PER_EVENT terms are removed per event
     (smallest first, and only if below PRUNE_LAMBDA). Events that find nothing to
     remove do NOTHING (no refit, no mask change) so they act as clean null controls.
  3. GCN learning-rate decay (cosine) in the joint phase, and W_L1 = 0 by default
     (hard thresholding + refit already provides sparsity; flip W_L1 to compare).
  4. L1 term plotted separately; Xi step size can follow 1/L of the Xi-Hessian;
     recovery is measured against the post-prune PLATEAU (not the pre-prune value),
     and a non-event "jitter" baseline is printed so you can tell pruning transients
     from ordinary Adam noise.
  5. Runs are wrapped in run(...) so keep/reset, dropout levels and seeds can be
     swept from the RUNS list at the bottom. A summary table is printed at the end,
     with support-recovery / coefficient-error metrics against true Lorenz.

Logged every joint step: loss components, ||grad Xi||, ||grad GCN||, momentum norm,
cos(prev momentum, gradient), #active terms, Xi-Hessian lambda_max / condition number,
and the Xi learning rate actually used.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==============================================================================
# CONFIG
# ==============================================================================
DT = 0.01                       # sampling interval of the CSV
PRETRAIN_STEPS = 1500           # GCN-only warm-up
JOINT_STEPS = 4000              # joint GCN + Xi training

# ---- pruning schedule ----
PRUNE_START = 800               # first event (joint-step index)
PRUNE_EVERY = 300               # steps between events
N_EVENTS = 8
PRUNE_LAMBDA = 0.5              # physical-unit threshold: only |coef| < this may be pruned
MAX_PRUNE_PER_EVENT = 4         # gradual pruning: at most this many terms per event

# ---- optimisation ----
GCN_LR = 2e-3                   # Adam for GCN weights
GCN_LR_DECAY = True             # cosine decay to 2% of GCN_LR over the joint phase
XI_LR_MODE = "inv_L"            # "inv_L": lr = XI_LR_SCALE / lambda_max (tracks curvature)
XI_LR_SCALE = 0.9               #          (Nesterov is stable up to ~1.36 / lambda_max)
XI_LR_FIXED = 0.05              # "fixed": constant lr
XI_MOMENTUM = 0.9

# ---- loss weights ----
W_DATA = 1.0
W_DERIV = 1.0                   # ties the velocity head to d/dt of the position head
W_SINDY = 1.0
W_L1 = 0.0                      # try 1e-3 to compare against the v1 setting

RECOVERY_TOL = 0.10             # "recovered" = smoothed metric within 10% of its post-prune plateau
SHOW_PLOTS = True               # False for sweeps (figures are still saved as PNG)

# ---- 3D loss-landscape snapshots (before / after one chosen pruning event) ----
# Plots the FULL joint loss (data + deriv + sindy) along two random directions in the
# combined (GCN weights, Xi) parameter space, filter/tensor-normalised per Li et al. 2018
# ("Visualizing the Loss Landscape of Neural Nets"), the same recipe as the reference image.
# Xi's own sub-problem is a convex quadratic (least squares) -> alone it renders as a smooth
# bowl, not a mountain. Including the GCN weights is what brings back the non-convex ruggedness.
PLOT_LANDSCAPE = True
LANDSCAPE_EVENT_STEP = 800      # which pruning event to snapshot (must be in the event schedule)
LANDSCAPE_GRID = 41             # grid resolution per axis (41x41 = 1681 forward passes, cheap)
LANDSCAPE_SPAN = 1.0            # alpha, beta range: [-LANDSCAPE_SPAN, +LANDSCAPE_SPAN]
LANDSCAPE_SEED = 0              # seed for the two random directions (same seed -> same directions
                                # for the pre- and post-prune plot, so the two surfaces are comparable)

library_labels = ["1", "x", "y", "z", "x^2", "xy", "xz", "y^2", "yz", "z^2"]
P = len(library_labels)

# True Lorenz (sigma=10, rho=28, beta=8/3) in this library ordering, for benchmark metrics only.
XI_TRUE = np.zeros((P, 3))
XI_TRUE[1, 0], XI_TRUE[2, 0] = -10.0, 10.0
XI_TRUE[1, 1], XI_TRUE[2, 1], XI_TRUE[6, 1] = 28.0, -1.0, -1.0
XI_TRUE[3, 2], XI_TRUE[5, 2] = -8.0 / 3.0, 1.0

# ==============================================================================
# Device
# ==============================================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# ==============================================================================
# Data (falls back to simulated Lorenz if the CSV is missing)
# ==============================================================================
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

csv_path = os.path.join(base_dir, "Lorentz_attractor_5s_0.010_rate.csv")
if os.path.exists(csv_path):
    data = pd.read_csv(csv_path)[["X", "Y", "Z"]].values.astype(np.float64)
    print(f"Loaded {len(data)} nodes from CSV.")
else:
    print("CSV not found -> simulating Lorenz for 5 s.")

    def _lorenz(t, s):
        return [10 * (s[1] - s[0]), s[0] * (28 - s[2]) - s[1], s[0] * s[1] - (8 / 3) * s[2]]

    data = solve_ivp(_lorenz, (0, 5), [-8.0, 7.0, 27.0], t_eval=np.arange(0, 5.0 + 1e-9, DT),
                     rtol=1e-10, atol=1e-10).y.T

N = len(data)
idx = np.arange(N)


# ==============================================================================
# Helpers shared by all runs
# ==============================================================================
def library_np(X):
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    return np.column_stack([np.ones(len(X)), x, y, z, x * x, x * y, x * z, y * y, y * z, z * z])


def library_t(X):
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    return torch.stack([torch.ones_like(x), x, y, z, x * x, x * y, x * z, y * y, y * z, z * z], dim=1)


def normalized_adjacency(n):
    a = np.eye(n)
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = 1.0
    d = a.sum(axis=1) ** -0.5
    return (d[:, None] * a) * d[None, :]


class GCNLayer(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.linear = nn.Linear(i, o)

    def forward(self, x, adj):
        return adj @ self.linear(x)


class TimeSeriesGCN(nn.Module):
    """Two-layer GCN, 6 outputs: 3 normalised positions + 3 normalised velocities."""

    def __init__(self, hidden=128):
        super().__init__()
        self.g1 = GCNLayer(3, hidden)
        self.g2 = GCNLayer(hidden, 6)

    def forward(self, x, adj):
        return self.g2(torch.relu(self.g1(x, adj)), adj)


def random_direction(model, xi_shape, mask_np, seed):
    """One random direction in the combined (GCN weights, Xi) space, filter/tensor-normalised:
    each GCN parameter TENSOR is rescaled so ||direction_tensor|| / ||param_tensor|| matches the
    tensor's own norm (Li et al. 2018's trick for making step sizes comparable across layers of
    very different scale). The Xi block is drawn only on currently-ACTIVE entries (pruned entries
    must stay exactly at 0, matching the lock-in in the real training loop) and is scaled to match
    the norm of the active Xi entries themselves."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    d_model = {}
    for name, p in model.named_parameters():
        d = torch.randn(p.shape, generator=g)
        d = d / (d.norm() + 1e-12) * p.detach().cpu().norm()
        d_model[name] = d.to(p.device)
    d_xi = torch.randn(xi_shape, generator=g)
    d_xi = d_xi * torch.tensor(mask_np, dtype=torch.float32)       # zero out pruned entries
    return d_model, d_xi


def landscape_loss_grid(model, base_xi, mask_t, d1_model, d2_model, d1_xi, d2_xi,
                        x_in_n, adj_t, data_n, valid_t, mu_t, sd_t, vsc_t, cs_t,
                        grid=41, span=1.0):
    """Total loss (data + deriv + sindy, equal weights) on a grid of (alpha, beta) offsets.
    Temporarily perturbs the model's own parameters in-place for each grid point and restores
    them afterward -- cheap (no cloning) since this GCN is tiny."""
    base_state = {n: p.detach().clone() for n, p in model.named_parameters()}
    alphas = np.linspace(-span, span, grid)
    betas = np.linspace(-span, span, grid)
    Z = np.zeros((grid, grid))
    with torch.no_grad():
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                for n, p in model.named_parameters():
                    p.copy_(base_state[n] + a * d1_model[n] + b * d2_model[n])
                xi_pert = (base_xi + a * d1_xi.to(base_xi.device) + b * d2_xi.to(base_xi.device)) * mask_t
                out = model(x_in_n, adj_t)
                X_n, V_n = out[:, :3], out[:, 3:]
                X_phys = X_n * sd_t + mu_t
                l_data = (((X_n - data_n) ** 2).mean(dim=1) * valid_t).sum() / valid_t.sum()
                D = (X_phys[2:] - X_phys[:-2]) / (2 * DT) / vsc_t
                l_deriv = ((V_n[1:-1] - D) ** 2).mean()
                Th = library_t(X_phys) / cs_t[None, :]
                l_sindy = ((V_n - Th @ xi_pert) ** 2).mean()
                Z[j, i] = (l_data + l_deriv + l_sindy).item()      # [row=beta, col=alpha] for plot_surface
        for n, p in model.named_parameters():                      # restore
            p.copy_(base_state[n])
    return alphas, betas, Z


def plot_landscape_pair(snap_pre, snap_post, step, base_dir, tag, model, mask,
                        x_in_n, adj_t, data_n, valid_t, mu_t, sd_t, vsc_t, cs_t,
                        grid=LANDSCAPE_GRID, span=LANDSCAPE_SPAN, seed=LANDSCAPE_SEED):
    """snap_pre / snap_post: dict(model_state, xi, mask) captured immediately before / after
    one pruning event (same GCN forward, since the GCN hasn't stepped between the two -- the
    ONLY thing that changes is which Xi entries are free and their refit/zeroed values, so this
    isolates exactly the landscape shift the pruning event causes). model/mask are the live
    objects from the training run; this function temporarily overwrites their contents for each
    snapshot and the caller is responsible for restoring the post-training state afterward."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

    fig = plt.figure(figsize=(15, 6.5))
    fig.suptitle(f"Loss landscape around the Xi/GCN point, before vs. after pruning (step {step}) | {tag}")
    surfaces = []
    for snap, title in [(snap_pre, "BEFORE prune"), (snap_post, "AFTER prune")]:
        model.load_state_dict(snap["model_state"])
        mask.copy_(snap["mask"])
        d1_model, d1_xi = random_direction(model, snap["xi"].shape, snap["mask"].cpu().numpy(), seed)
        d2_model, d2_xi = random_direction(model, snap["xi"].shape, snap["mask"].cpu().numpy(), seed + 1)
        alphas, betas, Z = landscape_loss_grid(
            model, snap["xi"], snap["mask"], d1_model, d2_model, d1_xi, d2_xi,
            x_in_n, adj_t, data_n, valid_t, mu_t, sd_t, vsc_t, cs_t, grid=grid, span=span)
        surfaces.append((alphas, betas, Z, title, int(snap["mask"].sum().item())))

    for k, (alphas, betas, Z, title, n_active) in enumerate(surfaces):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        A, B = np.meshgrid(alphas, betas)
        ax.plot_surface(A, B, np.log10(Z + 1e-8), cmap="viridis", linewidth=0, antialiased=True, rcount=grid, ccount=grid)
        ax.set_title(f"{title}  ({n_active} active Xi terms)")
        ax.set_xlabel("direction 1"); ax.set_ylabel("direction 2"); ax.set_zlabel("log10(total loss)")
        ax.view_init(elev=35, azim=-60)

    fig.tight_layout()
    fig_path = os.path.join(base_dir, f"loss_landscape_step{step}_{tag}.png")
    fig.savefig(fig_path, dpi=150)
    print(f"  Saved landscape figure -> {fig_path}")
    return fig


def refit_xi(Th_n, V_n, active):
    """Closed-form least squares per state on the active support. Th_n (N,P), V_n (N,3) float64."""
    out = np.zeros((P, 3))
    for k in range(3):
        a = np.where(active[:, k])[0]
        if len(a):
            out[a, k] = np.linalg.lstsq(Th_n[:, a], V_n[:, k], rcond=None)[0]
    return out


ADJ_T = torch.tensor(normalized_adjacency(N), dtype=torch.float32, device=device)


def to_t(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32, device=device)


# ==============================================================================
# One full experiment
# ==============================================================================
def run(prune_mode="keep", refit=False, seed=0, dropout=0.0):
    tag = f"{prune_mode}_refit{int(refit)}_drop{int(round(dropout * 100))}_seed{seed}"
    print(f"\n{'=' * 78}\nRUN {tag}\n{'=' * 78}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # ---- corrupted / missing nodes: loss ignores them, input is linearly filled ----
    valid = np.ones(N, dtype=bool)
    if dropout > 0:
        drop = rng.choice(np.arange(1, N - 1), size=int(dropout * N), replace=False)
        valid[drop] = False
    data_in = data.copy()
    for k in range(3):
        data_in[~valid, k] = np.interp(idx[~valid], idx[valid], data[valid, k])
    print(f"Valid nodes: {valid.sum()}/{N}")

    # ---- fixed normalisation constants ----
    mu, sd = data_in.mean(axis=0), data_in.std(axis=0)
    dX_fd = np.gradient(data_in, DT, axis=0)
    vsc = dX_fd.std(axis=0)
    col_scale = np.sqrt((library_np(data_in) ** 2).mean(axis=0))
    col_scale[col_scale < 1e-8] = 1.0

    def to_physical(xi_n):
        # Xi_phys[j,k] = Xi_n[j,k] * vsc[k] / col_scale[j]
        return xi_n * vsc[None, :] / col_scale[:, None]

    Xi0 = np.linalg.lstsq(library_np(data_in) / col_scale[None, :], dX_fd / vsc[None, :], rcond=None)[0]

    x_in_n = to_t((data_in - mu) / sd)
    data_n = to_t((data - mu) / sd)
    valid_t = to_t(valid.astype(np.float32))
    mu_t, sd_t, vsc_t, cs_t = to_t(mu), to_t(sd), to_t(vsc), to_t(col_scale)

    model = TimeSeriesGCN().to(device)
    Xi_n = nn.Parameter(to_t(Xi0))
    mask = torch.ones(P, 3, device=device)
    opt_gcn = torch.optim.Adam(model.parameters(), lr=GCN_LR)
    lr0 = XI_LR_FIXED if XI_LR_MODE == "fixed" else 0.05
    opt_xi = torch.optim.SGD([Xi_n], lr=lr0, momentum=XI_MOMENTUM, nesterov=True)

    def forward_all():
        out = model(x_in_n, ADJ_T)
        X_n, V_n = out[:, :3], out[:, 3:]
        return X_n, V_n, X_n * sd_t + mu_t

    def compute_losses(with_sindy=True):
        X_n, V_n, X_phys = forward_all()
        l_data = (((X_n - data_n) ** 2).mean(dim=1) * valid_t).sum() / valid_t.sum()
        D = (X_phys[2:] - X_phys[:-2]) / (2 * DT) / vsc_t
        l_deriv = ((V_n[1:-1] - D) ** 2).mean()
        if not with_sindy:
            return l_data, l_deriv, None, None, None
        Th = library_t(X_phys) / cs_t[None, :]
        xi_eff = Xi_n * mask
        l_sindy = ((V_n - Th @ xi_eff) ** 2).mean()
        return l_data, l_deriv, l_sindy, xi_eff.abs().sum(), Th

    # ---------------- Phase 1: GCN warm-up ----------------
    print("--- Phase 1: GCN warm-up ---")
    for ep in range(PRETRAIN_STEPS):
        opt_gcn.zero_grad()
        l_data, l_deriv, *_ = compute_losses(with_sindy=False)
        (W_DATA * l_data + W_DERIV * l_deriv).backward()
        opt_gcn.step()
        if ep % 500 == 0 or ep == PRETRAIN_STEPS - 1:
            print(f"  step {ep:5d} | data {l_data.item():.5f} | deriv-consistency {l_deriv.item():.5f}")

    sched = None
    if GCN_LR_DECAY:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_gcn, T_max=JOINT_STEPS, eta_min=GCN_LR * 0.02)

    # ---------------- pruning machinery ----------------
    event_steps = [PRUNE_START + i * PRUNE_EVERY for i in range(N_EVENTS)]
    assert max(event_steps) + 100 < JOINT_STEPS, "Not enough steps after the last pruning event"
    event_info = {}
    landscape_snaps = {}
    if PLOT_LANDSCAPE and LANDSCAPE_EVENT_STEP not in event_steps:
        print(f"  WARNING: LANDSCAPE_EVENT_STEP={LANDSCAPE_EVENT_STEP} is not one of {event_steps}; "
              f"no landscape will be captured for this run.")

    def hessian_stats(Th_np):
        """Hessian of L_sindy wrt column k of Xi is (2/(3N)) Th^T Th on the active terms."""
        G = Th_np.T @ Th_np
        m = mask.cpu().numpy() > 0
        lmax, kap = [], []
        for k in range(3):
            a = np.where(m[:, k])[0]
            if len(a) == 0:
                continue
            ev = np.linalg.eigvalsh((2.0 / (3 * N)) * G[np.ix_(a, a)])
            lmax.append(ev[-1])
            kap.append(ev[-1] / max(ev[0], 1e-12))
        return (max(lmax) if lmax else np.nan), (max(kap) if kap else np.nan)

    def prune(step):
        with torch.no_grad():
            _, V_n, X_phys = forward_all()
            Th_np = (library_t(X_phys) / cs_t[None, :]).cpu().numpy().astype(np.float64)
            V_np = V_n.cpu().numpy().astype(np.float64)
        active = mask.cpu().numpy() > 0
        cur = (Xi_n * mask).detach().cpu().numpy().astype(np.float64)
        ref = refit_xi(Th_np, V_np, active)
        dist = np.linalg.norm(cur - ref) / (np.linalg.norm(ref) + 1e-12)   # how far GD was from the exact minimiser

        xi_pre = ref if refit else cur
        phys = to_physical(xi_pre)
        cand = np.argwhere(active & (np.abs(phys) < PRUNE_LAMBDA))
        order = np.argsort([abs(phys[j, k]) for j, k in cand]) if len(cand) else []
        chosen = [(int(cand[i][0]), int(cand[i][1])) for i in order[:MAX_PRUNE_PER_EVENT]]

        want_snapshot = PLOT_LANDSCAPE and step == LANDSCAPE_EVENT_STEP
        if want_snapshot:
            landscape_snaps["pre"] = dict(
                model_state={n: p.detach().clone() for n, p in model.named_parameters()},
                xi=Xi_n.detach().clone(), mask=mask.clone())

        if not chosen:                                   # null control: touch nothing
            event_info[step] = dict(removed=[], n_removed=0, dist=dist)
            print(f"  [step {step}] NULL event (nothing below lambda={PRUNE_LAMBDA}); "
                  f"GD iterate is {dist:.1%} from the exact minimiser")
            if want_snapshot:
                print(f"  [step {step}] LANDSCAPE_EVENT_STEP was a null event (nothing pruned) -- "
                      f"'before' and 'after' will be identical. Pick a step that actually prunes something.")
                landscape_snaps["post"] = dict(model_state=landscape_snaps["pre"]["model_state"],
                                               xi=Xi_n.detach().clone(), mask=mask.clone())
            return

        new_active = active.copy()
        for j, k in chosen:
            new_active[j, k] = False
        if refit:
            xi_new = refit_xi(Th_np, V_np, new_active)
        else:
            xi_new = cur.copy()
            for j, k in chosen:
                xi_new[j, k] = 0.0

        with torch.no_grad():
            mask.copy_(torch.tensor(new_active, dtype=torch.float32, device=device))
            Xi_n.copy_(torch.tensor(xi_new, dtype=torch.float32, device=device))
            buf = opt_xi.state[Xi_n].get("momentum_buffer")
            if buf is not None:
                buf.mul_(mask)
                if prune_mode == "reset":
                    buf.zero_()
        removed = [f"{library_labels[j]}->{'xyz'[k]}" for j, k in chosen]
        event_info[step] = dict(removed=removed, n_removed=len(removed), dist=dist)
        print(f"  [step {step}] PRUNE removed {removed} | active now {int(new_active.sum())} | "
              f"GD iterate was {dist:.1%} from exact minimiser | refit={refit}, momentum={prune_mode}")

        if want_snapshot:
            landscape_snaps["post"] = dict(
                model_state={n: p.detach().clone() for n, p in model.named_parameters()},
                xi=Xi_n.detach().clone(), mask=mask.clone())

    # ---------------- Phase 2: joint training ----------------
    print(f"--- Phase 2: joint training ---")
    log, xi_path = [], []
    for step in range(JOINT_STEPS):
        if step in event_steps:
            prune(step)

        opt_gcn.zero_grad()
        opt_xi.zero_grad()
        l_data, l_deriv, l_sindy, l_l1, Th = compute_losses(with_sindy=True)
        total = W_DATA * l_data + W_DERIV * l_deriv + W_SINDY * l_sindy + W_L1 * l_l1
        total.backward()

        # ---- diagnostics (before the optimisers step) ----
        g = (Xi_n.grad * mask).detach()
        buf_prev = opt_xi.state[Xi_n].get("momentum_buffer")
        act = mask > 0
        if buf_prev is not None and act.any():
            gv, bv = g[act], buf_prev[act]
            cos = (gv * bv).sum() / (gv.norm() * bv.norm() + 1e-12)
            bnorm = bv.norm()
        else:
            cos = torch.tensor(0.0, device=device)
            bnorm = torch.tensor(0.0, device=device)
        g_gcn = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None))
        scal = torch.stack([total, l_data, l_deriv, l_sindy, l_l1, g.norm(), g_gcn, bnorm, cos]).detach().cpu().numpy()
        lam_max, kappa = hessian_stats(Th.detach().cpu().numpy().astype(np.float64))

        if XI_LR_MODE == "inv_L" and np.isfinite(lam_max) and lam_max > 0:
            opt_xi.param_groups[0]["lr"] = XI_LR_SCALE / lam_max
        xi_lr = opt_xi.param_groups[0]["lr"]

        opt_gcn.step()
        opt_xi.step()
        if sched is not None:
            sched.step()
        with torch.no_grad():                                    # lock-in
            Xi_n.mul_(mask)
            b = opt_xi.state[Xi_n].get("momentum_buffer")
            if b is not None:
                b.mul_(mask)

        log.append(dict(step=step, total=scal[0], data=scal[1], deriv=scal[2], sindy=scal[3], l1=scal[4],
                        l1_weighted=W_L1 * scal[4], grad_xi=scal[5], grad_gcn=scal[6], mom_norm=scal[7],
                        mom_cos=scal[8], n_active=int(act.sum().item()), lam_max=lam_max, kappa=kappa,
                        xi_lr=xi_lr))
        xi_path.append(to_physical((Xi_n * mask).detach().cpu().numpy()).reshape(-1))

        if step % 500 == 0 or step == JOINT_STEPS - 1:
            print(f"  step {step:5d} | total {scal[0]:.5f} | sindy {scal[3]:.6f} | "
                  f"active {int(act.sum().item())} | xi_lr {xi_lr:.3f} | cos {scal[8]:+.2f}")

    log = pd.DataFrame(log)
    xi_path = np.array(xi_path)
    log.to_csv(os.path.join(base_dir, f"pruning_log_{tag}.csv"), index=False)

    # ---------------- landscape plot (uses the 'pre'/'post' snapshots taken inside prune()) ----------------
    if PLOT_LANDSCAPE:
        if "pre" in landscape_snaps and "post" in landscape_snaps:
            final_state = {n: p.detach().clone() for n, p in model.named_parameters()}
            final_mask = mask.clone()
            fig_land = plot_landscape_pair(
                landscape_snaps["pre"], landscape_snaps["post"], LANDSCAPE_EVENT_STEP, base_dir, tag,
                model, mask, x_in_n, ADJ_T, data_n, valid_t, mu_t, sd_t, vsc_t, cs_t)
            with torch.no_grad():                                 # restore post-training state
                for n, p in model.named_parameters():
                    p.copy_(final_state[n])
                mask.copy_(final_mask)
            if not SHOW_PLOTS:
                plt.close(fig_land)
        else:
            print(f"  No landscape snapshot captured (LANDSCAPE_EVENT_STEP={LANDSCAPE_EVENT_STEP} "
                  f"was not reached or not a scheduled event in this run).")

    # ---------------- final result ----------------
    with torch.no_grad():
        _, V_n, X_phys = forward_all()
        Th_np = (library_t(X_phys) / cs_t[None, :]).cpu().numpy().astype(np.float64)
        V_np = V_n.cpu().numpy().astype(np.float64)
        X_sindy = X_phys.cpu().numpy()
    final_active = mask.cpu().numpy() > 0
    Xi = to_physical((Xi_n * mask).detach().cpu().numpy())
    Xi_refit = to_physical(refit_xi(Th_np, V_np, final_active))
    true_supp = XI_TRUE != 0

    print("\n  Identified equations (GD iterate):")
    for k, nm in enumerate(["dX/dt", "dY/dt", "dZ/dt"]):
        terms = [f"{Xi[j, k]:.4f}*{library_labels[j]}" if library_labels[j] != "1" else f"{Xi[j, k]:.4f}"
                 for j in range(P) if final_active[j, k]]
        print(f"  {nm} = " + (" + ".join(terms).replace("+ -", "- ") if terms else "0"))

    # ---------------- transient analysis ----------------
    rows = analyze(log, event_steps, event_info)
    jit = jitter_baseline(log, event_steps)
    real = rows[rows["removed"] > 0]
    print(f"\n  Non-event jitter baseline: 95th-percentile 'peak/pre' of L_sindy in quiet 50-step windows = {jit:.2f}")
    print("  (an event whose peak/pre is not clearly above this is indistinguishable from optimiser noise)")

    summary = dict(
        tag=tag, prune_mode=prune_mode, refit=refit, dropout=dropout, seed=seed,
        n_active=int(final_active.sum()),
        support_exact=bool(np.array_equal(final_active, true_supp)),
        false_pos=int((final_active & ~true_supp).sum()),
        false_neg=int((~final_active & true_supp).sum()),
        coef_err_gd=float(np.linalg.norm(Xi - XI_TRUE) / np.linalg.norm(XI_TRUE)),
        coef_err_refit=float(np.linalg.norm(Xi_refit - XI_TRUE) / np.linalg.norm(XI_TRUE)),
        mean_jump=float(real["peak_over_pre"].mean()) if len(real) else np.nan,
        mean_recovery_steps=float(real["recovery"].dropna().mean()) if len(real) and real["recovery"].notna().any() else np.nan,
        events_never_recovered=int(real["recovery"].isna().sum()) if len(real) else 0,
        jitter_p95=jit,
    )

    # ---------------- plots ----------------
    fig = make_plots(log, xi_path, event_info, event_steps, tag)
    fig_path = os.path.join(base_dir, f"pruning_diagnostics_{tag}.png")
    fig.savefig(fig_path, dpi=150)
    print(f"  Saved figure -> {fig_path}")
    if not SHOW_PLOTS:
        plt.close(fig)

    return dict(summary=summary, Xi=Xi, Xi_refit=Xi_refit, X_sindy=X_sindy, log=log, rows=rows)


# ==============================================================================
# Analysis helpers
# ==============================================================================
def analyze(log, event_steps, event_info, metric="sindy", pre_win=20, peak_win=50, tail_win=50):
    """Per-event transient table. Recovery is measured against the post-prune PLATEAU
    (mean of the last `tail_win` steps before the next event), so the permanent cost of
    removing a term is reported separately as settled/pre instead of masquerading as slow recovery."""
    m = log[metric].values
    sm = pd.Series(m).rolling(10, min_periods=1).mean().values
    ev = sorted(event_steps)
    rows = []
    print(f"\n  Transient table (metric = {metric}); recovery = steps until 10-step-smoothed metric is within "
          f"{RECOVERY_TOL:.0%} of the post-prune plateau")
    print(f"  {'step':>5} {'removed':>8} {'GD dist':>8} {'pre':>10} {'peak/pre':>9} {'settled/pre':>12} "
          f"{'recovery':>9} {'min cos':>8} {'GCN grad x':>11}")
    for i, e in enumerate(ev):
        nxt = ev[i + 1] if i + 1 < len(ev) else JOINT_STEPS
        pre = m[max(0, e - pre_win):e].mean()
        seg = m[e:nxt]
        peak = seg[:peak_win].max()
        settled = m[max(e + 1, nxt - tail_win):nxt].mean()
        ok = np.where(sm[e:nxt] <= settled * (1 + RECOVERY_TOL))[0]
        rec = int(ok[0]) if len(ok) else np.nan
        mincos = log["mom_cos"].values[e:e + 20].min()
        gpre = np.median(log["grad_gcn"].values[max(0, e - pre_win):e])
        gx = log["grad_gcn"].values[e:e + 20].max() / (gpre + 1e-12)
        info = event_info[e]
        rows.append(dict(step=e, removed=info["n_removed"], gd_dist=info["dist"], pre=pre,
                         peak_over_pre=peak / pre, settled_over_pre=settled / pre, recovery=rec,
                         min_cos=mincos, gcn_grad_x=gx))
        rec_s = f"{rec}" if not np.isnan(rec) else f">{len(seg)}"
        tag = "" if info["n_removed"] else "  (null)"
        print(f"  {e:5d} {info['n_removed']:8d} {info['dist']:8.1%} {pre:10.3e} {peak / pre:9.2f} "
              f"{settled / pre:12.2f} {rec_s:>9} {mincos:8.2f} {gx:11.1f}{tag}")
    return pd.DataFrame(rows)


def jitter_baseline(log, event_steps, metric="sindy", pre_win=20, peak_win=50):
    m = log[metric].values
    bad = np.zeros(len(m), dtype=bool)
    for e in event_steps:
        bad[max(0, e - pre_win):e + peak_win] = True
    ratios = []
    for s in range(pre_win, len(m) - peak_win, 5):
        if bad[s - pre_win:s + peak_win].any():
            continue
        ratios.append(m[s:s + peak_win].max() / m[s - pre_win:s].mean())
    return float(np.percentile(ratios, 95)) if ratios else np.nan


def make_plots(log, xi_path, event_info, event_steps, tag):
    fig, ax = plt.subplots(3, 2, figsize=(15, 11), sharex=True)
    s = log["step"]

    def mark(a):
        for e in event_steps:
            real = event_info[e]["n_removed"] > 0
            a.axvline(e, color="red" if real else "gray", ls=":", lw=0.9, alpha=0.7)

    ax[0, 0].semilogy(s, log["total"], label="total")
    ax[0, 0].semilogy(s, log["sindy"], label="L_sindy")
    ax[0, 0].semilogy(s, log["data"], label="L_data")
    ax[0, 0].semilogy(s, log["deriv"], label="L_deriv")
    if W_L1 > 0:
        ax[0, 0].semilogy(s, log["l1_weighted"], "--", label="W_L1 * L1")
    ax[0, 0].set_title("Loss components  (red dotted = real prune, gray = null event)")
    ax[0, 0].legend(); mark(ax[0, 0])

    ax[0, 1].semilogy(s, log["grad_xi"], label="||grad Xi||")
    ax[0, 1].semilogy(s, log["grad_gcn"], label="||grad GCN||")
    ax[0, 1].set_title("Gradient norms"); ax[0, 1].legend(); mark(ax[0, 1])

    ax[1, 0].plot(s, log["mom_norm"], label="||momentum buffer||")
    ax[1, 0].set_title("Momentum buffer norm (Xi)"); ax[1, 0].legend(); mark(ax[1, 0])

    ax[1, 1].plot(s, log["mom_cos"], lw=0.8)
    ax[1, 1].axhline(0, color="r", lw=0.8)
    ax[1, 1].set_title("cos(prev. momentum, gradient)  (<0 = momentum fighting the gradient)")
    ax[1, 1].set_ylim(-1.05, 1.05); mark(ax[1, 1])

    ax[2, 0].semilogy(s, log["lam_max"], label="lambda_max of Xi-Hessian")
    ax[2, 0].semilogy(s, log["kappa"], label="condition number")
    ax[2, 0].semilogy(s, 1.0 / log["xi_lr"], "g--", lw=0.9, label="1 / (Xi lr used)")
    ax[2, 0].set_title("Curvature of the Xi sub-problem (Xi lr is safe when 1/lr > ~0.75 * lambda_max)")
    ax[2, 0].legend(); mark(ax[2, 0]); ax[2, 0].set_xlabel("joint step")

    ax[2, 1].plot(s, xi_path, lw=0.7, alpha=0.8)
    ax[2, 1].set_yscale("symlog", linthresh=0.1)
    ax2 = ax[2, 1].twinx()
    ax2.step(s, log["n_active"], color="k", lw=1.2, where="post")
    ax2.set_ylabel("# active terms (black)")
    ax[2, 1].set_title("Physical coefficients (colored) & active-term count")
    ax[2, 1].set_xlabel("joint step"); mark(ax[2, 1])

    fig.suptitle(f"Pruning-transient diagnostics | {tag}", y=1.0)
    fig.tight_layout()
    return fig


# ==============================================================================
# Experiments to run
# ==============================================================================
if __name__ == "__main__":
    # Default: same seed, stale momentum vs reset, with refit-before-threshold.
    RUNS = [
        dict(prune_mode="keep",  refit=True, dropout=0.0, seed=3),
        dict(prune_mode="reset", refit=False, dropout=0.0, seed=3),
    ]
    # Ablation of the refit itself (reproduces the v1 failure mode):
    #   dict(prune_mode="keep", refit=False, dropout=0.0, seed=0)
    # Full sweep (set SHOW_PLOTS = False at the top first):
    #   RUNS = [dict(prune_mode=m, refit=True, dropout=d, seed=s)
    #           for m in ("keep", "reset") for d in (0.0, 0.05, 0.10) for s in range(3)]

    results = [run(**cfg) for cfg in RUNS]

    print("\n" + "=" * 78 + "\nSUMMARY ACROSS RUNS\n" + "=" * 78)
    summ = pd.DataFrame([r["summary"] for r in results]).drop(columns=["tag"])
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    summ.to_csv(os.path.join(base_dir, "pruning_summary.csv"), index=False)

    # Names match your old sections 7-8, so their forecasting code can be pasted below unchanged
    # (Xi_refit is usually the better one to forecast with).
    Xi = results[-1]["Xi"]
    X_sindy = results[-1]["X_sindy"]

    if SHOW_PLOTS:
        plt.show()