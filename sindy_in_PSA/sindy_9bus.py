"""
SINDy + MBAM multi-bus fault analysis (IEEE 9-bus)
==================================================
Workflow
  1. Ingest generator files (angle, speed, Pm, field) and bus files (3-phase V, I) -> Fortescue sequence phasors.
  2. Resample everything to one uniform grid, 4th-order central-difference derivatives.
  3. SINDy: candidate library Theta(X) (linear, cross-bus coupling, sequence ratios, trig angle differences),
     STLSQ fitted ONLY on the healthy pre-fault window.  Residual spikes + coefficient shifts => fault inception.
  4. MBAM-style manifold reduction: Fisher-information eigen-analysis per equation, iterative removal of the
     "sloppiest" parameter (smallest FIM eigen-direction) until the model is well conditioned.
  5. Fault type (V2/V1, V0/V1, lowest phase) and proximity localisation (sag gradient, V0 prominence, residual energy).
  6. Plots.

HOW TO USE (VS Code): edit only the CONFIG block, then Run.   pip install numpy pandas scipy matplotlib openpyxl
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# ================================ CONFIG: EDIT THIS ================================
# Paste file paths between the quotes ( r"..." on Windows ).  .csv or .xlsx both work.
# Buses: any buses in the network.  Add / remove freely.   {bus number: path}
BUS_FILES = {
    1: r"/Users/Arya/Desktop/vs_code/sindy_in_PSA/IVPQ_B1_Fag.csv",
    4: r"/Users/Arya/Desktop/vs_code/sindy_in_PSA/IVPQ_B4_Fag.csv",
    5: r"/Users/Arya/Desktop/vs_code/sindy_in_PSA/IVPQ_B5_Fag.csv",
}
# Generators: {generator number: path}.  Use {} to run bus-only.
GEN_FILES = {
    1: r"/Users/Arya/Desktop/vs_code/sindy_in_PSA/G1_Fag.csv",
    2: r"/Users/Arya/Desktop/vs_code/sindy_in_PSA/G2_Fag.csv",
    3: r"/Users/Arya/Desktop/vs_code/sindy_in_PSA/G3_Fag.csv",
}
# generator k is electrically attached to bus GEN_BUS[k]  (used for the delta - theta library terms)
GEN_BUS = {1: 1, 2: 2, 3: 3}

BUS_COLS = dict(t="t", Va="Va", Vb="Vb", Vc="Vc", Ia="Ia", Ib="Ib", Ic="Ic")
GEN_COLS = dict(t="t", delta="theta", omega="omega", Pm="Pm_pu", fd="fd_p")
GEN_ANGLE_UNIT = "deg"          # "deg" or "rad" (your theta behaves as degrees)
GEN_POLE_PAIRS = {}             # e.g. {1: 20}; empty -> auto = round(2*pi*F0 / omega(0)); G1 -> 20, G2/G3 -> 1

F0 = 60.0                       # system frequency (Hz)
DECIM = 2                       # keep every DECIM-th phasor sample (grid = 1440/DECIM Hz for your logging rate)
HEALTHY_WINDOW = (0.10, 0.29)   # clean pre-fault data (s): used to FIT the nominal model
TRAIN_FRAC = 0.7                # first part of the healthy window = fit, rest = held-out (calibrates residual scale)
STLS_THRESHOLD = 0.05           # sparsity threshold on normalised coefficients
RIDGE = 1e-4                    # ridge stabiliser (library terms are strongly collinear)
DETECT_Z = 6.0                  # alarm when RMS normalised residual exceeds this ...
DETECT_CONSEC = 3               # ... for this many consecutive samples
V2_ALARM = 0.05                 # |V2|/|V1| level that defines the fault window for the summary table
MBAM_EIG_RATIO = 1e-3           # prune until lambda_min/lambda_max of the FIM exceeds this
COEF_WIN = 0.08                 # s, sliding window for coefficient-shift tracking
RESIDUAL_LOC_WINDOW = 0.10      # s after inception over which per-bus residual energy is measured
LOW_PHASE_PU = 0.75             # a phase below this fraction of its pre-fault voltage counts as "low"
SAVE_FIGS = ("fault_overview.png", "sindy_mbam_details.png")   # None to skip
# ====================================================================================

a = np.exp(2j * np.pi / 3)
seq = lambda A, B, C: ((A + B + C) / 3, (A + a * B + a * a * C) / 3, (A + a * a * B + a * C) / 3)


# ------------------------------------ 1. INGESTION ------------------------------------
def read_table(path, cols):
    d = pd.read_excel(path) if str(path).lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    d = d.rename(columns={v: k for k, v in cols.items() if v in d.columns})
    missing = [k for k in cols if k not in d.columns]
    if missing:
        raise KeyError(f"{path}: missing columns {missing}; found {list(d.columns)}. Edit BUS_COLS / GEN_COLS.")
    return d.drop_duplicates("t").sort_values("t").reset_index(drop=True)


def bus_phasors(d):
    """One-cycle DFT in the rotating F0 frame: V_phasor(t) rms, time stamp at the window centre."""
    t = d["t"].values
    n = int(round(1 / (F0 * np.median(np.diff(t)))))
    fs = n * F0
    tu = np.arange(t.min(), t.max() + 1e-12, 1 / fs)
    w = np.exp(-2j * np.pi * np.arange(n) / n)[::-1] * np.sqrt(2) / n
    P = {}
    for c in ["Va", "Vb", "Vc", "Ia", "Ib", "Ic"]:
        x = np.interp(tu, t, d[c].values)
        P[c] = np.convolve(x, w)[: len(tu)] * np.exp(-2j * np.pi * F0 * tu)   # remove carrier rotation
    return tu - (n - 1) / 2 / fs, P, fs


print("Loading bus files ...")
BUS = {}
for k, f in BUS_FILES.items():
    tc, P, fs = bus_phasors(read_table(f, BUS_COLS))
    V0, V1, V2 = seq(P["Va"], P["Vb"], P["Vc"])
    I0, I1, I2 = seq(P["Ia"], P["Ib"], P["Ic"])
    BUS[k] = dict(tc=tc, P=P, V0=V0, V1=V1, V2=V2, I0=I0, I1=I1, I2=I2)

# common uniform grid (first bus, decimated, one full cycle in)
k0 = next(iter(BUS))
sel = np.where(BUS[k0]["tc"] >= 1.0 / F0)[0][::DECIM]
t = BUS[k0]["tc"][sel]
dt = float(np.median(np.diff(t)))
FS_COMMON = 1 / dt
print(f"  common grid: {FS_COMMON:.0f} Hz, {len(t)} samples, t = {t[0]:.3f} .. {t[-1]:.3f} s")


def on_grid(tc, z):
    if np.iscomplexobj(z):
        return np.interp(t, tc, z.real) + 1j * np.interp(t, tc, z.imag)
    return np.interp(t, tc, z)


for k, s in BUS.items():
    for key in ["V0", "V1", "V2", "I0", "I1", "I2"]:
        s[key] = on_grid(s["tc"], s[key])
    s["P"] = {c: on_grid(s["tc"], s["P"][c]) for c in s["P"]}

pre = (t >= HEALTHY_WINDOW[0]) & (t <= HEALTHY_WINDOW[1])
assert pre.sum() > 20, "HEALTHY_WINDOW contains too few samples"

print("Loading generator files ...")
GEN = {}
for k, f in GEN_FILES.items():
    d = read_table(f, GEN_COLS)
    w0 = d["omega"].iloc[0]
    p = GEN_POLE_PAIRS.get(k, max(1, int(round(2 * np.pi * F0 / w0))))
    ang = np.deg2rad(d["delta"].values) if GEN_ANGLE_UNIT == "deg" else d["delta"].values
    ang = np.unwrap(ang) * p                                   # electrical angle (rad)
    tg = d["t"].values
    cub = lambda y: PchipInterpolator(tg, y, extrapolate=True)(t)
    GEN[k] = dict(delta=cub(ang), omega=cub(d["omega"].values / w0), Pm=cub(d["Pm"].values), fd=cub(d["fd"].values),
                  n_native=len(tg), fs_native=1 / np.median(np.diff(tg)), p=p)
    print(f"  G{k}: pole pairs={p}, native rate={GEN[k]['fs_native']:.0f} Hz ({len(tg)} samples)")
if GEN:
    nat = min(g["fs_native"] for g in GEN.values())
    n_pre_native = int(nat * (HEALTHY_WINDOW[1] - HEALTHY_WINDOW[0]))
    if n_pre_native < 30:
        print(f"  WARNING: generator data is logged at {nat:.0f} Hz -> only ~{n_pre_native} real samples in the healthy window.\n"
              f"           Generator equations are interpolated; log generators at >=600 Hz for a trustworthy SINDy fit.")


# --------------------------- 2. STATES, DERIVATIVES, LIBRARY ---------------------------
def d4(x, h):
    """4th-order central difference (np.gradient at the 2 edge samples on each side)."""
    g = np.gradient(x, h)
    out = g.copy()
    out[2:-2] = (-x[4:] + 8 * x[3:-1] - 8 * x[1:-3] + x[:-4]) / (12 * h)
    return out


states, groups = {}, {}            # groups: which physical element each target belongs to
for k, g in GEN.items():
    for key in ["delta", "omega", "Pm", "fd"]:
        nm = f"{key}{k}"; states[nm] = g[key]; groups[nm] = f"G{k}"
for k, s in BUS.items():
    v1, i1 = np.abs(s["V1"]), np.abs(s["I1"])
    states[f"Vm{k}"] = v1 / v1[pre].mean();  groups[f"Vm{k}"] = f"B{k}"
    states[f"Im{k}"] = i1 / i1[pre].mean();  groups[f"Im{k}"] = f"B{k}"
    th = np.unwrap(np.angle(s["V1"])); states[f"th{k}"] = th - th[pre].mean(); groups[f"th{k}"] = f"B{k}"
aux = {}
for k, s in BUS.items():
    aux[f"r2_{k}"] = np.abs(s["V2"]) / np.abs(s["V1"])
    aux[f"r0_{k}"] = np.abs(s["V0"]) / np.abs(s["V1"])

target_names = list(states)
Xdot = np.column_stack([d4(states[n], dt) for n in target_names])


def build_library():
    cols, names = {}, []
    def add(n, v): cols[n] = np.asarray(v, float); names.append(n)
    for n in target_names: add(n, states[n])                                   # linear terms
    for n, v in aux.items(): add(n, v)                                         # sequence ratios V2/V1, V0/V1
    bl = list(BUS)
    for i in range(len(bl)):                                                   # cross-bus voltage coupling
        for j in range(i + 1, len(bl)):
            a_, b_ = bl[i], bl[j]
            dth = np.unwrap(np.angle(BUS[a_]["V1"])) - np.unwrap(np.angle(BUS[b_]["V1"]))
            m = states[f"Vm{a_}"] * states[f"Vm{b_}"]
            add(f"Vm{a_}Vm{b_}cos(th{a_}-th{b_})", m * np.cos(dth)); add(f"Vm{a_}Vm{b_}sin(th{a_}-th{b_})", m * np.sin(dth))
    gl = list(GEN)
    for i in range(len(gl)):                                                   # generator angle differences
        for j in range(i + 1, len(gl)):
            dd = GEN[gl[i]]["delta"] - GEN[gl[j]]["delta"]
            add(f"sin(d{gl[i]}-d{gl[j]})", np.sin(dd)); add(f"cos(d{gl[i]}-d{gl[j]})", np.cos(dd))
    for gk, bk in GEN_BUS.items():                                             # rotor angle vs terminal-bus angle
        if gk in GEN and bk in BUS:
            dd = GEN[gk]["delta"] - np.unwrap(np.angle(BUS[bk]["V1"]))
            add(f"sin(d{gk}-th{bk})", np.sin(dd)); add(f"cos(d{gk}-th{bk})", np.cos(dd))
    return names, np.column_stack([cols[n] for n in names])


lib_names, Theta = build_library()
n_lib = len(lib_names)
print(f"Library: {n_lib} candidate terms, {len(target_names)} state equations")

# ------------------------------ 3. SINDy ON THE HEALTHY WINDOW ------------------------------
idx_pre = np.where(pre)[0]
n_tr = int(TRAIN_FRAC * len(idx_pre))
tr, ho = idx_pre[:n_tr], idx_pre[n_tr:]
mu = Theta[tr].mean(0); sd = Theta[tr].std(0)
valid = sd > 1e-9                                  # columns that vary in healthy data (V2/V1, V0/V1 are ~0 -> dropped)
sd[~valid] = 1.0
Z = (Theta - mu) / sd                              # standardised library (valid columns only used)
Z = np.column_stack([np.ones(len(t)), Z])          # intercept column 0
vcols = np.concatenate([[True], valid])

full_std = Xdot.std(0)
eq_ok = full_std > 1e-9
ys = np.maximum(Xdot[tr].std(0), 0.01 * full_std); ys[~eq_ok] = 1.0
ymu = Xdot[tr].mean(0)
Yn = (Xdot - ymu) / ys


def ridge(A, Y):
    G = A.T @ A + RIDGE * len(A) * np.eye(A.shape[1])
    return np.linalg.solve(G, A.T @ Y)


def stls(A, y, support):
    s = support.copy()
    for _ in range(15):
        c = np.zeros(A.shape[1])
        if s.any(): c[s] = ridge(A[:, s], y)
        keep = s & (np.abs(c) >= STLS_THRESHOLD)
        if keep.sum() == s.sum(): break
        s = keep
    if s.any(): c[s] = ridge(A[:, s], y)
    return c, s


Xi = np.zeros((Z.shape[1], len(target_names)))
supp = np.zeros_like(Xi, dtype=bool)
for j in range(len(target_names)):
    if eq_ok[j]:
        Xi[:, j], supp[:, j] = stls(Z[tr], Yn[tr, j], vcols.copy())


# ------------------------------ 4. MBAM-STYLE MANIFOLD REDUCTION ------------------------------
def fim_spectrum(A):
    F = A.T @ A / len(A)                                  # Fisher information (Gaussian residual, unit noise)
    return np.sort(np.linalg.eigvalsh(F))[::-1]


Xi_m = np.zeros_like(Xi); supp_m = np.zeros_like(supp)
spec_before, spec_after, n_pruned = [], [], 0
for j in range(len(target_names)):
    if not eq_ok[j] or not supp[:, j].any(): continue
    s = supp[:, j].copy()
    spec_before.append(fim_spectrum(Z[tr][:, s]))
    while s.sum() > 1:
        idx = np.where(s)[0]
        F = Z[tr][:, idx].T @ Z[tr][:, idx] / len(tr)
        w, v = np.linalg.eigh(F)
        if w[0] / w[-1] > MBAM_EIG_RATIO: break
        drop = idx[np.argmax(np.abs(v[:, 0]))]            # parameter dominating the sloppiest direction
        s[drop] = False; n_pruned += 1
    c = np.zeros(Z.shape[1]); c[s] = ridge(Z[tr][:, s], Yn[tr, j])
    Xi_m[:, j], supp_m[:, j] = c, s
    spec_after.append(fim_spectrum(Z[tr][:, s]))
print(f"SINDy active terms: {int(supp.sum())}  ->  after MBAM pruning: {int(supp_m.sum())}  ({n_pruned} sloppy parameters removed)")


def residual_z(Xi_):
    pred = (Z @ Xi_) * ys + ymu
    res = Xdot - pred
    sig = np.maximum(res[ho].std(0), 0.01 * full_std)          # held-out healthy residual scale (+ floor)
    sig[~eq_ok] = np.inf
    return np.abs(res) / sig


Zres, Zres_m = residual_z(Xi), residual_z(Xi_m)
ok_idx = np.where(eq_ok)[0]
R, R_m = np.sqrt((Zres[:, ok_idx] ** 2).mean(1)), np.sqrt((Zres_m[:, ok_idx] ** 2).mean(1))


def first_alarm(r):
    hot = (r > DETECT_Z) & (t > HEALTHY_WINDOW[1] - 1e-9)
    for i in range(len(hot) - DETECT_CONSEC + 1):
        if hot[i:i + DETECT_CONSEC].all(): return t[i]
    return np.nan


t_det, t_det_m = first_alarm(R), first_alarm(R_m)
HALF = 0.5 / F0   # phasors are stamped at the window centre; add half a cycle for the causal (real-time) alarm time
print(f"SINDy residual alarm: window-centre t = {t_det:.3f} s (causal ~{t_det + HALF:.3f} s) | "
      f"MBAM-reduced model: {t_det_m:.3f} s (causal ~{t_det_m + HALF:.3f} s) | healthy held-out RMS z = {R[ho].mean():.2f}")
print("First alarm per element (MBAM-reduced model, window-centre time):")
for g in sorted(set(groups.values())):
    c_ = [i for i, n in enumerate(target_names) if groups[n] == g and eq_ok[i]]
    if c_:
        rg = np.sqrt((Zres_m[:, c_] ** 2).mean(1)); ta = first_alarm(rg)
        print(f"   {g}: {ta:.3f} s" if np.isfinite(ta) else f"   {g}: no alarm")

# coefficient shift: refit baseline support on sliding windows, compare with baseline Xi
shift_t, shift_v = [], []
nw = int(COEF_WIN / dt)
for i0 in range(0, len(t) - nw, max(1, nw // 4)):
    sl = slice(i0, i0 + nw); num = den = 0.0
    for j in ok_idx:
        s = supp_m[:, j]
        if s.sum() == 0: continue
        c = np.zeros(Z.shape[1]); c[s] = ridge(Z[sl][:, s], Yn[sl, j])
        num += np.sum((c - Xi_m[:, j]) ** 2); den += np.sum(Xi_m[:, j] ** 2)
    shift_t.append(t[i0 + nw // 2]); shift_v.append(np.sqrt(num / max(den, 1e-12)))
shift_t, shift_v = np.array(shift_t), np.array(shift_v)

# ------------------------------ 5. CLASSIFICATION & LOCALISATION ------------------------------
alarm_any = np.zeros(len(t), bool)
for s in BUS.values():
    alarm_any |= (np.abs(s["V2"]) / np.abs(s["V1"]) > V2_ALARM) | (np.abs(s["V0"]) / np.abs(s["V1"]) > 0.005)
alarm_any &= t > HEALTHY_WINDOW[1] - 0.02
ia = np.where(alarm_any)[0]
if len(ia) == 0:
    print("\nNo fault found: V2/V1 and V0/V1 stay below their alarm levels at all buses."); raise SystemExit
cyc = int(round(FS_COMMON / F0))
fa, fb = ia[0] + cyc, ia[-1] - cyc                  # trim one cycle each end (phasor window transients)
flt = np.zeros(len(t), bool); flt[fa:fb + 1] = True
print(f"\nFault window from sequence alarm: {t[ia[0]]:.3f} .. {t[ia[-1]]:.3f} s (summary uses {t[fa]:.3f} .. {t[fb]:.3f} s)")
t_on = t_det_m if np.isfinite(t_det_m) else (t_det if np.isfinite(t_det) else t[ia[0]])

rows = []
for k, s in BUS.items():
    v1 = np.abs(s["V1"]); p0 = v1[pre].mean()
    pu = np.array([np.abs(s["P"]["V" + c])[flt].mean() / np.abs(s["P"]["V" + c])[pre].mean() for c in "abc"])
    r0 = np.abs(s["V0"])[flt].mean() / p0
    wr = (t >= t_on) & (t <= t_on + RESIDUAL_LOC_WINDOW)
    cols = [i for i, n in enumerate(target_names) if groups[n] == f"B{k}" and eq_ok[i]]
    rows.append(dict(bus=k, V1_sag_pct=100 * (1 - v1[flt].mean() / p0), low_phase="abc"[pu.argmin()],
                     low_phase_pu=pu.min(), n_low=int((pu < LOW_PHASE_PU).sum()),
                     V2_over_V1=np.abs(s["V2"])[flt].mean() / v1[flt].mean(), V0_over_V1=r0,
                     sees_V0=bool(r0 > 0.005), resid_z=Zres_m[wr][:, cols].mean() if cols else np.nan))
T = pd.DataFrame(rows).set_index("bus")
pd.set_option("display.width", 220); pd.set_option("display.float_format", lambda x: f"{x:.4g}")
print("\n=== Per-bus features (fault window) ===\n" + T.to_string())

# fault type (buses that can see zero sequence are used for the phase count when available)
ref = T[T.sees_V0] if T.sees_V0.any() else T
n_low = int(ref.n_low.mode()[0]); lowph = ref.low_phase.mode()[0].upper()
ground = bool(T.sees_V0.any())
r2 = T.V2_over_V1.max()
if n_low >= 3 and r2 < 0.10: ftype = "ABC (three-phase)"
elif n_low == 2:
    pu_all = {c: np.mean([np.abs(BUS[k]["P"]["V" + c])[flt].mean() / np.abs(BUS[k]["P"]["V" + c])[pre].mean() for k in ref.index]) for c in "abc"}
    two = "".join(sorted(c.upper() for c, v in pu_all.items() if v < LOW_PHASE_PU)); ftype = f"{two}{'G' if ground else ''} ({'double line-to-ground' if ground else 'line-to-line'})"
else:
    ftype = f"{lowph}-G (single line-to-ground)" if ground else f"{lowph} low, no zero sequence seen (check: unusual)"
if not ground: ftype += "  [no bus sees zero sequence: ground involvement undetermined]" if n_low <= 1 else ""
print(f"\nFAULT TYPE      : {ftype}")
print(f"  evidence      : phases below {LOW_PHASE_PU} pu: {n_low}; max |V2|/|V1| = {r2:.2f}; "
      f"zero sequence seen at buses {list(T.index[T.sees_V0])}; lowest phase = {lowph}")

# proximity ranking.  Only buses that can be compared are ranked: if >= 2 buses see zero sequence, buses behind a delta
# winding (V0 blind, voltages phase-shifted) are listed separately instead of being scored against the HV buses.
comparable = T[T.sees_V0] if T.sees_V0.sum() >= 2 else T
metrics = pd.DataFrame({"V1 sag %": comparable.V1_sag_pct, "low-phase depth": 1 - comparable.low_phase_pu})
if comparable.sees_V0.all(): metrics["V0/V1"] = comparable.V0_over_V1
score = (metrics / metrics.max()).mean(axis=1)           # each metric as a fraction of the largest value among buses
order = score.sort_values(ascending=False)
closest = int(order.index[0])
votes = metrics.idxmax()
print("\n=== Proximity ranking among comparable buses (1 = closest) ===")
for r, (b, v) in enumerate(order.items(), 1):
    print(f"  {r}. Bus {b}   score {v:.2f}")
print("  metric leaders:", {m: int(b) for m, b in votes.items()})
blind = [int(b) for b in T.index if b not in comparable.index]
if blind: print(f"  not ranked (zero-sequence blind, behind a delta winding): buses {blind}")
agree = int((votes == closest).sum())
if len(order) > 1 and "V0/V1" in metrics:
    ratio = metrics.loc[order.index[0], "V0/V1"] / metrics.loc[order.index[1], "V0/V1"]
    conf = "HIGH" if (agree == len(metrics.columns) and ratio >= 2) else ("MEDIUM" if agree == len(metrics.columns) else "LOW")
    if len(order) < 3 and conf == "HIGH": conf = "MEDIUM"      # two comparable buses cannot rule out a fault beyond them
    print(f"CLOSEST BUS     : Bus {closest}   confidence {conf}  ({agree}/{len(metrics.columns)} metrics agree; "
          f"V0/V1 is {ratio:.1f}x that of Bus {order.index[1]})")
else:
    print(f"CLOSEST BUS     : Bus {closest}   confidence LOW (only {len(order)} comparable bus(es); sag gradient only)")
print("Caveats: only monitored buses are ranked - the fault may lie beyond them (add more buses to confirm); "
      "the V0 gradient is the strongest indicator.")

# ------------------------------------------ 6. PLOTS ------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
sh = lambda a_: (a_.axvspan(t[fa], t[fb], color="r", alpha=0.08), a_.grid(alpha=0.3))
for k, s in BUS.items():
    ax[0, 0].plot(t, np.abs(s["V1"]) / np.abs(s["V1"])[pre].mean(), label=f"Bus {k}")
    ax[0, 1].plot(t, 100 * np.abs(s["V2"]) / np.abs(s["V1"]), label=f"|V2|/|V1| Bus {k}")
    ax[0, 1].plot(t, 100 * np.abs(s["V0"]) / np.abs(s["V1"]), "--", label=f"|V0|/|V1| Bus {k}")
ax[0, 0].set_title("Positive-sequence voltage"); ax[0, 0].set_ylabel("|V1| (pu of pre-fault)")
ax[0, 1].set_title("Sequence ratios (solid V2, dashed V0)"); ax[0, 1].set_ylabel("%")
ax[1, 0].semilogy(t, R + 1e-3, label="SINDy"); ax[1, 0].semilogy(t, R_m + 1e-3, label="MBAM-reduced")
ax[1, 0].axhline(DETECT_Z, color="k", ls=":", label=f"alarm = {DETECT_Z}")
if np.isfinite(t_det_m): ax[1, 0].axvline(t_det_m, color="g", ls="--", label=f"alarm {t_det_m:.3f} s")
ax[1, 0].axvspan(*HEALTHY_WINDOW, color="g", alpha=0.08); ax[1, 0].set_title("SINDy residual (RMS normalised error, green = fit window)")
ax[1, 0].set_xlabel("t (s)"); ax[1, 0].set_ylabel("residual z")
cols_ = ["tab:red" if b == closest else "tab:blue" for b in order.index]
ax[1, 1].bar([f"Bus {b}" for b in order.index], order.values, color=cols_); ax[1, 1].set_title(f"Proximity score (closest: Bus {closest})")
ax[1, 1].set_ylabel("score (higher = closer)")
for a_ in (ax[0, 0], ax[0, 1], ax[1, 0]): sh(a_); a_.legend(fontsize=7, loc="best")
plt.tight_layout()
if SAVE_FIGS: plt.savefig(SAVE_FIGS[0], dpi=110)

fig2, bx = plt.subplots(1, 3, figsize=(15, 4.2))
for sp in spec_before: bx[0].semilogy(np.arange(1, len(sp) + 1), sp + 1e-16, color="tab:red", alpha=0.35)
for sp in spec_after: bx[0].semilogy(np.arange(1, len(sp) + 1), sp + 1e-16, color="tab:green", alpha=0.5)
bx[0].axhline(MBAM_EIG_RATIO, color="k", ls=":"); bx[0].set_title("FIM eigenvalues per equation\n(red: SINDy, green: after MBAM pruning)")
bx[0].set_xlabel("eigen-direction"); bx[0].set_ylabel("eigenvalue (normalised)")
bx[1].semilogy(shift_t, shift_v + 1e-6); bx[1].axvspan(t[fa], t[fb], color="r", alpha=0.08); bx[1].set_title("Coefficient shift ||Xi_win - Xi_0|| / ||Xi_0||")
bx[1].set_xlabel("t (s)")
for g in sorted(set(groups.values())):
    c_ = [i for i, n in enumerate(target_names) if groups[n] == g and eq_ok[i]]
    if c_: bx[2].semilogy(t, np.sqrt((Zres_m[:, c_] ** 2).mean(1)) + 1e-3, label=g)
bx[2].axhline(DETECT_Z, color="k", ls=":"); bx[2].axvspan(t[fa], t[fb], color="r", alpha=0.08)
bx[2].set_title("Residual by element (MBAM-reduced)"); bx[2].legend(fontsize=7, ncol=2); bx[2].set_xlabel("t (s)")
for a_ in bx: a_.grid(alpha=0.3)
plt.tight_layout()
if SAVE_FIGS: plt.savefig(SAVE_FIGS[1], dpi=110)
plt.show()