"""
IEEE 9-bus healthy-baseline SINDy prototype (Individual File Configuration)
===========================================================================
Use individually specified healthy run files ( *_H.csv ) to fit small robust
SINDy-style generator models, and fault run files ( *_Fag.csv / xlsx ) to
evaluate model residuals and bus-level voltage/sequence indicators.

HOW TO USE: Edit the file paths in the CONFIG block below with your absolute
or relative file paths.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import savgol_filter

# ------------------------------- CONFIG: EDIT PATHS HERE ----------------------------------
# Paste or type the exact path to each individual file between the quotes (use r"..." on Windows).
HEALTHY_GENERATORS = {
    1: r"/Users/Arya/Desktop/vs_code/sindy_in_PSA/healthy datasets/G1_H.csv",
    2: r"/Users/Arya/Desktop/vs_code/sindy_in_PSA/healthy datasets/G2_H.csv",
    3: r"/Users/Arya/Desktop/vs_code/sindy_in_PSA/healthy datasets/G3_H.csv"
}

FAULT_GENERATORS = {
    1: r"/Users/Arya/Desktop/vs_code/sindy_in_PSA/fault datasets/G1_Fag.csv",
    2: r"/Users/Arya/Desktop/vs_code/sindy_in_PSA/fault datasets/G2_Fag.csv",
    3: r"/Users/Arya/Desktop/vs_code/sindy_in_PSA/fault datasets/G3_Fag.csv"
}

HEALTHY_BUSES = {
    1: r"/Users/Arya/Desktop/vs_code/sindy_in_PSA/fault datasets/IVPQ_B1_Fag.csv",
    4: r"/Users/Arya/Desktop/vs_code/sindy_in_PSA/healthy datasets/IVPQ_B4_H.csv",
    5: r"/Users/Arya/Desktop/vs_code/sindy_in_PSA/healthy datasets/IVPQ_B5_H.csv"
}

FAULT_BUSES = {
    1: r"/Users/Arya/Desktop/vs_code/sindy_in_PSA/fault datasets/IVPQ_B1_Fag.csv",
    4: r"/Users/Arya/Desktop/vs_code/sindy_in_PSA/fault datasets/IVPQ_B4_Fag.csv",
    5: r"/Users/Arya/Desktop/vs_code/sindy_in_PSA/fault datasets/IVPQ_B5_Fag.csv"
}

F0 = 60.0 
GEN_ANGLE_UNIT = 'deg'       # theta in supplied CSVs appears to be degrees
TRAIN_FRACTION = 0.70        # first 70% of healthy run used for SINDy fit
STLS_THRESHOLD = 0.08        # threshold in normalized-response coordinates
HUBER_DELTA = 1.5
RIDGE = 1e-4
FIM_EIG_RATIO = 1e-7          # heuristic pruning only, not formal MBAM
HEALTHY_BASE_WINDOW = (0.10, 0.28)
FAULT_EVAL_WINDOW = (0.32, 0.48)  # trimmed interior for classification summaries
V1_SAG_ALARM = 0.15
V2_RATIO_FLOOR = 0.08
V0_RATIO_FLOOR = 0.005
DETECT_CONSEC_PHASOR_SAMPLES = 6
FAULT_START_TRUE = 0.30
FAULT_END_TRUE = 0.50
SAVE_PREFIX = 'ieee9_healthy_baseline'
OUTPUT_DIR = Path(__file__).resolve().parent
# ------------------------------------------------------------------------------------------

BUS_COLS = ['t', 'Va', 'Vb', 'Vc', 'Ia', 'Ib', 'Ic']
GEN_COLS = ['t', 'fd_p', 'Pm_pu', 'theta', 'omega']


def read_data_file(path, required):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'File not found: {path.resolve()}')
    
    # Support both CSV and Excel spreadsheet formats seamlessly
    if path.suffix.lower() in ['.xls', '.xlsx']:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
        
    n0 = len(df)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'{path}: missing columns {missing}; found {list(df.columns)}')
    df = df[required].copy()
    df = df.dropna(subset=['t']).drop_duplicates('t').sort_values('t').reset_index(drop=True)
    if len(df) < n0:
        print(f'WARNING: {path.name}: removed {n0-len(df)} rows with missing/duplicate timestamps.')
    for c in required:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def robust_scale(x):
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    return 1.4826 * np.nanmedian(np.abs(x - med))


def robust_huber_irls(A, y, delta=1.5, ridge=1e-4, iterations=40):
    """Huber IRLS with ridge stabilization; A columns should be scaled."""
    A = np.asarray(A, float)
    y = np.asarray(y, float)
    weights = np.ones(len(y))
    coef = np.zeros(A.shape[1])
    eye = np.eye(A.shape[1])
    penalty = eye.copy()
    penalty[0, 0] = 0.0
    for _ in range(iterations):
        sw = np.sqrt(weights)
        Aw = A * sw[:, None]
        yw = y * sw
        coef_new = np.linalg.solve(Aw.T @ Aw + ridge * penalty, Aw.T @ yw)
        r = y - A @ coef_new
        s = robust_scale(r) + 1e-10
        u = np.abs(r) / (delta * s)
        weights_new = np.ones_like(u)
        big = u > 1
        weights_new[big] = 1.0 / u[big]
        if np.linalg.norm(coef_new - coef) <= 1e-8 * (1 + np.linalg.norm(coef)):
            coef = coef_new
            break
        coef, weights = coef_new, weights_new
    return coef


def fit_robust_stlsq(X, y, threshold=0.08):
    """Small-library robust STLSQ. Stores scalers for out-of-sample prediction."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    x_mu = np.mean(X, axis=0)
    x_sd = np.std(X, axis=0)
    valid = x_sd > 1e-9
    x_sd[~valid] = 1.0
    Z = (X - x_mu) / x_sd
    Z[:, ~valid] = 0.0
    A = np.column_stack([np.ones(len(Z)), Z])
    y_mu = float(np.median(y))
    y_sd = max(robust_scale(y), 1e-7)
    yn = (y - y_mu) / y_sd

    support = np.ones(A.shape[1], dtype=bool)
    support[1:][~valid] = False
    for _ in range(15):
        ids = np.flatnonzero(support)
        if len(ids) == 0:
            support[0] = True
            ids = np.array([0])
        c = robust_huber_irls(A[:, ids], yn, HUBER_DELTA, RIDGE)
        keep = np.abs(c) >= threshold
        if 0 in ids:
            keep[np.where(ids == 0)[0][0]] = True
        if not keep.any():
            keep[np.argmax(np.abs(c))] = True
        new_support = np.zeros_like(support)
        new_support[ids[keep]] = True
        if np.array_equal(new_support, support):
            break
        support = new_support
    ids = np.flatnonzero(support)
    c = robust_huber_irls(A[:, ids], yn, HUBER_DELTA, RIDGE)
    coef = np.zeros(A.shape[1])
    coef[ids] = c
    return {
        'x_mu': x_mu, 'x_sd': x_sd, 'valid': valid,
        'y_mu': y_mu, 'y_sd': y_sd, 'coef': coef,
        'support': support, 'feature_count': X.shape[1]
    }


def predict_model(model, X):
    X = np.asarray(X, float)
    Z = (X - model['x_mu']) / model['x_sd']
    Z[:, ~model['valid']] = 0.0
    A = np.column_stack([np.ones(len(Z)), Z])
    return (A @ model['coef']) * model['y_sd'] + model['y_mu']


def fim_guided_prune(model, Xtrain, ytrain):
    Xtrain = np.asarray(Xtrain, float)
    Z = (Xtrain - model['x_mu']) / model['x_sd']
    Z[:, ~model['valid']] = 0.0
    A = np.column_stack([np.ones(len(Z)), Z])
    support = model['support'].copy()
    coef = model['coef'].copy()
    y_mu, y_sd = model['y_mu'], model['y_sd']
    yn = (np.asarray(ytrain) - y_mu) / y_sd
    removed = []
    while int(support.sum()) > 2:
        ids = np.flatnonzero(support)
        F = A[:, ids].T @ A[:, ids] / max(len(A), 1)
        vals, vecs = np.linalg.eigh(F)
        maxv = max(float(vals[-1]), 1e-12)
        if vals[0] / maxv >= FIM_EIG_RATIO:
            break
        weak_local = np.argmax(np.abs(vecs[:, 0]))
        drop = int(ids[weak_local])
        if drop == 0:
            candidates = np.flatnonzero(ids != 0)
            if len(candidates) == 0: break
            weak_local = candidates[np.argmax(np.abs(vecs[candidates, 0]))]
            drop = int(ids[weak_local])
        support[drop] = False
        removed.append(drop)
        ids = np.flatnonzero(support)
        coef[:] = 0.0
        c = robust_huber_irls(A[:, ids], yn, HUBER_DELTA, RIDGE)
        coef[ids] = c
    pruned = dict(model)
    pruned['support'] = support
    pruned['coef'] = coef
    pruned['removed_feature_indices'] = removed
    return pruned


def prepare_generator(df, pole_pairs=None, w0_ref=None):
    t = df['t'].to_numpy(float)
    raw_theta = df['theta'].to_numpy(float)
    w = df['omega'].to_numpy(float)
    w0 = float(w0_ref if w0_ref is not None else np.median(w[:max(3, len(w)//10)]))
    if pole_pairs is None:
        pole_pairs = max(1, int(round(2*np.pi*F0 / max(abs(w0), 1e-9))))
    ang = np.deg2rad(raw_theta) if GEN_ANGLE_UNIT == 'deg' else raw_theta.copy()
    delta = np.unwrap(ang) * pole_pairs
    omega_pu = w / w0
    out = pd.DataFrame({
        't': t,
        'delta': delta,
        'omega_pu': omega_pu,
        'Pm': df['Pm_pu'].to_numpy(float),
        'fd': df['fd_p'].to_numpy(float),
    })
    out.attrs['pole_pairs'] = pole_pairs
    out.attrs['w0'] = w0
    return out


def smooth_and_derivative(values, dt):
    values = np.asarray(values, float)
    wlen = min(7, len(values) if len(values) % 2 else len(values)-1)
    if wlen < 5:
        return values.copy(), np.gradient(values, dt, edge_order=1)
    smooth = savgol_filter(values, window_length=wlen, polyorder=2, mode='interp')
    deriv = savgol_filter(values, window_length=wlen, polyorder=2, deriv=1, delta=dt, mode='interp')
    return smooth, deriv


def detect_phasors(df):
    t = df['t'].to_numpy(float)
    dt = float(np.median(np.diff(t)))
    n = int(round(1.0 / (F0 * dt)))
    if n < 8:
        raise ValueError(f'Only {n} samples per power-frequency cycle; phasor estimate is not reliable.')
    kernel = np.exp(-2j*np.pi*np.arange(n)/n)
    out_t = (t[:len(t)-n+1] + t[n-1:]) / 2
    tstart = t[:len(t)-n+1]
    rot = np.exp(-2j*np.pi*F0*tstart)
    phases = {}
    for c in ['Va','Vb','Vc','Ia','Ib','Ic']:
        x = df[c].to_numpy(float)
        windows = sliding_window_view(x, n)
        local = np.sum(windows * kernel[None, :], axis=1) * (np.sqrt(2)/n)
        phases[c] = local * rot
    a = np.exp(2j*np.pi/3)
    def seq(A,B,C):
        return (A+B+C)/3, (A+a*B+a*a*C)/3, (A+a*a*B+a*C)/3
    V0,V1,V2 = seq(phases['Va'], phases['Vb'], phases['Vc'])
    I0,I1,I2 = seq(phases['Ia'], phases['Ib'], phases['Ic'])
    return {
        't': out_t, 'phases': phases,
        'V0': V0, 'V1': V1, 'V2': V2, 'I0': I0, 'I1': I1, 'I2': I2,
        'V2V1': np.abs(V2)/np.maximum(np.abs(V1), 1e-9),
        'V0V1': np.abs(V0)/np.maximum(np.abs(V1), 1e-9),
    }


def count_consecutive(mask, run_length):
    mask = np.asarray(mask, bool)
    if len(mask) < run_length:
        return np.array([], dtype=int)
    return np.flatnonzero(np.convolve(mask.astype(int), np.ones(run_length, int), mode='valid') == run_length)


def main():
    print('Loading individually configured healthy and fault data files ...')
    healthy_gen_raw = {k: read_data_file(p, GEN_COLS) for k, p in HEALTHY_GENERATORS.items()}
    fault_gen_raw = {k: read_data_file(p, GEN_COLS) for k, p in FAULT_GENERATORS.items()}
    healthy_bus_raw = {k: read_data_file(p, BUS_COLS) for k, p in HEALTHY_BUSES.items()}
    fault_bus_raw = {k: read_data_file(p, BUS_COLS) for k, p in FAULT_BUSES.items()}

    # Check whether healthy voltage/current records are distinct by bus.
    duplicated_pairs = []
    for b1, b2 in [(1, 4), (1, 5), (4, 5)]:
        same = all(np.allclose(healthy_bus_raw[b1][c].to_numpy(), healthy_bus_raw[b2][c].to_numpy(), equal_nan=True)
                   for c in ['Va','Vb','Vc','Ia','Ib','Ic'])
        if same:
            duplicated_pairs.append((b1, b2))
    if duplicated_pairs:
        print('\nWARNING: healthy bus waveform files are exact duplicates for these bus pairs:', duplicated_pairs)
        print('Using the fault-run pre-fault interval for per-bus phase-voltage baselines.')

    print('\nSampling summaries:')
    for k, d in healthy_gen_raw.items():
        dtg = np.median(np.diff(d.t.to_numpy()))
        print(f'  G{k}: {len(d)} samples, ~{1/dtg:.1f} Hz')
    for k, d in healthy_bus_raw.items():
        dtb = np.median(np.diff(d.t.to_numpy()))
        print(f'  Healthy Bus {k}: {len(d)} samples, ~{1/dtb:.1f} Hz')

    fits = {}
    healthy_processed = {}
    fault_processed = {}
    for k in (1, 2, 3):
        hg = prepare_generator(healthy_gen_raw[k])
        fg = prepare_generator(fault_gen_raw[k], pole_pairs=hg.attrs['pole_pairs'], w0_ref=hg.attrs['w0'])
        healthy_processed[k] = hg
        fault_processed[k] = fg
        dtg = float(np.median(np.diff(hg.t.to_numpy())))
        train_end = hg.t.iloc[0] + TRAIN_FRACTION * (hg.t.iloc[-1] - hg.t.iloc[0])
        hsmooth, hderiv = {}, {}
        fsmooth, fderiv = {}, {}
        for state in ['delta', 'omega_pu']:
            hsmooth[state], hderiv[state] = smooth_and_derivative(hg[state].to_numpy(), dtg)
            fsmooth[state], fderiv[state] = smooth_and_derivative(fg[state].to_numpy(), dtg)
        
        Xh = np.column_stack([
            hg['omega_pu'].to_numpy() - 1.0,
            hg['delta'].to_numpy() - hg['delta'].iloc[0],
            hg['Pm'].to_numpy() - np.median(hg['Pm'].to_numpy()),
            hg['fd'].to_numpy() - np.median(hg['fd'].to_numpy()),
        ])
        Xf = np.column_stack([
            fg['omega_pu'].to_numpy() - 1.0,
            fg['delta'].to_numpy() - fg['delta'].iloc[0],
            fg['Pm'].to_numpy() - np.median(hg['Pm'].to_numpy()),
            fg['fd'].to_numpy() - np.median(hg['fd'].to_numpy()),
        ])
        tr = (hg.t.to_numpy() < train_end) & (hg.t.to_numpy() > hg.t.iloc[0] + 3*dtg) & (hg.t.to_numpy() < hg.t.iloc[-1] - 3*dtg)
        ho = (hg.t.to_numpy() >= train_end) & (hg.t.to_numpy() < hg.t.iloc[-1] - 2*dtg)
        
        fits[k] = {}
        for target in ['delta', 'omega_pu']:
            ytrain = hderiv[target][tr]
            model = fit_robust_stlsq(Xh[tr], ytrain, STLS_THRESHOLD)
            pred_h = predict_model(model, Xh)
            resid_h = hderiv[target] - pred_h
            cal = resid_h[ho] if ho.sum() >= 5 else resid_h[tr]
            scale = max(robust_scale(cal), 1e-6)
            reduced = fim_guided_prune(model, Xh[tr], ytrain)
            pred_fr = predict_model(model, Xf)
            pred_rr = predict_model(reduced, Xf)
            r_fault = fderiv[target] - pred_rr
            fits[k][target] = {
                'full': model, 'reduced': reduced, 'scale': scale,
                'healthy_resid': resid_h, 'fault_resid': r_fault,
                'healthy_pred': pred_h, 'fault_pred': pred_rr,
                'train_mask': tr, 'holdout_mask': ho,
                'feature_names': ['omega_dev','delta_rel','Pm_dev','fd_dev']
            }
        print(f'G{k}: train={int(tr.sum())}, healthy calibration={int(ho.sum())}')

    hb = {k: detect_phasors(d) for k, d in healthy_bus_raw.items()}
    fb = {k: detect_phasors(d) for k, d in fault_bus_raw.items()}
    
    seq_stats = {}
    for k, b in hb.items():
        m = (b['t'] >= HEALTHY_BASE_WINDOW[0]) & (b['t'] <= HEALTHY_BASE_WINDOW[1])
        seq_stats[k] = {
            'r2_med': float(np.median(b['V2V1'][m])),
            'r2_scale': max(robust_scale(b['V2V1'][m]), 1e-4),
            'r0_med': float(np.median(b['V0V1'][m])),
            'r0_scale': max(robust_scale(b['V0V1'][m]), 1e-5),
        }

    bus_features = {}
    all_alarm = np.zeros_like(next(iter(fb.values()))['t'], dtype=bool)
    t_bus = next(iter(fb.values()))['t']
    for k, b in fb.items():
        pre = (b['t'] >= HEALTHY_BASE_WINDOW[0]) & (b['t'] <= HEALTHY_BASE_WINDOW[1])
        v1mag = np.abs(b['V1'])
        v1base = float(np.median(v1mag[pre]))
        sag = 1.0 - v1mag / max(v1base, 1e-9)
        st = seq_stats[k]
        r2_thresh = max(V2_RATIO_FLOOR, st['r2_med'] + 8*st['r2_scale'])
        r0_thresh = max(V0_RATIO_FLOOR, st['r0_med'] + 8*st['r0_scale'])
        r2_alarm = b['V2V1'] > r2_thresh
        r0_alarm = b['V0V1'] > r0_thresh
        sag_alarm = sag > V1_SAG_ALARM
        alarm = sag_alarm | r2_alarm | r0_alarm
        all_alarm |= alarm
        phases_pu = {}
        for ph in 'abc':
            phmag = np.abs(b['phases']['V'+ph])
            phbase = max(float(np.median(phmag[pre])), 1e-9)
            phases_pu[ph] = phmag / phbase
        bus_features[k] = {
            'sag': sag, 'r2': b['V2V1'], 'r0': b['V0V1'],
            'r2_thresh': r2_thresh, 'r0_thresh': r0_thresh,
            'sag_alarm': sag_alarm, 'r2_alarm': r2_alarm, 'r0_alarm': r0_alarm,
            'phase_pu': phases_pu, 'v1mag': v1mag, 'v1base': v1base
        }

    starts = count_consecutive(all_alarm, DETECT_CONSEC_PHASOR_SAMPLES)
    if len(starts):
        det_idx = int(starts[0])
        onset_est = float(t_bus[det_idx])
        active_idx = np.flatnonzero(all_alarm & (t_bus >= onset_est))
        clear_est = float(t_bus[active_idx[-1]]) if len(active_idx) else np.nan
    else:
        onset_est, clear_est = np.nan, np.nan

    if np.isfinite(onset_est) and np.isfinite(clear_est) and clear_est > onset_est:
        classify_mask = (t_bus >= onset_est + 1/F0) & (t_bus <= clear_est - 1/F0)
    else:
        classify_mask = (t_bus >= FAULT_EVAL_WINDOW[0]) & (t_bus <= FAULT_EVAL_WINDOW[1])
    if classify_mask.sum() < 5:
        classify_mask = (t_bus >= FAULT_EVAL_WINDOW[0]) & (t_bus <= FAULT_EVAL_WINDOW[1])

    phase_summary = {}
    bus_rows = []
    for k, b in fb.items():
        f = bus_features[k]
        phase_summary[k] = {ph: float(np.median(f['phase_pu'][ph][classify_mask])) for ph in 'abc'}
        bus_rows.append({
            'bus': k,
            'V1_sag_pct': 100 * float(np.median(f['sag'][classify_mask])),
            'median_V2_over_V1': float(np.median(f['r2'][classify_mask])),
            'median_V0_over_V1': float(np.median(f['r0'][classify_mask])),
            'Va_pu': phase_summary[k]['a'], 'Vb_pu': phase_summary[k]['b'], 'Vc_pu': phase_summary[k]['c'],
        })
    summary = pd.DataFrame(bus_rows).set_index('bus')

    print('\n=== Bus fault features summary ===')
    print(summary.round(4).to_string())

    # Plots
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for k in (1, 2, 3):
        g = fault_processed[k]
        ax[0].plot(g.t, g.delta - g.delta.iloc[0], label=f'G{k} relative electrical angle')
    ax[0].axvspan(FAULT_START_TRUE, FAULT_END_TRUE, color='red', alpha=.10, label='known fault window')
    ax[0].set_ylabel('angle change (rad)'); ax[0].grid(alpha=.25); ax[0].legend(ncol=2)
    
    for k, f in bus_features.items():
        ax[1].plot(t_bus, f['sag'] * 100, label=f'Bus {k}')
    ax[1].axhline(V1_SAG_ALARM * 100, color='black', ls='--', label='sag threshold')
    ax[1].axvspan(FAULT_START_TRUE, FAULT_END_TRUE, color='red', alpha=.10)
    ax[1].set_ylabel('positive-sequence sag (%)'); ax[1].grid(alpha=.25); ax[1].legend(ncol=3)
    
    for k, f in bus_features.items():
        ax[2].plot(t_bus, f['r2'], label=f'Bus {k} |V2/V1|')
    ax[2].axhline(V2_RATIO_FLOOR, color='black', ls='--', label='V2/V1 floor')
    ax[2].axvspan(FAULT_START_TRUE, FAULT_END_TRUE, color='red', alpha=.10)
    ax[2].set_ylabel('|V2|/|V1|'); ax[2].set_xlabel('time (s)'); ax[2].grid(alpha=.25); ax[2].legend(ncol=3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{SAVE_PREFIX}_overview.png', dpi=180)
    plt.show()

if __name__ == '__main__':
    main()