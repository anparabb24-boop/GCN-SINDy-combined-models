"""
IEEE 9-bus SINDy fault detection with robust preprocessing and FIM-guided pruning
--------------------------------------------------------------------------------
Dependencies: numpy pandas scipy matplotlib openpyxl

Configure BUS_FILES and GEN_FILES to match your local MATLAB-exported files.
The currently supplied files only cover buses 1, 4 and 5, and generator files are
reported to be sampled at about 60 Hz. The script explicitly warns about this:
interpolation cannot recover missing dynamics.

This is a robust SINDy + FIM-guided pruning implementation. It is NOT full MBAM:
full MBAM requires geodesic continuation to a model-manifold boundary and a
limiting-model derivation. Do not call the pruning routine formal MBAM in a paper.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.signal import medfilt

# ----------------------------- CONFIG ---------------------------------
BUS_FILES = {
    1: "/Users/Arya/Desktop/vs_code/sindy_in_PSA/fault datasets/IVPQ_B1_Fag.csv",
    4: "/Users/Arya/Desktop/vs_code/sindy_in_PSA/fault datasets/IVPQ_B4_Fag.csv",
    5: "/Users/Arya/Desktop/vs_code/sindy_in_PSA/fault datasets/IVPQ_B5_Fag.csv",
}
GEN_FILES = {
    1: "/Users/Arya/Desktop/vs_code/sindy_in_PSA/fault datasets/G1_Fag.csv",
    2: "/Users/Arya/Desktop/vs_code/sindy_in_PSA/fault datasets/G2_Fag.csv",
    3: "/Users/Arya/Desktop/vs_code/sindy_in_PSA/fault datasets/G3_Fag.csv",
}
GEN_BUS = {1: 1, 2: 2, 3: 3}
BUS_COLS = dict(t="t", Va="Va", Vb="Vb", Vc="Vc", Ia="Ia", Ib="Ib", Ic="Ic")
GEN_COLS = dict(t="t", delta="theta", omega="omega", Pm="Pm_pu", fd="fd_p")
F0 = 60.0
DECIM = 2
HEALTHY_WINDOW = (0.10, 0.29)
TRAIN_FRACTION = 0.70
RIDGE = 1e-4
STLS_THRESHOLD = 0.08
HUBER_DELTA = 1.5
DETECT_Z = 6.0
DETECT_CONSEC = 3
V2_ALARM = 0.05
V0_ALARM = 0.005
LOW_PHASE_PU = 0.75
FAULT_START_KNOWN = 0.30
FAULT_END_KNOWN = 0.50
SAVE_PREFIX = "ieee9_robust_sindy"
# ----------------------------------------------------------------------

def read_table(path, mapping):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path.resolve()}. Put the CSV in the working directory "
            "or edit BUS_FILES/GEN_FILES at the top of this script."
        )
    df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
    # Rename source column names to canonical names only when source exists.
    rename = {source: canonical for canonical, source in mapping.items() if source in df.columns}
    df = df.rename(columns=rename)
    missing = [canonical for canonical in mapping if canonical not in df.columns]
    if missing:
        raise KeyError(f"{path}: missing canonical columns {missing}; columns={list(df.columns)}")
    return df.dropna(subset=["t"]).drop_duplicates("t").sort_values("t").reset_index(drop=True)

def rolling_robust_smooth(x, kernel=5):
    x = np.asarray(x, float)
    if kernel % 2 == 0: kernel += 1
    kernel = min(kernel, len(x) if len(x)%2 else len(x)-1)
    if kernel < 3: return x.copy()
    med = medfilt(x, kernel_size=kernel)
    r = x - med
    scale = 1.4826*np.median(np.abs(r-np.median(r))) + 1e-10
    return med + np.clip(r, -5*scale, 5*scale)

def one_cycle_phasor(t, x, f0=60.0):
    """Sliding one-cycle least-squares phasor at centre time; avoids the phase-reference error."""
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    dt = np.median(np.diff(t))
    n = int(round(1/(f0*dt)))
    if n < 8:
        raise ValueError(f"Only {n} samples/cycle. Need finer raw bus sampling for phasors.")
    # Fit x ~= c + a*cos(w*t) + b*sin(w*t) in each window.
    # RMS phasor convention: (a - 1j*b)/sqrt(2) for cosine-positive convention.
    w = 2*np.pi*f0
    ph = np.full(len(t), np.nan + 1j*np.nan)
    half = n//2
    for i in range(half, len(t)-half):
        sl = slice(i-half, i-half+n)
        tt = t[sl]
        M = np.column_stack([np.ones(n), np.cos(w*tt), np.sin(w*tt)])
        coef, *_ = np.linalg.lstsq(M, x[sl], rcond=None)
        ph[i] = (coef[1] - 1j*coef[2])/np.sqrt(2)
    good = np.isfinite(ph.real)
    if good.sum() < 5:
        raise ValueError("Insufficient valid phasor samples.")
    return t[good], ph[good]

def fortescue(A, B, C):
    a = np.exp(2j*np.pi/3)
    V0 = (A+B+C)/3
    V1 = (A+a*B+a*a*C)/3
    V2 = (A+a*a*B+a*C)/3
    return V0, V1, V2

def huber_regression(A, y, delta=1.5, ridge=1e-4, n_iter=30):
    w = np.ones(len(y))
    coef = np.zeros(A.shape[1])
    for _ in range(n_iter):
        Aw = A*np.sqrt(w[:,None])
        yw = y*np.sqrt(w)
        coef_new = np.linalg.solve(Aw.T@Aw + ridge*np.eye(A.shape[1]), Aw.T@yw)
        r = y-A@coef_new
        s = 1.4826*np.median(np.abs(r-np.median(r))) + 1e-10
        u = np.abs(r)/(delta*s)
        w_new = np.ones_like(u)
        m = u > 1
        w_new[m] = 1/u[m]
        if np.linalg.norm(coef_new-coef) < 1e-8*(1+np.linalg.norm(coef)):
            coef = coef_new
            break
        coef, w = coef_new, w_new
    return coef

def build_library(states, bus, gen):
    names, cols = [], []
    def add(name, value):
        names.append(name); cols.append(np.asarray(value, float))
    for name, val in states.items(): add(name, val)
    for k, s in bus.items():
        v1 = np.maximum(np.abs(s["V1"]), 1e-8)
        add(f"V2V1_B{k}", np.abs(s["V2"])/v1)
        add(f"V0V1_B{k}", np.abs(s["V0"])/v1)
    bkeys = list(bus)
    for i in range(len(bkeys)):
        for j in range(i+1, len(bkeys)):
            bi,bj=bkeys[i],bkeys[j]
            dth=np.unwrap(np.angle(bus[bi]["V1"]))-np.unwrap(np.angle(bus[bj]["V1"]))
            add(f"Vm{bi}Vm{bj}_cos_dtheta", states[f"Vm{bi}"]*states[f"Vm{bj}"]*np.cos(dth))
            add(f"Vm{bi}Vm{bj}_sin_dtheta", states[f"Vm{bi}"]*states[f"Vm{bj}"]*np.sin(dth))
    gkeys = list(gen)
    for i in range(len(gkeys)):
        for j in range(i+1,len(gkeys)):
            gi,gj=gkeys[i],gkeys[j]
            dd=gen[gi]["delta"]-gen[gj]["delta"]
            add(f"sin_delta_G{gi}_G{gj}",np.sin(dd))
            add(f"cos_delta_G{gi}_G{gj}",np.cos(dd))
    for gk,bk in GEN_BUS.items():
        if gk in gen and bk in bus:
            dd=gen[gk]["delta"]-np.unwrap(np.angle(bus[bk]["V1"]))
            add(f"sin_delta_theta_G{gk}_B{bk}",np.sin(dd))
    return names, np.column_stack(cols)

def fit_stlsq(A, Y, threshold=0.08):
    """Per-target robust STLSQ on standardized library and standardized targets."""
    Xi = np.zeros((A.shape[1],Y.shape[1]))
    target_mu = np.median(Y,axis=0)
    target_scale = 1.4826*np.median(np.abs(Y-target_mu),axis=0)+1e-8
    Yn=(Y-target_mu)/target_scale
    for j in range(Y.shape[1]):
        support=np.ones(A.shape[1],dtype=bool)
        for _ in range(15):
            c=huber_regression(A[:,support],Yn[:,j],HUBER_DELTA,RIDGE)
            keep=np.abs(c)>=threshold
            if not keep.any(): keep[np.argmax(np.abs(c))]=True
            old=support.copy()
            ids=np.flatnonzero(support)
            support[:]=False
            support[ids[keep]]=True
            if np.array_equal(old,support): break
        if support.any():
            c=huber_regression(A[:,support],Yn[:,j],HUBER_DELTA,RIDGE)
            Xi[support,j]=c*target_scale[j]
    return Xi,target_mu,target_scale

def fim_guided_prune(A, Y, Xi, eig_ratio=1e-7):
    """FIM-guided term pruning approximation; not formal MBAM."""
    out=Xi.copy()
    for j in range(Xi.shape[1]):
        active=np.flatnonzero(np.abs(out[:,j])>1e-10)
        while len(active)>1:
            F=A[:,active].T@A[:,active]/len(A)
            vals,vecs=np.linalg.eigh(F)
            if vals[-1] <= 1e-12 or vals[0]/vals[-1] >= eig_ratio: break
            drop=active[np.argmax(np.abs(vecs[:,0]))]
            active=active[active!=drop]
            out[:,j]=0
            if len(active):
                out[active,j]=huber_regression(A[:,active],Y[:,j],HUBER_DELTA,RIDGE)
    return out

def main():
    bus_raw={}
    for k,path in BUS_FILES.items():
        d=read_table(path,BUS_COLS)
        # robust clipping of isolated spikes in raw waveform before phasor extraction
        for c in ["Va","Vb","Vc","Ia","Ib","Ic"]:
            d[c]=rolling_robust_smooth(d[c].to_numpy(), kernel=5)
        tc=None; P={}
        for c in ["Va","Vb","Vc","Ia","Ib","Ic"]:
            tt,pp=one_cycle_phasor(d.t.to_numpy(),d[c].to_numpy(),F0)
            if tc is None: tc=tt
            else:
                # interpolation is allowed for aligning phasor channels only; not upsampling generator information.
                pass
            P[c]=pp
        # common valid phasor times are identical for columns from same file
        V0,V1,V2=fortescue(P["Va"],P["Vb"],P["Vc"])
        I0,I1,I2=fortescue(P["Ia"],P["Ib"],P["Ic"])
        bus_raw[k]=dict(t=tc,V0=V0,V1=V1,V2=V2,I0=I0,I1=I1,I2=I2,
                        phaseV={c:P["V"+c] for c in "abc"},phaseI={c:P["I"+c] for c in "abc"})
    # Choose coarsest bus phasor grid as common target grid.
    t0=max(s["t"][0] for s in bus_raw.values())
    t1=min(s["t"][-1] for s in bus_raw.values())
    ref=next(iter(bus_raw.values()))["t"]
    t=ref[(ref>=t0)&(ref<=t1)][::DECIM]
    if len(t)<30: raise RuntimeError("Too few common samples after phasor extraction.")
    bus={}
    def interp_complex(tx,z): return np.interp(t,tx,z.real)+1j*np.interp(t,tx,z.imag)
    for k,s in bus_raw.items():
        bus[k]={key:interp_complex(s["t"],s[key]) for key in ["V0","V1","V2","I0","I1","I2"]}
        bus[k]["phaseV"]={c:interp_complex(s["t"],s["phaseV"][c]) for c in "abc"}
        bus[k]["phaseI"]={c:interp_complex(s["t"],s["phaseI"][c]) for c in "abc"}
    dt=float(np.median(np.diff(t)))
    healthy=(t>=HEALTHY_WINDOW[0])&(t<=HEALTHY_WINDOW[1])
    if healthy.sum()<20: raise RuntimeError("Healthy window has too few samples. Export more pre-fault data.")
    gen={}
    for k,path in GEN_FILES.items():
        d=read_table(path,GEN_COLS)
        tg=d.t.to_numpy()
        fs=1/np.median(np.diff(tg))
        print(f"G{k}: {len(tg)} samples, native sampling rate ~{fs:.1f} Hz")
        if fs < 10*F0:
            print(f"WARNING: G{k} is sparsely sampled for fast electromechanical/electrical dynamics; interpolation will not restore missing information.")
        w0=float(d.omega.iloc[0])
        pole_pairs=max(1,int(round(2*np.pi*F0/w0)))
        angle=np.deg2rad(d.delta.to_numpy()) if GEN_COLS["delta"]=="theta" else d.delta.to_numpy()
        angle=np.unwrap(angle)*pole_pairs
        gen[k]=dict(
            delta=PchipInterpolator(tg,angle,extrapolate=False)(t),
            omega=PchipInterpolator(tg,d.omega.to_numpy()/max(abs(w0),1e-8),extrapolate=False)(t),
            Pm=PchipInterpolator(tg,d.Pm.to_numpy(),extrapolate=False)(t),
            fd=PchipInterpolator(tg,d.fd.to_numpy(),extrapolate=False)(t)
        )
    # remove points outside generator data range due to extrapolate=False
    valid=np.ones(len(t),bool)
    for g in gen.values():
        for arr in g.values(): valid &= np.isfinite(arr)
    t=t[valid]; healthy=healthy[valid]
    for k in bus:
        for key in ["V0","V1","V2","I0","I1","I2"]:
            bus[k][key]=bus[k][key][valid]
        for key in ["phaseV","phaseI"]:
            bus[k][key]={c:z[valid] for c,z in bus[k][key].items()}
    for k in gen:
        gen[k]={key:z[valid] for key,z in gen[k].items()}
    # states, normalized bus variables, plus generator variables
    states={}
    groups={}
    for k,g in gen.items():
        for key in ["delta","omega","Pm","fd"]:
            states[f"{key}{k}"]=g[key]; groups[f"{key}{k}"]=f"G{k}"
    for k,s in bus.items():
        vm=np.abs(s["V1"]); im=np.abs(s["I1"])
        states[f"Vm{k}"]=vm/(np.mean(vm[healthy])+1e-8); groups[f"Vm{k}"]=f"B{k}"
        states[f"Im{k}"]=im/(np.mean(im[healthy])+1e-8); groups[f"Im{k}"]=f"B{k}"
        states[f"th{k}"]=np.unwrap(np.angle(s["V1"])); groups[f"th{k}"]=f"B{k}"
    state_names=list(states)
    # Derivative estimates are from smoothed signals. Derivative of generator channels is especially uncertain at low native rate.
    Xdot=np.column_stack([np.gradient(states[n],t,edge_order=2) for n in state_names])
    lib_names,Theta=build_library(states,bus,gen)
    idx=np.flatnonzero(healthy)
    ntrain=max(15,int(TRAIN_FRACTION*len(idx)))
    tr=idx[:ntrain]; ho=idx[ntrain:]
    mu=Theta[tr].mean(axis=0); sd=Theta[tr].std(axis=0)
    valid_col=sd>1e-9; sd[~valid_col]=1
    A=(Theta-mu)/sd; A[:,~valid_col]=0
    # Restrict targets/scales to healthy training data only (no post-fault leakage).
    ymu=np.median(Xdot[tr],axis=0)
    ys=1.4826*np.median(np.abs(Xdot[tr]-ymu),axis=0)+1e-8
    Y=(Xdot-ymu)/ys
    Xi_full,_,_=fit_stlsq(A[tr][:,valid_col],Y[tr],STLS_THRESHOLD)
    # restore zeroed invalid columns into full library matrix
    Xi0=np.zeros((A.shape[1],Xi_full.shape[1])); Xi0[valid_col]=Xi_full
    Xi_red=fim_guided_prune(A[tr][:,valid_col],Y[tr],Xi_full,1e-7)
    XiR=np.zeros_like(Xi0); XiR[valid_col]=Xi_red
    pred_full=(A@Xi0)*ys+ymu
    pred_red=(A@XiR)*ys+ymu
    residual_full=Xdot-pred_full
    residual_red=Xdot-pred_red
    # Robust held-out healthy residual scale; fallback to MAD, never use post-fault samples.
    if len(ho)<5:
        print("WARNING: held-out healthy interval is short; residual calibration may be unstable.")
        ho=tr[-max(5,len(tr)//5):]
    sig=1.4826*np.median(np.abs(residual_red[ho]-np.median(residual_red[ho],axis=0)),axis=0)+1e-8
    Z=np.abs(residual_red)/sig
    R=np.sqrt(np.mean(Z**2,axis=1))
    hot=(R>DETECT_Z)&(t>HEALTHY_WINDOW[1])
    alarm=np.nan
    for i in range(len(hot)-DETECT_CONSEC+1):
        if hot[i:i+DETECT_CONSEC].all():
            alarm=t[i]; break
    print("\n=== SINDy / FIM-guided reduction summary ===")
    print(f"Common grid: {1/dt:.1f} Hz; samples={len(t)}; states={len(state_names)}; library={len(lib_names)}")
    print(f"Active coefficients: full={np.count_nonzero(Xi0)}, reduced={np.count_nonzero(XiR)}")
    print(f"Reduced-model residual alarm: {alarm:.4f} s" if np.isfinite(alarm) else "No residual alarm detected.")
    print("Known configured fault interval: %.3f–%.3f s" % (FAULT_START_KNOWN,FAULT_END_KNOWN))
    print("Interpret alarms against known simulation timing; the known interval is not used to fit the model.")
    # Sequence metrics and fault type heuristic
    rows=[]
    for k,s in bus.items():
        v1=np.maximum(np.abs(s["V1"]),1e-8)
        v2r=np.abs(s["V2"])/v1; v0r=np.abs(s["V0"])/v1
        phase_ratios=[]
        for c in "abc":
            vv=np.abs(s["phaseV"][c])
            base=np.median(vv[healthy])
            phase_ratios.append(vv/(base+1e-8))
        phase_ratios=np.array(phase_ratios)
        fw=(t>=FAULT_START_KNOWN)&(t<=FAULT_END_KNOWN)
        pre=healthy
        sag=1-np.mean(v1[fw])/(np.mean(v1[pre])+1e-8)
        rows.append(dict(bus=k,V1_sag_pct=100*sag,
                         V2_over_V1=float(np.mean(v2r[fw])),
                         V0_over_V1=float(np.mean(v0r[fw])),
                         min_phase_pu=float(np.min(np.mean(phase_ratios[:,fw],axis=1))),
                         residual_z=float(np.mean(R[fw]))))
    table=pd.DataFrame(rows).set_index("bus")
    print("\nBus features over configured fault interval (heuristic, not validated localization):")
    print(table.round(4).to_string())
    # Detection overview plot
    fig,ax=plt.subplots(3,1,figsize=(12,9),sharex=True)
    ax[0].plot(t,R,label="reduced-model residual score")
    ax[0].axhline(DETECT_Z,color="r",ls="--",label="threshold")
    ax[0].axvspan(FAULT_START_KNOWN,FAULT_END_KNOWN,color="red",alpha=.1,label="known fault interval")
    if np.isfinite(alarm): ax[0].axvline(alarm,color="k",ls=":",label=f"alarm {alarm:.3f}s")
    ax[0].set_ylabel("robust residual z"); ax[0].legend(); ax[0].grid(alpha=.25)
    for k,s in bus.items():
        ax[1].plot(t,np.abs(s["V1"]),label=f"Bus {k}")
    ax[1].axvspan(FAULT_START_KNOWN,FAULT_END_KNOWN,color="red",alpha=.1)
    ax[1].set_ylabel("|V1|"); ax[1].legend(); ax[1].grid(alpha=.25)
    ax[2].plot(t,np.abs(bus[next(iter(bus))]["V2"])/np.maximum(np.abs(bus[next(iter(bus))]["V1"]),1e-8),label="|V2/V1|")
    ax[2].plot(t,np.abs(bus[next(iter(bus))]["V0"])/np.maximum(np.abs(bus[next(iter(bus))]["V1"]),1e-8),label="|V0/V1|")
    ax[2].axvspan(FAULT_START_KNOWN,FAULT_END_KNOWN,color="red",alpha=.1)
    ax[2].set_ylabel("sequence ratio"); ax[2].set_xlabel("time (s)"); ax[2].legend(); ax[2].grid(alpha=.25)
    plt.tight_layout()
    plt.savefig(f"{SAVE_PREFIX}_overview.png",dpi=180)
    table.to_csv(f"{SAVE_PREFIX}_bus_features.csv")
    pd.DataFrame(XiR,index=lib_names,columns=state_names).to_csv(f"{SAVE_PREFIX}_coefficients_reduced.csv")
    plt.show()

if __name__ == "__main__":
    main()
