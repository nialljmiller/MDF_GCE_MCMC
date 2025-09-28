#!/usr/bin/env python3
import argparse, re, numpy as np, h5py, matplotlib.pyplot as plt
from pathlib import Path

def read_pcard_bounds(pcard_path):
    """
    Parse bounds like:
      tmax_2_list = [0.1, 10.0]
      infall_timescale_2_list = [0.1, 8.0]
    Returns dict: name -> (lo, hi)
    """
    bounds = {}
    pat = re.compile(r'^\s*([A-Za-z0-9_]+)_list\s*=\s*\[\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*\]')
    for line in Path(pcard_path).read_text().splitlines():
        m = pat.match(line)
        if m:
            name, lo, hi = m.group(1), float(m.group(2)), float(m.group(3))
            bounds[name] = (lo, hi)
    return bounds

def load_hdf_chain(h5path):
    with h5py.File(h5path, "r") as f:
        # emcee v3 HDF backend layout
        chain     = f["mcmc/chain"][...]        # shape: (niter, nwalkers, ndim)
        log_prob  = f["mcmc/log_prob"][...]
        acc_frac  = f["mcmc/acceptance_fraction"][...] if "mcmc/acceptance_fraction" in f else None
        param_names = None
        if "mcmc/parameter_names" in f:
            param_names = [n.decode() for n in f["mcmc/parameter_names"][...]]
    return chain, log_prob, acc_frac, param_names

def integrated_autocorr(x):
    """
    Heuristic IAT per walker via FFT ACF (no emcee dependency).
    Returns median across walkers.
    """
    # x: (niter, nwalkers)
    n, w = x.shape
    x = x - x.mean(axis=0, keepdims=True)
    # next pow2 for FFT
    nfft = 1 << (2*n - 1).bit_length()
    fx = np.fft.rfft(x, n=nfft, axis=0)
    acf = np.fft.irfft(fx * np.conjugate(fx), n=nfft, axis=0)[:n].real
    acf /= acf[0:1,:]
    # Geyer initial positive sequence truncation
    tau = np.ones(w)
    for j in range(w):
        s = 1.0
        for t in range(1, n):
            if acf[t, j] <= 0:
                break
            s += 2.0 * acf[t, j]
        tau[j] = s
    return float(np.median(tau)), acf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True)
    ap.add_argument("--pcard")
    ap.add_argument("--param-names", nargs="+", default=None,
                    help="Subset (order must match chain ordering if names unavailable).")
    ap.add_argument("--burn", type=int, default=0)
    args = ap.parse_args()

    chain, logp, acc_frac, names_in = load_hdf_chain(args.chain)
    niter, nwalkers, ndim = chain.shape
    burn = max(0, args.burn)

    # parameter names
    if args.param_names:
        names = args.param_names
        if len(names) != ndim:
            # allow subset: if user passed fewer names, take first k
            if len(names) < ndim:
                names = names + [f"p{j}" for j in range(len(names), ndim)]
            else:
                names = names[:ndim]
    elif names_in:
        names = names_in
    else:
        names = [f"p{j}" for j in range(ndim)]

    bounds = read_pcard_bounds(args.pcard) if args.pcard else {}

    samples = chain[burn:]  # (npost, nwalkers, ndim)
    flat = samples.reshape(-1, ndim)

    # --- 1) Acceptance fractions per walker
    if acc_frac is None:
        # estimate from number of repeats in logp; fallback marker
        acc = np.full(nwalkers, np.nan)
    else:
        acc = acc_frac

    plt.figure(figsize=(7,3))
    plt.bar(np.arange(nwalkers), acc, width=0.9)
    plt.axhline(0.2, ls="--", lw=1)
    plt.axhline(0.5, ls="--", lw=1)
    plt.ylim(0,1)
    plt.xlabel("Walker")
    plt.ylabel("Acceptance frac")
    plt.title("Acceptance fractions (target ~0.2–0.5)")
    plt.tight_layout()
    plt.savefig("diag_acceptance.png", dpi=180)

    # --- 2) Trace plots for requested params (or all if <=6)
    show_idx = list(range(min(ndim, 6))) if args.param_names is None else list(range(min(len(names), 6)))
    T = samples.shape[0]
    tgrid = np.arange(T)

    fig, axes = plt.subplots(len(show_idx), 1, figsize=(9, 1.8*len(show_idx)), sharex=True)
    if len(show_idx) == 1: axes = [axes]
    for ax, j in zip(axes, show_idx):
        for w in range(nwalkers):
            ax.plot(tgrid, samples[:, w, j], lw=0.4, alpha=0.6)
        lohi = bounds.get(names[j].replace(" ", "_"), None)
        if lohi:
            ax.axhline(lohi[0], color="k", lw=1, ls=":")
            ax.axhline(lohi[1], color="k", lw=1, ls=":")
        ax.set_ylabel(names[j])
    axes[-1].set_xlabel("Post-burn iterations")
    fig.suptitle("Trace plots")
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig("diag_traces.png", dpi=180)

    # --- 3) 1D histograms with edge-hit indicators
    cols = min(4, len(show_idx))
    rows = int(np.ceil(len(show_idx)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2*cols, 2.6*rows))
    axes = np.array(axes).reshape(-1)
    for k, j in enumerate(show_idx):
        ax = axes[k]
        ax.hist(flat[:, j], bins=60, density=True, histtype="step")
        name = names[j].replace(" ", "_")
        if name in bounds:
            lo, hi = bounds[name]
            ax.axvline(lo, ls=":", lw=1)
            ax.axvline(hi, ls=":", lw=1)
            # edge-hit fraction within 1% of bound
            width = hi - lo
            if width > 0:
                f_lo = np.mean(flat[:, j] <= lo + 0.01*width)
                f_hi = np.mean(flat[:, j] >= hi - 0.01*width)
                ax.set_title(f"{names[j]}  (edge {100*(f_lo+f_hi):.1f}%)")
        else:
            ax.set_title(names[j])
    for ax in axes[len(show_idx):]:
        ax.axis("off")
    fig.suptitle("1D marginals (+edge % if near bounds)")
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig("diag_histograms.png", dpi=180)

    # --- 4) Autocorrelation time estimate per param
    taus = []
    for j in range(ndim):
        x = samples[:, :, j]  # (T, W)
        tau_j, _ = integrated_autocorr(x)
        taus.append(tau_j)
    taus = np.array(taus)
    # Text report + bar plot for first 12
    with open("diag_stats.txt","w") as fh:
        fh.write(f"niter_total={niter}  nwalkers={nwalkers}  ndim={ndim}  burn={burn}\n")
        fh.write("Param\tIAT_tau\tESS\n")
        for name, tau in zip(names, taus):
            ess = nwalkers * (T / max(1.0, tau))
            fh.write(f"{name}\t{tau:.1f}\t{ess:.0f}\n")

    plt.figure(figsize=(8,3))
    k = min(12, ndim)
    order = np.argsort(-taus)[:k]
    plt.bar(range(k), taus[order])
    plt.xticks(range(k), [names[i] for i in order], rotation=45, ha="right")
    plt.ylabel("IAT τ (iters)")
    plt.title("Integrated autocorrelation (larger = worse mixing)")
    plt.tight_layout()
    plt.savefig("diag_autocorr.png", dpi=180)

if __name__ == "__main__":
    main()
