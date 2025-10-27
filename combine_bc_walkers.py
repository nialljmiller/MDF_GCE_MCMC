#!/usr/bin/env python3

"""Combine emcee walkers from ``bc_*`` folders and make a corner plot.

The script discovers per-chunk ``emcee`` chains produced under batch chunk
directories (by default folders whose name starts with ``bc_``), stitches the
walkers together into a single HDF5 chain, and then generates a corner plot for
the flattened posterior samples.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

try:
    import h5py  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    h5py = None  # type: ignore

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    np = None  # type: ignore


MCMC_GROUP = "mcmc"
CHAIN_DATASET = f"{MCMC_GROUP}/chain"
LOGP_DATASET = f"{MCMC_GROUP}/log_prob"
ACC_DATASET = f"{MCMC_GROUP}/acceptance_fraction"
NAMES_DATASET = f"{MCMC_GROUP}/parameter_names"


def _find_chain_file(directory: Path) -> Optional[Path]:
    """Return the first HDF5 file in *directory* that looks like an emcee chain."""

    if not directory.is_dir():
        return None

    candidates = sorted(directory.rglob("*.h5")) + sorted(directory.rglob("*.hdf5"))
    for path in candidates:
        try:
            with h5py.File(path, "r") as h5:
                if CHAIN_DATASET in h5:
                    return path
        except OSError:
            continue
    return None


def _load_chain(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
    """Load chain, log-prob, acceptance fractions, and parameter names from *path*."""

    with h5py.File(path, "r") as h5:
        chain = h5[CHAIN_DATASET][...]
        log_prob = h5[LOGP_DATASET][...] if LOGP_DATASET in h5 else None
        acc_frac = h5[ACC_DATASET][...] if ACC_DATASET in h5 else None
        names = None
        if NAMES_DATASET in h5:
            raw = h5[NAMES_DATASET][...]
            names = [n.decode() if isinstance(n, (bytes, bytearray)) else str(n) for n in raw]
    return chain, log_prob, acc_frac, names


def _ensure_same_shape(chains: Sequence[np.ndarray]) -> Tuple[int, int, int]:
    """Check chains for consistent iteration/parameter counts and return their shape."""

    if not chains:
        raise ValueError("No chains to combine.")

    niter = min(c.shape[0] for c in chains)
    ndim = chains[0].shape[2]
    for c in chains:
        if c.shape[2] != ndim:
            raise ValueError("Chains have mismatched parameter dimensions; cannot combine.")
    return niter, ndim, sum(c.shape[1] for c in chains)


def _combine_arrays(arrays: Sequence[np.ndarray], axis: int, niter: Optional[int] = None) -> np.ndarray:
    """Concatenate *arrays* along *axis*, cropping to *niter* along the first axis if set."""

    if niter is not None:
        arrays = [a[:niter] for a in arrays]
    return np.concatenate(arrays, axis=axis)


def _write_combined_chain(
    output: Path,
    chain: np.ndarray,
    log_prob: Optional[np.ndarray],
    acc_frac: Optional[np.ndarray],
    names: Optional[Sequence[str]],
    sources: Sequence[Path],
) -> None:
    """Write combined results to *output* in the emcee HDF5 layout."""

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    with h5py.File(output, "w") as h5:
        grp = h5.create_group(MCMC_GROUP)
        grp.create_dataset("chain", data=chain, compression="gzip", compression_opts=3)
        if log_prob is not None:
            grp.create_dataset("log_prob", data=log_prob, compression="gzip", compression_opts=3)
        if acc_frac is not None:
            grp.create_dataset("acceptance_fraction", data=acc_frac)
        if names is not None:
            grp.create_dataset("parameter_names", data=np.array(names, dtype="S"))
        grp.attrs["combined_from"] = np.array([str(p) for p in sources], dtype="S")


def _prepare_flat_samples(
    chain: np.ndarray,
    log_prob: Optional[np.ndarray],
    burn: int,
    thin: int,
    max_samples: Optional[int],
    seed: Optional[int],
) -> np.ndarray:
    """Return flattened (post-burn, thinned) samples optionally filtered by log_prob."""

    if burn < 0:
        raise ValueError("burn must be non-negative")
    if thin <= 0:
        raise ValueError("thin must be >= 1")

    niter = chain.shape[0]
    if burn >= niter:
        raise ValueError(f"burn ({burn}) must be smaller than the number of iterations ({niter})")

    samples = chain[burn::thin]
    flat = samples.reshape(-1, samples.shape[-1])

    if log_prob is not None:
        lp = log_prob[burn::thin]
        mask = np.isfinite(lp).reshape(-1)
        if mask.any():
            flat = flat[mask]

    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive or None")
        if flat.shape[0] > max_samples:
            rng = np.random.default_rng(seed)
            idx = rng.choice(flat.shape[0], size=max_samples, replace=False)
            flat = flat[idx]
    return flat


def _make_corner(samples: np.ndarray, names: Sequence[str], output: Path) -> None:
    """Render a corner plot for *samples* using *names* and save to *output*."""

    if samples.size == 0:
        raise ValueError("No samples available for corner plot.")

    try:
        import corner
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "The 'corner' package is required to draw the corner plot. Install it with 'pip install corner'."
        ) from exc

    fig = corner.corner(
        samples,
        labels=names,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3f",
        label_kwargs={"fontsize": 10},
    )
    fig.suptitle("Combined posterior samples", fontsize=12)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    return None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path.cwd(), help="Directory to search for bc_* folders (default: cwd).")
    ap.add_argument("--pattern", default="bc_*", help="Glob pattern for walker folders (default: bc_*).")
    ap.add_argument("--output", type=Path, default=Path("combined_chain.h5"), help="Path for the merged HDF5 chain.")
    ap.add_argument("--corner", type=Path, default=Path("combined_corner.png"), help="Output path for the corner plot image.")
    ap.add_argument("--burn", type=int, default=0, help="Number of initial iterations to discard before flattening.")
    ap.add_argument("--thin", type=int, default=1, help="Thin factor applied before flattening (default: 1).")
    ap.add_argument("--max-samples", type=int, default=200000,
                    help="Maximum number of samples to draw for the corner plot (default: 200000; use 0 for no limit).")
    ap.add_argument("--seed", type=int, default=None, help="Random seed used when sub-sampling for the corner plot.")
    ap.add_argument("--param-names", nargs="*", default=None,
                    help="Optional list of parameter labels to use in the corner plot.")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if h5py is None:
        print("Error: the 'h5py' package is required to combine walker chains. Install it with 'pip install h5py'.")
        return 1
    if np is None:
        print("Error: the 'numpy' package is required to process walker chains. Install it with 'pip install numpy'.")
        return 1

    root = args.root.expanduser().resolve()
    pattern = args.pattern
    walker_dirs = sorted(p for p in root.rglob(pattern) if p.is_dir())

    if not walker_dirs:
        print(f"No directories matching pattern '{pattern}' found under {root}.")
        return 1

    chains = []
    logps: List[Optional[np.ndarray]] = []
    accs: List[Optional[np.ndarray]] = []
    param_names: Optional[List[str]] = None
    used_paths: List[Path] = []

    for wdir in walker_dirs:
        chain_path = _find_chain_file(wdir)
        if chain_path is None:
            print(f"[warn] Skipping {wdir}: no emcee chain found.")
            continue
        chain, logp, acc, names = _load_chain(chain_path)
        chains.append(chain)
        logps.append(logp)
        accs.append(acc)
        used_paths.append(chain_path)
        if names and not param_names:
            param_names = list(names)

    if not chains:
        print(f"No emcee chain files found in directories matching '{pattern}'.")
        return 1

    try:
        niter, ndim, nwalkers = _ensure_same_shape(chains)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1
    combined_chain = _combine_arrays(chains, axis=1, niter=niter)

    if any(lp is None for lp in logps):
        combined_logp = None
    else:
        combined_logp = _combine_arrays([lp for lp in logps if lp is not None], axis=1, niter=niter)

    if any(acc is None for acc in accs):
        combined_acc = None
    else:
        combined_acc = np.concatenate([acc for acc in accs if acc is not None], axis=0)

    if args.param_names:
        names = list(args.param_names)
        if len(names) < ndim:
            names.extend([f"p{j}" for j in range(len(names), ndim)])
        elif len(names) > ndim:
            names = names[:ndim]
    else:
        names = param_names if param_names else [f"p{j}" for j in range(ndim)]

    _write_combined_chain(args.output, combined_chain, combined_logp, combined_acc, names, used_paths)
    print(f"Combined chain saved to {args.output} (iterations={niter}, walkers={nwalkers}, ndim={ndim}).")

    max_samples = None if args.max_samples in (None, 0) else int(args.max_samples)
    try:
        flat = _prepare_flat_samples(
            combined_chain, combined_logp, args.burn, args.thin, max_samples, args.seed
        )
    except ValueError as exc:
        print(f"Error while preparing samples: {exc}")
        return 1

    try:
        _make_corner(flat, names, args.corner)
    except ValueError as exc:
        print(f"Error while drawing corner plot: {exc}")
        return 1

    print(f"Corner plot written to {args.corner} with {flat.shape[0]} samples.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
