#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MDF-only MCMC with GA-style plots

- Reads observational MDF exactly like the GA: equal_weight_mdf.dat
- Runs omega_plus forward model and computes a proper likelihood:
  Gaussian in the normalized MDF space on the observed Fe/H grid.
- Optionally adds weak AMR/alpha regularizers (off by default).
- Produces GA-style plots for MDF + AMR/alpha and a true posterior corner.
- Keeps the public surface used by MDF_MCMC_Launcher.py:
    * GalacticEvolutionMCMC(...) constructor (accepts the same args; extras ignored)
    * run(nwalkers, nsteps, p0=None, out_h5=None, output_interval=None, PP=False)
    * save_partial_results(generation)

This file is self-contained and does not depend on GA code.
"""

import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee

# Forward model (NuPyCEE)
from JINAPyCEE import omega_plus

# GA-style plotting helpers (already in repo)
import mdf_plotting
from mdf_plotting import plot_mdf_curves, plot_omni_figure  # compact figures
import phys_plot  # optional, not fatal if plotting fails


# ----------------------------- Utility helpers ------------------------------

def _as_list(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return [x]

def _finite_mask(*arrs):
    m = np.ones_like(arrs[0], dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m


# ------------------------------ Main MCMC class -----------------------------

class GalacticEvolutionMCMC:
    """
    Continuous-parameter MCMC for bulge MDF.

    Active parameters are those with a non-zero range in the p-card arrays.
    Categorical knobs (IMF table, yields, etc.) are treated as fixed at the
    single choices supplied in the inlist.
    """

    # spec: (public name, key in pcard dict, log-scale mapping?)
    _CONT_SPEC = [
        ("sigma_2",              "sigma_2_list",              False),
        ("tmax_1",               "tmax_1_list",               True),
        ("tmax_2",               "tmax_2_list",               True),
        ("infall_timescale_1",   "infall_timescale_1_list",   True),
        ("infall_timescale_2",   "infall_timescale_2_list",   True),
        ("sfe",                  "sfe_array",                 False),
        ("delta_sfe",            "delta_sfe_array",           False),
        ("imf_upper_limits",     "imf_upper_limits",          True),
        ("mgal_values",          "mgal_values",               True),
        ("nb_array",             "nb_array",                  True),
    ]

    def __init__(self,
                 output_path,
                 sn1a_header, iniab_header,
                 sigma_2_list, tmax_1_list, tmax_2_list,
                 infall_timescale_1_list, infall_timescale_2_list,
                 comp_array, imf_array, sfe_array, delta_sfe_array,
                 imf_upper_limits, sn1a_assumptions, stellar_yield_assumptions,
                 mgal_values, nb_array, sn1a_rates, timesteps, A1, A2,
                 feh=None, normalized_count=None, obs_age_data=None,
                 loss_metric=None, obs_age_data_loss_metric=None,
                 obs_age_data_target=None, mdf_vs_age_weight=None,
                 tau=1.0, PP=False, **kwargs):
        """
        Accepts the launcher/inlist fields; ignores GA-only knobs.
        """
        self.output_path = str(output_path)
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        # Fixed categorical choices (singletons only)
        self.sn1a_header = sn1a_header
        self.iniab_header = iniab_header
        self.comp_array = _as_list(comp_array)
        self.imf_array = _as_list(imf_array)
        self.sn1a_assumptions = _as_list(sn1a_assumptions)
        self.stellar_yield_assumptions = _as_list(stellar_yield_assumptions)
        self.sn1a_rates = _as_list(sn1a_rates)

        self._comp   = self.comp_array[0]
        self._imf    = self.imf_array[0]
        self._sn1a   = self.sn1a_assumptions[0]
        self._yields = self.stellar_yield_assumptions[0]
        self._sn1ar  = self.sn1a_rates[0]

        # Continuous ranges: freeze singletons
        cfg = {
            "sigma_2_list":            _as_list(sigma_2_list),
            "tmax_1_list":             _as_list(tmax_1_list),
            "tmax_2_list":             _as_list(tmax_2_list),
            "infall_timescale_1_list": _as_list(infall_timescale_1_list),
            "infall_timescale_2_list": _as_list(infall_timescale_2_list),
            "sfe_array":               _as_list(sfe_array),
            "delta_sfe_array":         _as_list(delta_sfe_array),
            "imf_upper_limits":        _as_list(imf_upper_limits),
            "mgal_values":             _as_list(mgal_values),
            "nb_array":                _as_list(nb_array),
        }

        self._cont_active: List[Dict] = []   # dicts: {name, lo, hi, log}
        self._cont_frozen: Dict[str, float] = {}
        for name, key, logscale in self._CONT_SPEC:
            arr = np.asarray(cfg[key], float)
            if arr.size == 1 or not np.isfinite(arr).all() or (arr.max() - arr.min()) <= 0:
                self._cont_frozen[name] = float(arr.ravel()[0])
            else:
                lo = float(arr.min()); hi = float(arr.max())
                if logscale and (lo <= 0 or hi <= 0):
                    logscale = False
                self._cont_active.append(dict(name=name, lo=lo, hi=hi, log=logscale))

        self.param_names = [d["name"] for d in self._cont_active]
        self.nd_active   = len(self.param_names)

        # Misc model cfg
        self.timesteps = timesteps
        self.A1, self.A2 = float(A1), float(A2)

        # Observational MDF (GA style)
        self._load_obs_mdf_ga()  # sets self.obs_feh, self.obs_mdf_norm

        # Small config dict for likelihood knobs
        self.cfg = {
            "sigma_mdf": 0.02,   # normalized-shape noise (in PDF units)
            "use_amr":   False,
            "sigma_amr": 0.15,
            "use_alpha": False,
            "sigma_alpha": 0.10,
        }

        self.tau = float(tau)
        self.PP  = bool(PP)

        # Stubs the plotting helpers expect
        self.results: List[np.ndarray] = []
        self.mdf_data: List[Tuple[np.ndarray, np.ndarray]] = []
        self.labels: List[str] = []

        # Walker history holder for plotting code (keep it simple)
        self._walker_history_list: List[np.ndarray] = []
        self.walker_history = []

    # ------------------------- Observational MDF -------------------------

    def _load_obs_mdf_ga(self):
        """
        Load observational MDF exactly as used by the GA:
        file equal_weight_mdf.dat with two columns: [Fe/H], Normalized_Count
        """
        candidates = [
            Path("equal_weight_mdf.dat"),
            Path("data") / "equal_weight_mdf.dat",
            Path(self.output_path) / "equal_weight_mdf.dat",
        ]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError("equal_weight_mdf.dat not found.")

        arr = np.loadtxt(path)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError("equal_weight_mdf.dat must have 2+ columns: FeH, Normalized_Count")

        self.obs_feh = arr[:, 0].astype(float)
        self.obs_mdf_norm = np.clip(arr[:, 1].astype(float), 0, None)
        if not np.all(np.isfinite(self.obs_mdf_norm)) or self.obs_mdf_norm.max() <= 0:
            raise ValueError("Observed MDF contains non-finite or non-positive values")

    # ------------------- Unit-cube mapping (active params) -------------------

    def _u_to_phys(self, u: np.ndarray) -> Dict[str, float]:
        """Map u∈(0,1)^nd_active to physical params; add frozen ones."""
        phys = {}
        u = np.asarray(u, float)
        for k, d in enumerate(self._cont_active):
            lo, hi, log = d["lo"], d["hi"], d["log"]
            if log:
                a, b = np.log(lo), np.log(hi)
                phys[d["name"]] = float(np.exp(a + u[k]*(b - a)))
            else:
                phys[d["name"]] = float(lo + u[k]*(hi - lo))
        phys.update(self._cont_frozen)
        return phys

    def in_active_bounds(self, u: np.ndarray) -> bool:
        u = np.asarray(u, float)
        if u.shape[-1] != self.nd_active:
            return False
        return np.all((u > 0.0) & (u < 1.0) & np.isfinite(u))

    # --------------------------- Forward model -----------------------------

    def _forward_model(self, phys: Dict[str, float]):
        """
        Run omega_plus once and return:
          MDF_x, MDF_y (normalized PDF on [Fe/H])
          age_x (Gyr), age_y ([Fe/H])
          alpha_arrs: list of (Fe/H, [α/Fe]) for elements
        """
        kwargs = {
            'special_timesteps': self.timesteps,
            'twoinfall_sigmas': [1300, phys["sigma_2"]],
            'galradius': 1800,
            'exp_infall': [[self.A1, phys["tmax_1"] * 1e9, phys["infall_timescale_1"] * 1e9],
                           [self.A2, phys["tmax_2"] * 1e9, phys["infall_timescale_2"] * 1e9]],
            'tauup': [0.02e9, 0.02e9],
            'mgal': phys["mgal_values"],
            'iniZ': 0.0,
            'mass_loading': 0.0,
            'table': self.sn1a_header + self._yields,
            'sfe': phys["sfe"],
            'delta_sfe': phys["delta_sfe"],
            'imf_type': self._imf,
            'sn1a_table': self.sn1a_header + self._sn1a,
            'imf_yields_range': [1, phys["imf_upper_limits"]],
            'iniabu_table': self.iniab_header + self._comp,
            'nb_1a_per_m': phys["nb_array"],
            'sn1a_rate': self._sn1ar
        }
        GCE = omega_plus.omega_plus(**kwargs)

        MDF_x, MDF_y = GCE.inner.plot_mdf(axis_mdf='[Fe/H]', sigma_gauss=0.1,
                                          norm=True, return_x_y=True)
        MDF_x = np.asarray(MDF_x, float)
        MDF_y = np.asarray(MDF_y, float)

        age_x, age_y = GCE.inner.plot_spectro(xaxis='age', yaxis='[Fe/H]', return_x_y=True)
        age_x = np.asarray(age_x, float) / 1e9  # -> Gyr
        age_y = np.asarray(age_y, float)

        alpha_elems = ['[Si/Fe]','[Ca/Fe]','[Mg/Fe]','[Ti/Fe]']
        alpha_arrs = []
        for el in alpha_elems:
            ax, ay = GCE.inner.plot_spectro(xaxis='[Fe/H]', yaxis=el, return_x_y=True)
            alpha_arrs.append((np.asarray(ax, float), np.asarray(ay, float)))

        return MDF_x, MDF_y, age_x, age_y, alpha_arrs

    # ---------------------------- Likelihoods -----------------------------

    def _interp_model_to_obs(self, x_mod, y_mod, x_obs):
        return np.interp(x_obs, x_mod, y_mod, left=np.nan, right=np.nan)

    def _loglike_mdf(self, x_mod, y_mod) -> float:
        """
        Gaussian log-likelihood in normalized space.
        """
        x_obs = self.obs_feh
        y_obs = self.obs_mdf_norm
        y_model_on_obs = self._interp_model_to_obs(x_mod, y_mod, x_obs)
        m = np.isfinite(y_model_on_obs)
        if m.sum() < 6:
            return -np.inf
        sigma = float(self.cfg.get("sigma_mdf", 0.02))
        resid = y_model_on_obs[m] - y_obs[m]
        return float(-0.5 * np.sum((resid / sigma) ** 2 + np.log(2*np.pi*sigma**2)))

    def _loglike_amr(self, age_gyr, feh_curve):
        if not self.cfg.get("use_amr", False):
            return 0.0
        m = _finite_mask(age_gyr, feh_curve)
        if m.sum() < 10:
            return -np.inf
        sigma = float(self.cfg.get("sigma_amr", 0.15))
        d = np.diff(feh_curve[m])
        return float(-0.5 * np.sum((d / sigma)**2 + np.log(2*np.pi*sigma**2)))

    def _loglike_alpha(self, alpha_arrs):
        if not self.cfg.get("use_alpha", False):
            return 0.0
        sigma = float(self.cfg.get("sigma_alpha", 0.10))
        ll = 0.0
        for (x, y) in alpha_arrs:
            m = _finite_mask(x, y)
            if m.sum() < 10:
                continue
            d = np.diff(y[m])
            ll += -0.5 * np.sum((d / sigma)**2 + np.log(2*np.pi*sigma**2))
        return float(ll)

    def log_prob(self, u_active: np.ndarray) -> float:
        """
        Box prior on u∈(0,1)^nd. Likelihood = MDF (required) + optional AMR/α.
        """
        if self.nd_active == 0:
            return -np.inf
        u = np.asarray(u_active, float)
        if not self.in_active_bounds(u):
            return -np.inf

        phys = self._u_to_phys(u)
        try:
            mdf_x, mdf_y, age_x, age_y, alpha_arrs = self._forward_model(phys)
        except Exception:
            return -np.inf

        ll = self._loglike_mdf(mdf_x, mdf_y)
        if not np.isfinite(ll):
            return -np.inf
        ll += self._loglike_amr(age_x, age_y)
        ll += self._loglike_alpha(alpha_arrs)
        return float(ll / self.tau)

    # ------------------------------ Sampler --------------------------------

    def initial_positions(self, nwalkers: int, seed: int = 42):
        if self.nd_active <= 0:
            raise ValueError("All parameters are frozen; nothing to sample.")
        rng = np.random.default_rng(seed)
        return rng.uniform(0.1, 0.9, size=(nwalkers, self.nd_active))

    def run(self, nwalkers, nsteps, p0=None, out_h5=None, output_interval=None, PP=False):
        """
        Run emcee with StretchMove. Writes posterior CSV + corner + GA plots.
        """
        if p0 is None:
            p0 = self.initial_positions(nwalkers, seed=42)

        # Backend (safe)
        backend = None
        resume = False
        if out_h5:
            from emcee.backends import HDFBackend
            out_dir = os.path.dirname(out_h5) or "."
            os.makedirs(out_dir, exist_ok=True)
            existing = os.path.exists(out_h5)
            backend = HDFBackend(out_h5)
            if existing and backend.iteration > 0:
                resume = True
                print(f"Resuming with {backend.iteration} iterations at {out_h5}")

        pool = None
        if PP:
            import multiprocessing as mp
            pool = mp.Pool()

        try:
            from emcee.moves import StretchMove
            move = StretchMove(a=2.0)
            sampler = emcee.EnsembleSampler(
                nwalkers, self.nd_active, self.log_prob,
                moves=move, backend=backend, pool=pool
            )
            if not resume:
                sampler.run_mcmc(p0, 1, progress=False)

            done0 = backend.iteration if backend is not None else 1
            done = done0
            while done < nsteps:
                step = min(output_interval or 128, nsteps - done)
                sampler.run_mcmc(None, step, progress=False)
                done += step
                # record for walker-history compatible stub
                try:
                    last = sampler.get_last_sample()
                    self._walker_history_list.append(last.coords.copy())
                    self.walker_history = list(self._walker_history_list)
                except Exception:
                    pass
                self._save_outputs(sampler, step=done)

            return sampler
        finally:
            if pool is not None:
                pool.close(); pool.join()

    # ------------------------------ Outputs --------------------------------

    def _flat_chain(self, sampler, thin=1, discard=None):
        """
        Return flat samples/logp with a sane early-iteration burn-in policy.
        - If iteration < 50, don't discard.
        - Otherwise discard at most half, but never >= iteration.
        """
        it = int(getattr(sampler, "iteration", 0))
        if discard is None:
            if it < 50:
                discard = 0
            else:
                discard = min(max(100, it // 2), it - 2)
        # absolute safety: never discard >= iteration or negative
        discard = max(0, min(discard, max(0, it - 1)))

        try:
            chain = sampler.get_chain(discard=discard, flat=True, thin=thin)
            logp  = sampler.get_log_prob(discard=discard, flat=True, thin=thin)
            m = np.isfinite(logp)
            if m.sum() == 0:
                return None, None
            return chain[m], logp[m]
        except Exception:
            return None, None


    def _active_chain_to_physical(self, flat: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        if flat is None or flat.size == 0:
            return None, []
        cols, labels = [], []
        for j, d in enumerate(self._cont_active):
            lo, hi, log = d["lo"], d["hi"], d["log"]
            u = flat[:, j]
            if log:
                a, b = np.log(lo), np.log(hi)
                x = np.exp(a + u*(b - a))
            else:
                x = lo + u*(hi - lo)
            cols.append(x); labels.append(d["name"])
        return np.column_stack(cols), labels

    def _save_outputs(self, sampler, step: int):
        outdir = Path(self.output_path); outdir.mkdir(parents=True, exist_ok=True)

        flat, logp = self._flat_chain(sampler, thin=1, discard=None)
        if flat is None or flat.size == 0:
            print(f"[skip] No samples at step {step}")
            return

        # 1) CSV of unit-cube samples (active only) + logprob
        df = pd.DataFrame(flat, columns=self.param_names)
        df["logprob"] = logp
        df.to_csv(outdir / f"posteriors_step_{step}.csv", index=False)

        # 2) Corner of physical parameters
        X_phys, labels = self._active_chain_to_physical(flat)
        if X_phys is not None and X_phys.shape[1] >= 1:
            try:
                import corner
                uniq_ok = all(np.unique(X_phys[:, j]).size >= 8 for j in range(X_phys.shape[1]))
                with_contours = (uniq_ok and X_phys.shape[0] >= 500)
                fig = corner.corner(
                    X_phys, labels=labels, show_titles=True,
                    quantiles=[0.16, 0.50, 0.84],
                    plot_contours=with_contours, fill_contours=with_contours,
                    plot_datapoints=True
                )
                #fig.savefig(outdir / f"corner_step_{step}.png", dpi=180, bbox_inches="tight")
                fig.savefig(outdir / f"corner.png", dpi=180, bbox_inches="tight")                
                plt.close(fig)
            except Exception as e:
                print(f"[corner] skipped: {e}")

        # 3) Best posterior point -> GA-style plots
        i_best = int(np.argmax(logp))
        phys_best = self._u_to_phys(flat[i_best])
        try:
            mdf_x, mdf_y, age_x, age_y, alpha_arrs = self._forward_model(phys_best)
        except Exception:
            return

        # provide GA-like attributes
        self.mdf_data = [(mdf_x, mdf_y)]
        self.age_data = [(age_x * 1e9, age_y)]
        self.alpha_data = [alpha_arrs]
        # GA results row: 15-element vector with continuous values at positions
        # [5]=sigma_2, [7]=t_2, [9]=infall_2 (used by plot_mdf_curves to tag "best")
        row = np.zeros(15, dtype=float)
        name_to_idx = {
            "sigma_2":5, "tmax_1":6, "tmax_2":7, "infall_timescale_1":8, "infall_timescale_2":9,
            "sfe":10, "delta_sfe":11, "imf_upper_limits":12, "mgal_values":13, "nb_array":14,
        }
        for d in self._cont_active:
            row[name_to_idx[d["name"]]] = phys_best[d["name"]]
        for k,v in self._cont_frozen.items():
            if k in name_to_idx: row[name_to_idx[k]] = v
        self.results = [row]
        self.labels = ["best"]

        # MDF overlay (GA-style)
        try:
            #plot_mdf_curves(self, self.obs_feh, self.obs_mdf_norm, results_df=None, save_path=str(outdir / f"MDF_fit_step_{step}.png"))
            plot_mdf_curves(self, self.obs_feh, self.obs_mdf_norm, results_df=None, save_path=str(outdir / f"MDF_fit.png"))            
        except Exception as e:
            print(f"[mdf plot] skipped: {e}")

        # AMR+alpha compact panel (GA-style)
        try:
            plot_omni_figure(self,
                             # pass MDF obs so function can annotate; it loads α data internally
                             Fe_H=np.array([]), age_Joyce=np.array([]), age_Bensby=np.array([]),
                             Mg_Fe=np.array([]), Si_Fe=np.array([]), Ca_Fe=np.array([]), Ti_Fe=np.array([]),
                             feh_mdf=self.obs_feh, normalized_count_mdf=self.obs_mdf_norm,
                             results_df=None,
                             save_path=str(outdir / f"AMR_alpha.png"))
                             #save_path=str(outdir / f"AMR_alpha_step_{step}.png"))
        except Exception as e:
            print(f"[AMR/alpha] plot skipped: {e}")

    # Compatibility with launcher: save CSV + call generate_all_plots
    def save_partial_results(self, generation: int):
        """
        Build a GA-like results table with a single 'best' row so that
        mdf_plotting.generate_all_plots can run without GA baggage.
        """
        outdir = Path(self.output_path)
        outdir.mkdir(parents=True, exist_ok=True)

        # If there is a posterior CSV for this step, prefer its MAP
        post = outdir / f"posteriors_step_{generation}.csv"
        if post.is_file():
            dfp = pd.read_csv(post)
            if "logprob" in dfp.columns and dfp.shape[0] > 0:
                ib = int(np.argmax(dfp["logprob"].values))
                u = dfp[self.param_names].values[ib]
                phys = self._u_to_phys(u)
            else:
                phys = {d["name"]: (d["lo"]+d["hi"])/2 for d in self._cont_active}
                phys.update(self._cont_frozen)
        else:
            phys = {d["name"]: (d["lo"]+d["hi"])/2 for d in self._cont_active}
            phys.update(self._cont_frozen)

        # Evaluate model once
        try:
            mdf_x, mdf_y, age_x, age_y, alpha_arrs = self._forward_model(phys)
        except Exception:
            # write an empty csv to keep pipeline moving
            (outdir / f"mcmc_results_step_{generation}.csv").write_text("comp_idx,imf_idx,sn1a_idx,sy_idx,sn1ar_idx,sigma_2,t_1,t_2,infall_1,infall_2,sfe,delta_sfe,imf_upper,mgal,nb,fitness\n")
            return

        # Single-row results matching GA CSV schema
        row = {
            'comp_idx': 0, 'imf_idx': 0, 'sn1a_idx': 0, 'sy_idx': 0, 'sn1ar_idx': 0,
            'sigma_2': phys.get("sigma_2", np.nan),
            't_1': phys.get("tmax_1", np.nan),
            't_2': phys.get("tmax_2", np.nan),
            'infall_1': phys.get("infall_timescale_1", np.nan),
            'infall_2': phys.get("infall_timescale_2", np.nan),
            'sfe': phys.get("sfe", np.nan),
            'delta_sfe': phys.get("delta_sfe", np.nan),
            'imf_upper': phys.get("imf_upper_limits", np.nan),
            'mgal': phys.get("mgal_values", np.nan),
            'nb': phys.get("nb_array", np.nan),
            'ks': np.nan, 'ensemble': np.nan, 'wrmse': np.nan, 'mae': np.nan, 'mape': np.nan,
            'huber': np.nan, 'cosine': np.nan, 'log_cosh': np.nan,
            'fitness': 0.0, 'age_meta_fitness': 0.0, 'physics_penalty': 0.0
        }
        df = pd.DataFrame([row])
        results_file = outdir / f"mcmc_results_step_{generation}.csv"
        df.to_csv(results_file, index=False)

        # Supply attributes for GA plotting
        self.mdf_data = [(mdf_x, mdf_y)]
        self.results = [np.array([
            row['comp_idx'], row['imf_idx'], row['sn1a_idx'], row['sy_idx'], row['sn1ar_idx'],
            row['sigma_2'], row['t_1'], row['t_2'], row['infall_1'], row['infall_2'],
            row['sfe'], row['delta_sfe'], row['imf_upper'], row['mgal'], row['nb'],
            0, 0, 0, 0, 0, 0, 0, 0, row['fitness'], 0, 0
        ], dtype=float)]
        self.labels = ["best"]

        # Call the GA plotting dashboard
        try:
            mdf_plotting.generate_all_plots(self, self.obs_feh, self.obs_mdf_norm, str(results_file))
        except Exception as e:
            print(f"[warn] generate_all_plots failed: {e}")
