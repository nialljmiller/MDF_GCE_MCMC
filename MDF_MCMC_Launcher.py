#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import shutil
import ast
import numpy as np
import pandas as pd
import mdf_plotting
from MDF_MCMC import GalacticEvolutionMCMC
from multiprocessing import cpu_count


def load_bensby_data(file_path='data/Bensby_Data.tsv'):
    return pd.read_csv(file_path, sep='\t')


def parse_inlist(path):
    data = {}
    with open(path, 'r') as f:
        for raw in f:
            line = raw.split('#', 1)[0].strip()
            if not line or ':' not in line:
                continue
            k, v = line.split(':', 1)
            k = k.strip(); v = v.strip()
            if not v:
                continue
            try:
                data[k] = ast.literal_eval(v)
            except Exception:
                data[k] = v.strip().strip("'\"")
    defaults = {
        'loss_metric': 'huber',
        'obs_age_data_loss_metric': None,
        'obs_age_data_target': 'joyce',
        'mdf_vs_age_weight': 1.0,
        'output_path': 'mcmc_out/',
        'gaussian_sigma_scale': 0.02,
        'cat_move_prob': 0.2,
        'tau': 1.0,
        'steps': 2000,
        'walkers': -1,
    }
    for k, v in defaults.items():
        data.setdefault(k, v)
    return data



def _resolve_pcard_path(default='bulge_pcard.txt'):
    for a in sys.argv[1:]:
        if a and not a.startswith('-'):
            return a
    for i, a in enumerate(sys.argv[1:], start=1):
        if a.startswith('--pcard='):
            return a.split('=', 1)[1]
        if a == '--pcard' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def main():
    pcard_path = _resolve_pcard_path('bulge_pcard.txt')
    params = parse_inlist(pcard_path)

    # Normalize scalars -> lists (freezing handled in MDF_MCMC)
    def _ensure_list(x):
        if isinstance(x, (list, tuple)): return list(x)
        return [x]

    scalar_synonyms = {
        'sigma_2':'sigma_2_list',
        'tmax_1':'tmax_1_list',
        'tmax_2':'tmax_2_list',
        'infall_timescale_1':'infall_timescale_1_list',
        'infall_timescale_2':'infall_timescale_2_list',
    }
    for k_scalar, k_list in scalar_synonyms.items():
        if k_scalar in params and k_list not in params:
            params[k_list] = _ensure_list(params[k_scalar])

    cont_keys = [
        'sigma_2_list','tmax_1_list','tmax_2_list',
        'infall_timescale_1_list','infall_timescale_2_list',
        'sfe_array','delta_sfe_array','imf_upper_limits','mgal_values','nb_array'
    ]
    for k in cont_keys:
        if k in params:
            params[k] = _ensure_list(params[k])

    cat_keys = ['comp_array','imf_array','sn1a_assumptions','stellar_yield_assumptions','sn1a_rates']
    for k in cat_keys:
        if k in params and not isinstance(params[k], (list, tuple)):
            params[k] = [params[k]]

    # Output path
    output_path = params['output_path']
    os.makedirs(output_path, exist_ok=True)
    try:
        shutil.copy(pcard_path, os.path.join(output_path, 'bulge_pcard.txt'))
    except Exception:
        pass

    # Observational MDF
    feh, count = np.loadtxt(params['obs_file'], usecols=(0, 1), unpack=True)
    normalized_count = count / count.max()

    # Observational age-[Fe/H]
    try:
        obs_age_data = load_bensby_data('data/Bensby_Data.tsv')
    except Exception:
        obs_age_data = load_bensby_data('/project/galacticbulge/MDF_GCE_GA/data/Bensby_Data.tsv')

    # Emcee knobs (walker count auto if -1)
    ndim_nominal = 15
    walkers = int(params.get('walkers', -1)) if str(params.get('walkers', -1)).strip() != '' else -1
    if walkers < 0:
        walkers = max(64, 4 * ndim_nominal)
    if walkers < 2 * ndim_nominal:
        walkers = 2 * ndim_nominal
    if walkers % 2 != 0:
        walkers += 1


    steps = int(params.get('steps', 2000)) if str(params.get('steps', '')).strip() != '' else 2000
    output_interval = params.get('output_interval', None)
    try:
        output_interval = int(output_interval) if output_interval is not None else None
    except Exception:
        output_interval = None

    tau = float(params.get('tau', 1.0)) if str(params.get('tau', '')).strip() != '' else 1.0
    cat_move_prob = float(params.get('cat_move_prob', 0.2))
    frac_of_range = float(params.get('gaussian_sigma_scale', 0.02))

    out_h5 = os.path.join(output_path, 'chain.h5')
    resume = os.path.exists(out_h5)

    mcmc = GalacticEvolutionMCMC(
        output_path=output_path,
        sn1a_header=params['sn1a_header'],
        iniab_header=params['iniab_header'],
        sigma_2_list=params['sigma_2_list'],
        tmax_1_list=params['tmax_1_list'],
        tmax_2_list=params['tmax_2_list'],
        infall_timescale_1_list=params['infall_timescale_1_list'],
        infall_timescale_2_list=params['infall_timescale_2_list'],
        comp_array=params['comp_array'],
        imf_array=params['imf_array'],
        sfe_array=params['sfe_array'],
        delta_sfe_array=params['delta_sfe_array'],
        imf_upper_limits=params['imf_upper_limits'],
        sn1a_assumptions=params['sn1a_assumptions'],
        stellar_yield_assumptions=params['stellar_yield_assumptions'],
        mgal_values=params['mgal_values'],
        nb_array=params['nb_array'],
        sn1a_rates=params['sn1a_rates'],
        timesteps=params['timesteps'],
        A1=params['A1'], A2=params['A2'],
        feh=feh, normalized_count=normalized_count,
        obs_counts=count.astype(int),
        obs_age_data=obs_age_data,
        tau=tau,
        PP=False
    )

    p0 = mcmc.initial_positions(nwalkers=walkers, seed=42)

    sampler = mcmc.run(
        nwalkers=walkers,
        nsteps=steps,
        p0=None if resume else p0,
        out_h5=out_h5,
        output_interval=output_interval,
        PP=False
    )


    print(f"Done. Walkers={walkers}, Steps={steps}, Resume={'yes' if resume else 'no'}\nHDF5 chain: {out_h5}")


if __name__ == "__main__":
    main()
