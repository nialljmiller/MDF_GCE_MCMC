#!/usr/bin/env python3

import ast
import os
import re
import shutil
import subprocess
from itertools import product

def parse_pcard(file_path):
    params = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                try:
                    params[key] = ast.literal_eval(value)
                except:
                    params[key] = value
    return params

def write_pcard(params, output_path, base_lines):
    with open(output_path, 'w') as f:
        for line in base_lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and ':' in stripped:
                key = stripped.split(':', 1)[0].strip()
                if key in params:
                    value = params[key]
                    if isinstance(value, (list, tuple)):
                        value_str = str(list(value))
                    else:
                        value_str = repr(value)
                    f.write(f"{key}: {value_str}\n")
                    continue
            f.write(line)

def main():
    base_pcard = 'bulge_pcard.txt'
    if not os.path.exists(base_pcard):
        print(f"Base pcard file not found: {base_pcard}")
        return

    with open(base_pcard, 'r') as f:
        base_lines = f.readlines()

    params = parse_pcard(base_pcard)

    cat_keys = [
        'comp_array',
        'imf_array',
        'sn1a_assumptions',
        'stellar_yield_assumptions',
        'sn1a_rates'
    ]

    cat_dict = {k: params.get(k, []) for k in cat_keys if k in params and params[k]}

    if not cat_dict:
        print("No categorical parameters found in pcard.")
        return

    cat_lists = list(cat_dict.values())
    cat_names = list(cat_dict.keys())

    combinations = list(product(*cat_lists))

    print(f"Found {len(cat_lists)} categorical parameters.")
    print(f"Total unique combinations: {len(combinations)}")

    base_output_path = params.get('output_path', 'mcmc_out/').rstrip('/')

    for combo in combinations:
        combo_dict = dict(zip(cat_names, combo))

        combo_name_parts = []
        for k, v in combo_dict.items():
            short_k = k.replace('_array', '').replace('_assumptions', '').replace('_rates', '')
            if isinstance(v, float):
                v_str = f"{v:.1f}".replace('.', 'p')
            elif isinstance(v, int):
                v_str = str(v)
            else:
                v_str = str(v).lower().replace(' ', '_')
            combo_name_parts.append(f"{short_k}_{v_str}")

        combo_name = '_'.join(combo_name_parts)

        config_dir = f"config_{combo_name}"
        os.makedirs(config_dir, exist_ok=True)

        new_pcard_path = os.path.join(config_dir, 'bulge_pcard.txt')

        new_params = params.copy()
        for k, v in combo_dict.items():
            new_params[k] = [v]  # Set to singleton list to fix the choice

        new_params['output_path'] = f"{base_output_path}_{combo_name}/"

        write_pcard(new_params, new_pcard_path, base_lines)

        submit_cmd = ['python', 'submit_mcmc.py', os.path.abspath(config_dir)]
        print(f"\nSubmitting job for combination: {combo_name}")
        print(f"Config dir: {config_dir}")
        print(f"Output path: {new_params['output_path']}")
        print(f"Command: {' '.join(submit_cmd)}")

        try:
            result = subprocess.run(submit_cmd, capture_output=True, text=True, check=True)
            print(result.stdout)
            if result.stderr:
                print(f"Warning: {result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"Submission failed for {combo_name}: {e}")
            print(e.stdout)
            print(e.stderr)
        except Exception as e:
            print(f"Unexpected error submitting {combo_name}: {e}")

if __name__ == "__main__":
    main()