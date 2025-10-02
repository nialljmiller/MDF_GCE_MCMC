#!/usr/bin/env bash
# grid_mcmc.sh — sweep categorical combos, rewrite pcard, submit with sbatch.
# Usage: ./grid_mcmc.sh [path/to/bulge_pcard.txt] [root_out_dir]
# Defaults: pcard=./bulge_pcard.txt, root_out=runs_mcmc/

set -euo pipefail

PCARD="${1:-bulge_pcard.txt}"
ROOT_OUT="${2:-runs_mcmc}"
SBATCH_SCRIPT="$(pwd)/mcmc_sbatch.sh"   # absolute path to keep sbatch happy

if [[ ! -f "$PCARD" ]]; then
  echo "Error: pcard not found: $PCARD" >&2
  exit 1
fi
if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "Error: mcmc_sbatch.sh not found at $SBATCH_SCRIPT" >&2
  exit 1
fi

mkdir -p "$ROOT_OUT"

# --- Pull categorical lists directly from the pcard using Python (robust to quoting) ---
readarray -t COMP_LIST < <(python3 - <<'PY' "$PCARD"
import ast, sys, re
pc = sys.argv[1]
txt = open(pc,'r',encoding='utf-8').read()
def grab(key):
    m = re.search(rf'^\s*{re.escape(key)}\s*:\s*(.+)$', txt, flags=re.MULTILINE)
    if not m: return []
    raw = m.group(1).split('#',1)[0].strip()
    try:
        v = ast.literal_eval(raw)
    except Exception:
        v = raw.strip().strip("'\"")
    if isinstance(v, (list, tuple)): return [str(x) for x in v]
    return [str(v)]
for key in ("comp_array","imf_array","sn1a_assumptions","stellar_yield_assumptions","sn1a_rates"):
    vals = grab(key)
    print("__KEY__", key)
    for x in vals: print(x)
PY
)
# COMP_LIST is a flattened block like:
# __KEY__ comp_array
# val1
# val2
# __KEY__ imf_array
# val1
# ...

# Split the flattened block into separate Bash arrays
extract_block () {
  local key="$1"; shift
  local -n outarr="$1"; shift
  local in_key=0
  outarr=()
  for line in "${COMP_LIST[@]}"; do
    if [[ "$line" == "__KEY__ "* ]]; then
      in_key=0
      [[ "$line" == "__KEY__ $key" ]] && in_key=1
      continue
    fi
    (( in_key )) && outarr+=("$line")
  done
}

declare -a COMP IMFS SNIa SY SNIaR
extract_block "comp_array" COMP
extract_block "imf_array" IMFS
extract_block "sn1a_assumptions" SNIa
extract_block "stellar_yield_assumptions" SY
extract_block "sn1a_rates" SNIaR

# Sanity check
for arrname in COMP IMFS SNIa SY SNIaR; do
  eval "len=\${#${arrname}[@]}"
  if (( len == 0 )); then
    echo "Warning: no entries found for $arrname — will proceed with a single blank value." >&2
  fi
done

# Escape a value for safe sed replacement (slashes and ampersands)
sed_escape () {
  printf '%s' "$1" | sed -E 's/[\/&]/\\&/g'
}

# Write a per-combo pcard with categorical singletons and a unique output_path
write_pcard () {
  local src="$1" dst="$2" outdir="$3"
  local comp="$4" imf="$5" s1a="$6" sy="$7" s1ar="$8"

  cp "$src" "$dst"

  # Replace output_path (single-quoted) — tolerate missing existing line by appending
  if grep -qE '^\s*output_path\s*:' "$dst"; then
    sed -i -E "s|^\s*output_path\s*:\s*.*$|output_path: '$(sed_escape "$outdir/")'|g" "$dst"
  else
    printf "\noutput_path: '%s'\n" "$outdir/" >> "$dst"
  fi

  # Force each categorical to a single-element list (strings)
  sed -i -E "s|^\s*comp_array\s*:\s*.*$|comp_array: ['$(sed_escape "$comp")']|g" "$dst"
  sed -i -E "s|^\s*imf_array\s*:\s*.*$|imf_array: ['$(sed_escape "$imf")']|g" "$dst"
  sed -i -E "s|^\s*sn1a_assumptions\s*:\s*.*$|sn1a_assumptions: ['$(sed_escape "$s1a")']|g" "$dst"
  sed -i -E "s|^\s*stellar_yield_assumptions\s*:\s*.*$|stellar_yield_assumptions: ['$(sed_escape "$sy")']|g" "$dst"
  sed -i -E "s|^\s*sn1a_rates\s*:\s*.*$|sn1a_rates: ['$(sed_escape "$s1ar")']|g" "$dst"
}

# Compact tag making (letters to keep paths short)
tagify () {
  # lowercase, replace spaces/slashes with dashes, strip odd chars
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's|[ /]|-|g; s|[^a-z0-9._-]||g'
}

# Iterate all categorical combinations (5 nested)
submit_count=0
for comp in "${COMP[@]:-""}"; do
  for imf in "${IMFS[@]:-""}"; do
    for s1a in "${SNIa[@]:-""}"; do
      for sy in "${SY[@]:-""}"; do
        for s1ar in "${SNIaR[@]:-""}"; do

          t_comp="$(tagify "$comp")"
          t_imf="$(tagify "$imf")"
          t_s1a="$(tagify "$s1a")"
          t_sy="$(tagify "$sy")"
          t_s1ar="$(tagify "$s1ar")"

          RUN_TAG="c_${t_comp}__i_${t_imf}__a_${t_s1a}__y_${t_sy}__r_${t_s1ar}"
          RUN_DIR="${ROOT_OUT}/${RUN_TAG}"
          PC_DST="${RUN_DIR}/bulge_pcard.txt"

          mkdir -p "$RUN_DIR"
          write_pcard "$PCARD" "$PC_DST" "$RUN_DIR" "$comp" "$imf" "$s1a" "$sy" "$s1ar"

          # Submit from inside the run dir so relative paths inside your stack still work
          ( cd "$ROOT_OUT" && sbatch "$SBATCH_SCRIPT" && "$RUN_DIR")
          ((submit_count++))
          echo "Submitted: ${RUN_TAG}"
        done
      done
    done
  done
done

echo "Done. Submitted ${submit_count} jobs."
