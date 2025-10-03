#!/usr/bin/env bash
# Usage: ./grid_mcmc.sh [pcard] [root_out] [sbatch_script]
# Defaults: pcard=./bulge_pcard.txt, root_out=./mcmc_runs, sbatch_script=./mcmc_sbatch.sh

set -euo pipefail

PCARD="${1:-bulge_pcard.txt}"
ROOT_OUT="${2:-mcmc_runs}"
SBATCH_SCRIPT="${3:-mcmc_sbatch.sh}"

[[ -f "$PCARD" ]] || { echo "No pcard: $PCARD" >&2; exit 1; }
[[ -f "$SBATCH_SCRIPT" ]] || { echo "No sbatch script: $SBATCH_SCRIPT" >&2; exit 1; }

mkdir -p "$ROOT_OUT"

# Return list (space-separated) from a key in pcard; accepts commas/spaces/semicolons
get_list () {
  local key="$1"
  awk -F= -v k="^${key}[[:space:]]*" '
    $1 ~ k {
      gsub(/[;\t]/,",",$2);
      gsub(/[[:space:]]+/," ",$2);
      gsub(/^[[:space:]]+|[[:space:]]+$/,"",$2);
      gsub(/,+/," ",$2);
      print $2
    }' "$PCARD"
}

# Standard categorical parameters used across your stacks
COMP_LIST=( $(get_list comp_array) )
IMF_LIST=( $(get_list imf_array) )
SN1A_LIST=( $(get_list sn1a_assumptions) )
YIELD_LIST=( $(get_list stellar_yield_assumptions) )
SN1AR_LIST=( $(get_list sn1a_rates) )
MGAL_LIST=( $(get_list mgal_values) )
NB_LIST=( $(get_list nb_array) )

# Fallback to single 'default' so the loops still run if a list is empty
[[ ${#COMP_LIST[@]} -eq 0 ]] && COMP_LIST=(default)
[[ ${#IMF_LIST[@]}  -eq 0 ]] && IMF_LIST=(default)
[[ ${#SN1A_LIST[@]} -eq 0 ]] && SN1A_LIST=(default)
[[ ${#YIELD_LIST[@]} -eq 0 ]] && YIELD_LIST=(default)
[[ ${#SN1AR_LIST[@]} -eq 0 ]] && SN1AR_LIST=(default)
[[ ${#MGAL_LIST[@]} -eq 0 ]] && MGAL_LIST=(default)
[[ ${#NB_LIST[@]}   -eq 0 ]] && NB_LIST=(default)

submit_count=0

for comp in "${COMP_LIST[@]}"; do
for imf in "${IMF_LIST[@]}"; do
for s1a in "${SN1A_LIST[@]}"; do
for sy in "${YIELD_LIST[@]}"; do
for s1ar in "${SN1AR_LIST[@]}"; do
for mgal in "${MGAL_LIST[@]}"; do
for nb in "${NB_LIST[@]}"; do

  TAG="comp=${comp}__imf=${imf}__sn1a=${s1a}__yields=${sy}__sn1ar=${s1ar}__mgal=${mgal}__nb=${nb}"
  RUN_DIR="${ROOT_OUT}/${TAG}"
  mkdir -p "${RUN_DIR}/logs"

  # Write per-run pcard with categorical picks + output_folder
  PC_DST="${RUN_DIR}/bulge_pcard.txt"
  cp "$PCARD" "$PC_DST"

  # Replace keys ONLY if they exist in the source pcard
  sed -i -E \
    -e "s|^(comp_array)[[:space:]]*=.*|\1 = ${comp}|;t" \
    -e "s|^(imf_array)[[:space:]]*=.*|\1 = ${imf}|;t" \
    -e "s|^(sn1a_assumptions)[[:space:]]*=.*|\1 = ${s1a}|;t" \
    -e "s|^(stellar_yield_assumptions)[[:space:]]*=.*|\1 = ${sy}|;t" \
    -e "s|^(sn1a_rates)[[:space:]]*=.*|\1 = ${s1ar}|;t" \
    -e "s|^(mgal_values)[[:space:]]*=.*|\1 = ${mgal}|;t" \
    -e "s|^(nb_array)[[:space:]]*=.*|\1 = ${nb}|;t" \
    "$PC_DST"

  # Force output_folder to the run dir (relative is fine since we --chdir)
  if grep -qE '^[[:space:]]*output_folder[[:space:]]*=' "$PC_DST"; then
    sed -i -E "s|^(output_folder)[[:space:]]*=.*|\1 = ./|g" "$PC_DST"
  else
    printf "\noutput_folder = ./\n" >> "$PC_DST"
  fi

  # Submit: chdir into run dir, use job name = tag, logs per run
  sbatch --chdir "$RUN_DIR" \
         --job-name "$TAG" \
         --output "logs/%x_%j.out" \
         --error  "logs/%x_%j.err" \
         "$SBATCH_SCRIPT"

  ((submit_count++))
  echo "Submitted: $TAG"

done; done; done; done; done; done; done

echo "Done. Submitted ${submit_count} jobs."
