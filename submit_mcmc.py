#!/usr/bin/env python3
import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
import os

SBATCH_FILENAME = "mcmc.sbatch"  # default; override with -s

# ---------- helpers ----------

def bash_single_quote(s: str) -> str:
    # Safe single-quoting for bash, including embedded apostrophes.
    return "'" + s.replace("'", "'\"'\"'") + "'"

def sanitize_job_suffix(s: str, max_len: int = 60) -> str:
    """
    Make a SLURM-friendly suffix from file stem:
    keep [A-Za-z0-9_-], map others to '-', collapse repeats, trim length.
    """
    s = re.sub(r"[^A-Za-z0-9_-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:max_len] if len(s) > max_len else s

def set_job_name(sbatch_text: str, suffix: str) -> str:
    """
    If a job-name exists, append '-{suffix}' (truncated if needed).
    If not, insert a job-name after the shebang or at top.
    SLURM allows up to 128 chars for --job-name.
    """
    job_re = re.compile(r"^(#SBATCH\s+--job-name=)(.*)$", flags=re.MULTILINE)
    m = job_re.search(sbatch_text)
    if m:
        base = m.group(2).strip()
        # If base already has the suffix, leave it; else append.
        if not base.endswith("-" + suffix):
            new = f"{base}-{suffix}"
        else:
            new = base
        # Truncate to 128 characters (conservative).
        if len(new) > 128:
            new = new[:128]
        return job_re.sub(rf"\1{new}", sbatch_text, count=1)

    # No existing job-name; inject one near top (after shebang if present)
    lines = sbatch_text.splitlines(keepends=True)
    directive = f"#SBATCH --job-name={suffix}\n"
    if lines and lines[0].startswith("#!"):
        return "".join([lines[0], directive, *lines[1:]])
    return directive + "".join(lines)

def set_file_path(sbatch_text: str, file_path: str) -> str:
    """
    Replace FILE_PATH=... if present; else insert above first srun.
    """
    quoted = bash_single_quote(file_path)
    pat = re.compile(r"^(FILE_PATH\s*=).*$", flags=re.MULTILINE)
    if pat.search(sbatch_text):
        return pat.sub(rf"\1{quoted}", sbatch_text, count=1)

    srun_pat = re.compile(r"^\s*srun\b.*$", flags=re.MULTILINE)
    m = srun_pat.search(sbatch_text)
    inject = f"FILE_PATH={quoted}\n\n"
    if m:
        start = m.start()
        return sbatch_text[:start] + inject + sbatch_text[start:]
    # fallback: append
    return sbatch_text + ("" if sbatch_text.endswith("\n") else "\n") + inject

def submit(temp_path: pathlib.Path) -> int:
    proc = subprocess.run(
        ["sbatch", str(temp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        # show stderr/stdout to help debugging
        sys.stderr.write(proc.stderr or proc.stdout)
        raise SystemExit(proc.returncode)

    m = re.search(r"Submitted batch job\s+(\d+)", proc.stdout or "")
    if not m:
        print(proc.stdout.strip())
        raise SystemExit("Could not parse job ID from sbatch output.")
    jobid = int(m.group(1))
    print(f"Submitted batch job {jobid}")
    return jobid

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Inject FILE_PATH and job-name into sbatch, submit, and archive the submitted sbatch next to the input file."
    )
    ap.add_argument("file_path", help="Path to pass into the Slurm job")
    ap.add_argument("cluster_pc", default=0)
    ap.add_argument(
        "-s", "--sbatch",
        default=SBATCH_FILENAME,
        help=f"Path to the sbatch script (default: {SBATCH_FILENAME})",
    )

    args = ap.parse_args()

    sbatch_path = pathlib.Path(args.sbatch).expanduser().resolve()
    if not sbatch_path.is_file():
        print(f"sbatch script not found: {sbatch_path}")

    user_file = pathlib.Path(args.file_path).expanduser().resolve()
    if not user_file.exists():
        raise SystemExit(f"Input file does not exist: {user_file}")

    file_abs = str(user_file)

    # Build modified sbatch text
    sbatch_text = sbatch_path.read_text()

    # 1) set FILE_PATH
    sbatch_text = set_file_path(sbatch_text, file_abs)

    # 2) set/extend job-name
    suffix = sanitize_job_suffix(user_file.stem)
    sbatch_text = set_job_name(sbatch_text, suffix)

    cluster_pc = int(args.cluster_pc)


    if cluster_pc == 1:
    
        # Write temp sbatch in same dir as original to preserve any relative references
        with tempfile.NamedTemporaryFile(
            "w", prefix=sbatch_path.stem + "_", suffix=".sbatch", dir=str(sbatch_path.parent), delete=False
        ) as tf:
            tf.write(sbatch_text)
            temp_sbatch = pathlib.Path(tf.name)

        # Submit
        try:
            jobid = submit(temp_sbatch)
        except Exception:
            # leave temp file for debugging
            raise

        # Archive/move the submitted sbatch next to the input file
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest_dir = user_file.parent / "submitted_jobs"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_name = f"{sbatch_path.stem}_{suffix}_{ts}_jid{jobid}.sbatch"
        dest_path = dest_dir / dest_name
        

        try:
            shutil.move(str(temp_sbatch), str(dest_path))
            print(f"Archived submitted sbatch to: {dest_path}")
        except Exception as e:
            print(f"Warning: failed to move submitted sbatch to archive ({e}). Left at: {temp_sbatch}")

        # Final echo for easy grep in logs
        print(f"JOBID={jobid}")
        print(f"INPUT_FILE={user_file}")
        print(f"SBATCH_ARCHIVE={dest_path if dest_path.exists() else temp_sbatch}")



    else:

        submit_cmd = ['python', 'MDF_MCMC_Launcher.py', os.path.abspath(args.file_path)]

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
