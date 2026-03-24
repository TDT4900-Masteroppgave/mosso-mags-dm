import subprocess
import urllib.request
import gzip
import glob
import pandas as pd

from scripts.config import *
import logging
import sys

def get_fastutil_path():
    fastutil_files = glob.glob("fastutil-*.jar")
    return fastutil_files[0] if fastutil_files else "fastutil-missing.jar"


def setup_directories():
    for d in [DATASETS_DIR, OUTPUT_DIR, BENCHMARK_DIR, VERSIONS_DIR]:
        os.makedirs(d, exist_ok=True)


def retrieve_github_code(target_dir: str, algo_name: str, repo_url: str, branch: str, logger):
    try:
        if not os.path.exists(target_dir):
            logger.info(f"    -> [{algo_name}] Target directory not found. Cloning fresh...")
            subprocess.run(["git", "clone", "-q", "--branch", branch, "--single-branch", repo_url, target_dir],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        else:
            logger.info(f"    -> [{algo_name}] Target directory exists. Pulling latest updates...")
            subprocess.run(["git", "pull", "-q"], cwd=target_dir,
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        raise e


def download_and_prepare_dataset(url, filename, logger):
    gz_path = os.path.join(DATASETS_DIR, filename + ".gz")
    txt_path = os.path.join(DATASETS_DIR, filename)

    if not os.path.exists(txt_path):
        try:
            if not os.path.exists(gz_path):
                logger.info(f"[*] Downloading {filename}")
                urllib.request.urlretrieve(url, gz_path)

            logger.debug(f"Extracting and cleaning {filename}...")
            with gzip.open(gz_path, 'rt') as f_in, open(txt_path, 'w') as f_out:
                for line in f_in:
                    f_out.write(line)
            os.remove(gz_path)

        except Exception as e:
            logger.error(f"[!] Preparing dataset failed for {filename}: {e}")
            if os.path.exists(txt_path):
                os.remove(txt_path)
            return None

    return txt_path


def setup_logging(log_file_path):
    logger = logging.getLogger("Benchmark")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))

    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger.addHandler(ch)
    logger.addHandler(fh)

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            logger.warning("[!] Execution interrupted by user (KeyboardInterrupt).")
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    return logger


def format_dataframe_with_baseline(df, strategies, baseline_algo=None):
    """Helper function to calculate inline relative performance factors and variance."""
    display_df = df.copy()

    for strat in strategies:
        time_col, ratio_col = f"Time_{strat}", f"Ratio_{strat}"
        t_std_col, r_std_col = f"Time_std_{strat}", f"Ratio_std_{strat}"

        formatted_times, formatted_ratios = [], []

        for _, row in df.iterrows():
            t_val, r_val = row.get(time_col), row.get(ratio_col)
            t_std = row.get(t_std_col, 0.0)
            r_std = row.get(r_std_col, 0.0)

            if pd.notna(t_val):
                t_str = f"{t_val:.3f}s ± {t_std:.3f}s" if t_std > 0 else f"{t_val:.3f}s"
            else:
                t_str = "N/A"

            if pd.notna(r_val):
                r_str = f"{r_val:.5f} ± {r_std:.5f}" if r_std > 0 else f"{r_val:.5f}"
            else:
                r_str = "N/A"

            if baseline_algo and baseline_algo in strategies and strat != baseline_algo:
                t_base = row.get(f"Time_{baseline_algo}")
                r_base = row.get(f"Ratio_{baseline_algo}")

                if pd.notna(t_val) and pd.notna(t_base) and t_val > 0:
                    speedup = t_base / t_val
                    t_str += f" ({speedup:.2f}x)"

                if pd.notna(r_val) and pd.notna(r_base) and r_base > 0:
                    ratio_mult = r_val / r_base
                    r_str += f" ({ratio_mult:.2f}x)"

            formatted_times.append(t_str)
            formatted_ratios.append(r_str)

        display_df[time_col] = formatted_times
        display_df[ratio_col] = formatted_ratios

    std_cols_to_drop = [c for c in display_df.columns if "_std_" in c]
    display_df = display_df.drop(columns=std_cols_to_drop)

    return display_df


def get_datasets_to_run(args):
    """Deciding which datasets to process."""
    datasets_to_run = []
    if args.group == "all":
        for group in DATASETS.values():
            datasets_to_run.extend(group)
    else:
        datasets_to_run.extend(DATASETS.get(args.group, []))

    return datasets_to_run
