import os
import shutil
import subprocess
import urllib.request
import gzip
import glob
import argparse
import pandas as pd
from tabulate import tabulate

from config import *
import logging
import sys
from datetime import datetime
from config import LOG_DIR


def get_fastutil_path():
    fastutil_files = glob.glob("fastutil-*.jar")
    return fastutil_files[0] if fastutil_files else "fastutil-missing.jar"


def setup_directories():
    for d in [DATASETS_DIR, OUTPUT_DIR, BENCHMARK_DIR, RUNS_DIR, SUMMARIZED_DIR, SWEEP_DIR, LOG_DIR, VERSIONS_DIR]:
        os.makedirs(d, exist_ok=True)


def build_jars(skip_build, is_local, logger):
    if skip_build:
        return

    fastutil = get_fastutil_path()
    if not os.path.exists(fastutil):
        logger.error(f"[!] Error: {fastutil} missing. Download it to root first.")
        exit(1)

    if is_local:
        logger.info("[*] Compiling current Local code...")
        try:
            subprocess.run(["bash", "compile.sh"], cwd=".", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            shutil.move("mosso-1.0.jar", "mosso-Local.jar")
            logger.info("\t[OK] Successfully built mosso-Local.jar")
        except subprocess.CalledProcessError as e:
            logger.error(f"\t[!] Failed to build Local code. Compile Error: {e.stderr.strip()}")
            return
        except Exception as e:
            logger.error(f"\t[!] Unexpected error building Local code: {e}")
            return

    logger.info("[*] Compiling all configured algorithms...")

    for algo_name, config in ALGORITHMS.items():
        if algo_name == "local":
            continue

        repo_url = config['repo']
        branch = config['branch']
        jar_name = f"mosso-{algo_name}.jar"
        target_dir = os.path.join(VERSIONS_DIR, algo_name)

        logger.debug(f"   -> Building {algo_name} (Repo: {repo_url.split('/')[-1]} | Branch: {branch})...")

        try:
            if not os.path.exists(target_dir):
                subprocess.run(
                    ["git", "clone", "-q", "--branch", branch, "--single-branch", repo_url, target_dir],
                    check=True, stderr=subprocess.PIPE, text=True
                )
            else:
                subprocess.run(["git", "pull", "-q"], cwd=target_dir, check=True, stderr=subprocess.PIPE)

            shutil.copy(fastutil, os.path.join(target_dir, fastutil))
            subprocess.run(["bash", "compile.sh"], cwd=target_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            shutil.move(os.path.join(target_dir, "mosso-1.0.jar"), jar_name)
            logger.info(f"      [OK] Successfully built {jar_name}")

        except subprocess.CalledProcessError as e:
            logger.error(f"      [!] Failed to build {algo_name}. Git/Compile Error: {e.stderr.strip()}")
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir) # Clean up broken clones
            return
        except Exception as e:
            logger.error(f"      [!] Unexpected error building {algo_name}: {e}")
            return

    logger.info("[*] All requested Java compilations finished")


def prepare_dataset(filepath, logger):
    filename = os.path.basename(filepath)
    prepared_path = os.path.join(DATASETS_DIR, f"prepared_{filename}")
    if os.path.exists(prepared_path):
        return prepared_path

    logger.debug(f"Cleaning {filename} (Undirected, No Self-Loops, No Multi-Edges)...")
    seen_edges = set()

    try:
        with open(filepath, 'r') as f_in, open(prepared_path, 'w') as f_out:
            for line in f_in:
                if line.startswith('#'): continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        u, v = int(parts[0]), int(parts[1])
                        if u == v: continue
                        edge = tuple(sorted((u, v)))
                        if edge in seen_edges: continue
                        seen_edges.add(edge)
                        f_out.write(f"{u}\t{v}\t1\n")
                    except ValueError:
                        continue
        return prepared_path

    except Exception as e:
        logger.error(f"[!] Failed to prepare local dataset {filename}: {e}")
        if os.path.exists(prepared_path):
            os.remove(prepared_path)
        return None


def download_and_prepare_dataset(url, filename, logger):
    gz_path = os.path.join(DATASETS_DIR, filename + ".gz")
    txt_path = os.path.join(DATASETS_DIR, filename)

    if not os.path.exists(txt_path):
        try:
            if not os.path.exists(gz_path):
                logger.info(f"[*] Downloading {filename}")
                urllib.request.urlretrieve(url, gz_path)

            logger.debug(f"Extracting and cleaning {filename}...")
            seen_edges = set()
            with gzip.open(gz_path, 'rt') as f_in, open(txt_path, 'w') as f_out:
                for line in f_in:
                    if line.startswith('#'): continue
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            u, v = int(parts[0]), int(parts[1])
                            if u == v: continue
                            edge = tuple(sorted((u, v)))
                            if edge in seen_edges: continue
                            seen_edges.add(edge)
                            f_out.write(f"{u}\t{v}\t1\n")
                        except ValueError:
                            continue

            os.remove(gz_path)

        except Exception as e:
            logger.error(f"[!] Preparing dataset failed for {filename}: {e}")
            if os.path.exists(txt_path):
                os.remove(txt_path)
            return None

    return txt_path


def setup_logging(run_type):
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # The unique ID
    log_file = os.path.join(LOG_DIR, f"{run_type}_{timestamp}.log")

    logger = logging.getLogger("MoSSo")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    # Console Handler: Clean, concise output
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))

    # File Handler: Verbose output for debugging
    fh = logging.FileHandler(log_file)
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

    return logger, timestamp

def parse_and_filter_args(script_type="benchmark"):
    """CLI parsing and ALGORITHM dictionary filtering."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Specific local graph file.")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--escape", type=int, default=3)
    parser.add_argument("--b", type=int, default=5)
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--group", choices=["all"] + list(DATASETS.keys()), default="all")
    parser.add_argument("--algos", nargs='+', help="Specific algorithms to run (e.g. local baseline strat_1)")
    parser.add_argument("--keep-summaries", action="store_true")
    parser.add_argument("--baseline", type=str, help="Algorithm to use as baseline for relative comparisons")

    if script_type == "sweep":
        parser.add_argument("--param", choices=list(SWEEP_CONFIG.keys()), required=True)
        parser.add_argument("--range", type=int, nargs=3)
        parser.add_argument("--values", type=int, nargs='+')

    args = parser.parse_args()
    args.local = False

    # Filter ALGORITHMS
    if args.algos:
        if "local" in args.algos:
            args.local = True
        for a in args.algos:
            if a not in ALGORITHMS.keys():
                print(f"[!] Unknown algorithm: {a}. Available options: {list(ALGORITHMS.keys())}")
                exit(1)
        for key in list(ALGORITHMS.keys()):
            if key not in args.algos:
                ALGORITHMS.pop(key, None)
    else:
        ALGORITHMS.pop("local", None)

    if args.baseline and args.baseline not in ALGORITHMS:
        print(f"[!] The specified baseline '{args.baseline}' is not in the active algorithms list.")
        exit(1)

    return args

def format_dataframe_with_baseline(df, strategies, baseline_algo=None):
    """Helper function to calculate inline relative % differences."""
    display_df = df.copy()

    for strat in strategies:
        time_col, ratio_col = f"Time_{strat}", f"Ratio_{strat}"
        formatted_times, formatted_ratios = [], []

        for _, row in df.iterrows():
            t_val, r_val = row.get(time_col), row.get(ratio_col)

            # If a baseline is provided and this column IS NOT the baseline
            if baseline_algo and baseline_algo in strategies and strat != baseline_algo:
                t_base = row.get(f"Time_{baseline_algo}")
                r_base = row.get(f"Ratio_{baseline_algo}")

                # Format Time (Lower is better: -X% is faster)
                if pd.notna(t_val) and pd.notna(t_base) and t_base > 0:
                    pct_change = ((t_val - t_base) / t_base) * 100
                    formatted_times.append(f"{t_val:.3f}s ({pct_change:+.1f}%)")
                else:
                    formatted_times.append(f"{t_val:.3f}s" if pd.notna(t_val) else "N/A")

                # Format Ratio (Lower is better: -X% is more compression)
                if pd.notna(r_val) and pd.notna(r_base) and r_base > 0:
                    pct_change = ((r_val - r_base) / r_base) * 100
                    formatted_ratios.append(f"{r_val:.5f} ({pct_change:+.2f}%)")
                else:
                    formatted_ratios.append(f"{r_val:.5f}" if pd.notna(r_val) else "N/A")

            # Standard formatting if no baseline comparison applies
            else:
                formatted_times.append(f"{t_val:.3f}s" if pd.notna(t_val) else "N/A")
                formatted_ratios.append(f"{r_val:.5f}" if pd.notna(r_val) else "N/A")

        display_df[time_col] = formatted_times
        display_df[ratio_col] = formatted_ratios

    return display_df

def get_datasets_to_run(args):
    """Deciding which datasets to process."""
    datasets_to_run = [("local", args.file)] if args.file else []
    if not args.file:
        if args.group == "all":
            for cat, data_list in DATASETS.items():
                for url, filename in data_list:
                    datasets_to_run.append((url, filename))
        else:
            for url, filename in DATASETS[args.group]:
                datasets_to_run.append((url, filename))
    return datasets_to_run


def print_sweep_table(results, logger, title, sweep_param=None, baseline_algo=None):
    df = pd.DataFrame(results)
    strategies = [col.replace("Time_", "") for col in df.columns if col.startswith("Time_")]

    display_df = format_dataframe_with_baseline(df, strategies, baseline_algo)

    # Prettify Headers
    new_cols = {c: f"{c.replace('Time_', '').capitalize()} Time" if c.startswith("Time_") else f"{c.replace('Ratio_', '').capitalize()} Ratio" if c.startswith("Ratio_") else c.capitalize() for c in display_df.columns}
    display_df.rename(columns=new_cols, inplace=True)

    table_str = tabulate(display_df, headers='keys', tablefmt='grid', showindex=False)
    line_width = len(table_str.split('\n')[0])

    logger.info("=" * line_width)
    logger.info(f"{title:^{line_width}}")
    logger.info("=" * line_width)

    for line in table_str.split('\n'):
        logger.info(line)

    # Average table
    logger.info("=" * line_width)
    logger.info(f"{'AVERAGES BY PARAMETER VALUE':^{line_width}}")
    logger.info("=" * line_width)

    avg_df = df.groupby(sweep_param).mean(numeric_only=True).reset_index()
    avg_df.insert(0, 'Dataset', 'ALL (Avg)')

    display_avg = format_dataframe_with_baseline(avg_df, strategies, baseline_algo)

    # Prettify Headers
    new_cols[sweep_param] = sweep_param.capitalize()
    display_avg.rename(columns=new_cols, inplace=True)

    avg_table_str = tabulate(display_avg, headers='keys', tablefmt='grid', showindex=False)

    for line in avg_table_str.split('\n'):
        logger.info(line)

def print_benchmark_table(results, logger, title, baseline_algo=None):
    df = pd.DataFrame(results)
    strategies = [col.replace("Time_", "") for col in df.columns if col.startswith("Time_")]

    if "Dataset" in df.columns and len(df.columns) > 2 and "Dataset" == df.columns[0]:
        avg_row = df.mean(numeric_only=True).to_dict()
        avg_row['Dataset'] = 'AVERAGE'
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    display_df = format_dataframe_with_baseline(df, strategies, baseline_algo)

    # Prettify Headers
    new_cols = {c: f"{c.replace('Time_', '').capitalize()} Time" if c.startswith("Time_") else f"{c.replace('Ratio_', '').capitalize()} Ratio" if c.startswith("Ratio_") else c.capitalize() for c in display_df.columns}
    display_df.rename(columns=new_cols, inplace=True)

    table_str = tabulate(display_df, headers='keys', tablefmt='grid', showindex=False)
    line_width = len(table_str.split('\n')[0])

    logger.info("=" * line_width )
    logger.info(f"{title:^{line_width}}")
    logger.info("=" * line_width)

    for line in table_str.split('\n'):
        logger.info(line)
