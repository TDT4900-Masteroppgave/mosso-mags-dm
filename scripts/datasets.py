"""Dataset utilities: edge parsing, partial snapshots, download and git clone."""
import gzip
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from typing import Optional

from scripts.config import DATASETS_DIR

EDGE_RE = re.compile(r"^\s*(\d+)\s+(\d+)")


def clean_and_write(src: str, dst: str, edge_format: str = "{u}\t{v}\n") -> None:
    """Parse edges from src, write cleaned output to dst.

    Skips comment lines (#, %), removes self-loops and duplicate (undirected) edges.
    edge_format controls the output template; must contain {u} and {v}.
    """
    seen: set[tuple[int, int]] = set()
    with (open(src, "r", encoding="utf-8") as f_in,
          open(dst, "w", encoding="utf-8") as f_out):
        for line in f_in:
            if line.startswith(("#", "%")):
                continue
            m = EDGE_RE.search(line)
            if not m:
                continue
            u, v = int(m.group(1)), int(m.group(2))
            if u == v:
                continue
            edge = tuple(sorted((u, v)))
            if edge in seen:
                continue
            seen.add(edge)
            f_out.write(edge_format.format(u=u, v=v))


def create_partial_dataset(
    dataset_path: str,
    fraction: float,
    total_edges: int,
    logger=None,
) -> str:
    """Write a file containing the first (fraction * total_edges) unique edges.

    Cached: if the partial file already exists it is returned immediately.
    """
    target_edges = int(total_edges * fraction)
    base_dir = os.path.dirname(dataset_path)
    basename = os.path.basename(dataset_path)
    partial_dir = os.path.join(base_dir, "partial")
    os.makedirs(partial_dir, exist_ok=True)
    partial_path = os.path.join(partial_dir, f"p{int(fraction * 100)}_{basename}")

    if os.path.exists(partial_path):
        return partial_path

    edges_written = 0
    seen: set[tuple[int, int]] = set()
    with (open(dataset_path, "r", encoding="utf-8") as f_in,
          open(partial_path, "w", encoding="utf-8") as f_out):
        for line in f_in:
            if line.startswith(("#", "%")):
                continue
            m = EDGE_RE.search(line)
            if not m:
                continue
            u, v = int(m.group(1)), int(m.group(2))
            if u == v:
                continue
            edge = tuple(sorted((u, v)))
            if edge in seen:
                continue
            seen.add(edge)
            f_out.write(line)
            edges_written += 1
            if edges_written >= target_edges:
                break

    if logger:
        logger.debug(
            f"\t[*] Partial dataset ({fraction:.0%}): {edges_written:,} edges -> {partial_path}"
        )
    return partial_path


def retrieve_github_code(target_dir: str, algo_name: str, repo_url: str, branch: str, logger) -> None:
    try:
        if not os.path.exists(target_dir):
            logger.info(f"    -> [{algo_name}] Target directory not found. Cloning fresh...")
            subprocess.run(
                ["git", "clone", "-q", "--branch", branch, "--single-branch", repo_url, target_dir],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            )
        else:
            logger.info(f"    -> [{algo_name}] Target directory exists. Pulling latest updates...")
            subprocess.run(
                ["git", "pull", "-q"], cwd=target_dir,
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            )
    except subprocess.CalledProcessError as e:
        raise e


def download_dataset(url: str, filename: str, logger, retries: int = 3, timeout: int = 60) -> str | None:
    """Download and extract a dataset with retry/timeout on network failures."""
    gz_path = os.path.join(DATASETS_DIR, filename + ".gz")
    txt_path = os.path.join(DATASETS_DIR, filename)

    if os.path.exists(txt_path):
        return txt_path

    try:
        if not os.path.exists(gz_path):
            logger.info(f"[*] Downloading {filename}")
            for attempt in range(1, retries + 1):
                try:
                    with (urllib.request.urlopen(url, timeout=timeout) as response,
                          open(gz_path, "wb") as out_file):
                        out_file.write(response.read())
                    break
                except (urllib.error.URLError, OSError) as e:
                    if attempt < retries:
                        wait = 2 ** attempt
                        logger.warning(
                            f"[!] Download attempt {attempt}/{retries} failed: {e}. Retrying in {wait}s..."
                        )
                        time.sleep(wait)
                    else:
                        raise RuntimeError(f"Download failed after {retries} attempts: {e}") from e

        logger.debug(f"Extracting and cleaning {filename}...")
        with gzip.open(gz_path, "rt") as f_in, open(txt_path, "w") as f_out:
            for line in f_in:
                f_out.write(line)
        os.remove(gz_path)

    except Exception as e:
        logger.error(f"[!] Preparing dataset failed for {filename}: {e}")
        if os.path.exists(txt_path):
            os.remove(txt_path)
        return None

    return txt_path
