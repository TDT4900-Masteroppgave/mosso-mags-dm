import gzip
import random
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from scripts.config import DATASETS_DIR

def clean_and_write(src: str, dst: str, edge_format: str = "{u}\t{v}\n") -> None:
    """Parse edges from src, write cleaned output to dst."""
    seen = set()
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            if line.startswith(("#", "%")):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                u, v = int(parts[0]), int(parts[1])
                if u != v:
                    edge = (min(u, v), max(u, v))
                    if edge not in seen:
                        seen.add(edge)
                        fout.write(edge_format.format(u=u, v=v))
            except ValueError:
                continue


def create_partial_dataset(dataset_path: str, fraction: float, total_edges: int, logger=None) -> str:
    """Write a file containing the first (fraction * total_edges) unique edges."""
    src_path = Path(dataset_path)
    target_edges = int(total_edges * fraction)

    partial_dir = src_path.parent / "partial"
    partial_dir.mkdir(exist_ok=True)

    out_path = partial_dir / f"p{int(fraction * 100)}_{src_path.name}"

    if out_path.exists():
        return str(out_path)

    seen = set()
    with open(src_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if line.startswith(("#", "%")):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                u, v = int(parts[0]), int(parts[1])
                if u != v:
                    edge = (min(u, v), max(u, v))
                    if edge not in seen:
                        seen.add(edge)
                        fout.write(line)
                        if len(seen) >= target_edges:
                            break
            except ValueError:
                continue

    if logger:
        logger.debug(f"\t[*] Partial dataset ({fraction:.0%}): {len(seen):,} edges -> {out_path}")

    return str(out_path)


def retrieve_github_code(target_dir: str, algo_name: str, repo_url: str, branch: str, logger) -> None:
    target = Path(target_dir)
    try:
        if not target.exists():
            logger.debug(f"[{algo_name}] Target directory not found. Cloning fresh...")
            subprocess.run(["git", "clone", "-q", "--branch", branch, "--single-branch", repo_url, str(target)],
                           check=True, capture_output=True, text=True)
        else:
            logger.debug(f"[{algo_name}] Target directory exists. Pulling latest updates...")
            subprocess.run(["git", "pull", "-q"], cwd=str(target),
                           check=True, capture_output=True, text=True)

    except subprocess.CalledProcessError as e:
        raise e


def download_dataset(url: str, filename: str, timeout: int = 60) -> str | None:
    """Download and extract a dataset with retry/timeout on network failures."""
    gz_path = Path(DATASETS_DIR) / f"{filename}.gz"
    txt_path = Path(DATASETS_DIR) / filename

    if txt_path.exists():
        return str(txt_path)

    try:
        if not gz_path.exists():
            with urllib.request.urlopen(url, timeout=timeout) as response, open(gz_path, "wb") as out:
                out.write(response.read())
        with gzip.open(gz_path, "rt") as f_in, open(txt_path, "w") as f_out:
            for line in f_in:
                f_out.write(line)

        gz_path.unlink()
        return str(txt_path)

    except Exception as e:
        if txt_path.exists():
            txt_path.unlink()
        raise e

def generate_dynamic_stream_graph(src: str, dst: str, p_delete: float = 0.1) -> None:
    """
    Generates a Fully Dynamic (FD) stream from a clean edge list.
    Edges are inserted in random order. With probability p_delete,
    an edge is deleted at a random time strictly after its insertion.
    """
    events = []

    with open(src, "r", encoding="utf-8") as fin:
        for line in fin:
            parts = line.split()
            if len(parts) >= 2:
                u, v = parts[0], parts[1]

                t_insert = random.random()
                events.append((t_insert, u, v, "1"))

                if random.random() < p_delete:
                    t_delete = random.uniform(t_insert, 1.0)
                    events.append((t_delete, u, v, "-1"))

    events.sort(key=lambda x: x[0])

    with open(dst, "w", encoding="utf-8") as fout:
        for _, u, v, indicator in events:
            fout.write(f"{u}\t{v}\t{indicator}\n")

def generate_dynamic_batch_graph(src: str, dst: str, p_delete: float = 0.1) -> None:
    """
    Generates the final static state of a fully dynamic stream.
    Since 10% of edges are deleted during the stream, the final batch
    graph is simply the remaining 90% of the original edges.
    """
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            # 90% chance to survive to the end of the stream
            if random.random() >= p_delete:
                fout.write(line)