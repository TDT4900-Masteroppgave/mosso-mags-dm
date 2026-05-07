import gzip
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

from scripts.config import DATASETS_DIR, DATASET_GROUP, DATASETS


def clean_and_write(src: str, dst: str, algo_type: str) -> None:
    is_text_format = src.endswith((".txt", ".edges"))
    is_mtx = src.endswith(".mtx")

    if not (is_text_format or is_mtx):
        raise ValueError(f"Unsupported file format: {src}")

    seen = set()
    header_skipped = False

    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            if line.startswith(("#", "%")):
                continue

            if is_mtx and not header_skipped:
                header_skipped = True
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            if is_text_format and len(parts) >= 4:
                # The timestamp is usually the 4th column (index 3)
                if parts[3] in ("0", "0.0", "\\N", "-1"):
                    continue

            try:
                u, v = int(parts[0]), int(parts[1])
                if u != v:
                    edge = (min(u, v), max(u, v))
                    if edge not in seen:
                        seen.add(edge)
                        if algo_type == "mosso":
                            fout.write(f"{u}\t{v}\t1\n")
                        else:
                            fout.write(f"{u}\t{v}\n")
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

import tarfile

def download_dataset(url: str, filename: str, timeout: int = 60) -> str:
    archive_path = Path(DATASETS_DIR) / Path(url).name
    txt_path = Path(DATASETS_DIR) / filename

    if txt_path.exists():
        return str(txt_path)

    try:
        if not archive_path.exists():
            with urllib.request.urlopen(url, timeout=timeout) as response, open(archive_path, "wb") as out:
                out.write(response.read())

        # 1. Handle Enron/Network Repository .zip format
        if url.endswith(".zip"):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                # Find the .edges file (it's often inside a subdirectory)
                member = next(m for m in zip_ref.namelist() if m.endswith(".edges"))
                with zip_ref.open(member) as f_in, open(txt_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

        # 2. Handle Reactome .tar.bz2 format
        elif url.endswith(".tar.bz2"):
            with tarfile.open(archive_path, "r:bz2") as tar:
                member = next(m for m in tar.getmembers() if "out." in m.name)
                with tar.extractfile(member) as f_in, open(txt_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

        # 3. Handle EU/SuiteSparse .tar.gz format
        elif url.endswith(".tar.gz"):
            with tarfile.open(archive_path, "r:gz") as tar:
                member = next(m for m in tar.getmembers() if m.name.endswith(".mtx"))
                with tar.extractfile(member) as f_in, open(txt_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

        else:
            # 4. Standard .gz logic for SNAP datasets
            with gzip.open(archive_path, "rt") as f_in, open(txt_path, "w") as f_out:
                shutil.copyfileobj(f_in, f_out)

        archive_path.unlink()  # Clean up the .zip/.gz archive
        return str(txt_path)
    except Exception as e:
        if txt_path.exists(): txt_path.unlink()
        raise e


import random


def generate_dynamic_stream_graph(src: str, dst: str, p_delete: float = 0.1) -> int:
    """
    Generates a Fully Dynamic (FD) stream from a clean edge list.
    Edges are inserted in random order. With probability p_delete,
    an edge is deleted at a random time strictly after its insertion.

    Returns:
        int: The total number of edges that were scheduled for deletion.
    """
    events = []
    removed_count = 0

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
                    removed_count += 1

    events.sort(key=lambda x: x[0])

    with open(dst, "w", encoding="utf-8") as fout:
        for _, u, v, indicator in events:
            fout.write(f"{u}\t{v}\t{indicator}\n")

    return removed_count


def generate_dynamic_batch_graph(src: str, dst: str, p_delete: float = 0.1) -> int:
    """
    Generates the final static state of a fully dynamic stream.
    Since 10% of edges are deleted during the stream, the final batch
    graph is simply the remaining 90% of the original edges.

    Returns:
        int: The total number of edges that were dropped (deleted).
    """
    removed_count = 0

    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            # 90% chance to survive to the end of the stream
            if random.random() >= p_delete:
                fout.write(line)
            else:
                removed_count += 1

    return removed_count


from collections import defaultdict


def calculate_topology(file_path: str) -> tuple[int, int, float, int, float]:
    """
    Reads a preprocessed file to calculate graph topology.
    """
    nodes = set()
    edge_count = 0
    degrees = defaultdict(int)

    with open(file_path, "r", encoding="utf-8") as fin:
        for line in fin:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    u, v = int(parts[0]), int(parts[1])
                    nodes.add(u)
                    nodes.add(v)
                    edge_count += 1
                    degrees[u] += 1
                    degrees[v] += 1
                except ValueError:
                    continue

    num_nodes = len(nodes)
    avg_deg = (2.0 * edge_count / num_nodes) if num_nodes > 0 else 0.0
    max_deg = max(degrees.values()) if degrees else 0

    density = 0.0
    if num_nodes > 1:
        density = (2.0 * edge_count) / (num_nodes * (num_nodes - 1))

    return num_nodes, edge_count, round(avg_deg, 2), max_deg, density


def prepare_datasets(datasets_to_run, active_algos, logger) -> dict:
    prepared_paths = {}
    required_algo_types = {config.get("type") for config in active_algos.values() if config.get("type")}

    log_file = DATASETS_DIR / "preprocessing_log.json"
    prep_log = {}
    if log_file.exists():
        with open(log_file, "r", encoding="utf-8") as f:
            prep_log = json.load(f)

    for ds in datasets_to_run:
        with logger.status(f"[bold cyan]Preprocessing Dataset: {ds.get('filename', 'N/A')} [/bold cyan]"):
            short_name = ds.get("short_name", "N/A")
            prepared_paths[short_name] = {}

            if short_name not in prep_log:
                prep_log[short_name] = {}

            raw_path = Path(download_dataset(ds.get("url"), ds.get("filename")))
            is_dynamic = "dynamic" in sys.argv and short_name in DATASET_GROUP.get("dynamic", [])
            stream_type = "FD" if is_dynamic else "IO"

            temp_clean = raw_path.parent / f"{raw_path.stem}_temp_clean.txt"
            topo_data = None

            for algo_type in required_algo_types:
                save_dir = raw_path.parent / algo_type.capitalize()
                save_dir.mkdir(exist_ok=True)
                formatted_path = save_dir / f"{algo_type}_{raw_path.stem}_{stream_type}.txt"

                deleted = 0
                p_del_val = 0.1 if stream_type == "FD" else 0.0

                if not formatted_path.exists():
                    logger.status(f"[bold cyan]Preprocessing {short_name} for {algo_type} ({stream_type})[/bold cyan]")

                    if not temp_clean.exists():
                        clean_and_write(str(raw_path), str(temp_clean), "{u}\t{v}\n")

                    if topo_data is None:
                        logger.status(f"[bold cyan]Calculating topology for {short_name}[/bold cyan]")
                        topo_data = calculate_topology(str(temp_clean))

                    num_nodes, total_edges, avg_deg, max_deg, density = topo_data

                    if stream_type == "IO":
                        if algo_type == "mosso":
                            # Fast line-by-line format mapping
                            with open(temp_clean, "r", encoding="utf-8") as fin, open(formatted_path, "w",
                                                                                      encoding="utf-8") as fout:
                                for line in fin:
                                    fout.write(f"{line.strip()}\t1\n")
                        else:
                            shutil.copy(temp_clean, formatted_path)

                    elif stream_type == "FD":
                        if algo_type == "mosso":
                            deleted = generate_dynamic_stream_graph(str(temp_clean), str(formatted_path),
                                                                    p_delete=p_del_val)
                        elif algo_type == "mags":
                            deleted = generate_dynamic_batch_graph(str(temp_clean), str(formatted_path),
                                                                   p_delete=p_del_val)

                    # --- NEW LOGGING STRUCTURE ---
                    if "dataset_metadata" not in prep_log[short_name]:
                        prep_log[short_name]["dataset_metadata"] = {
                            "stream_type": stream_type,
                            "p_delete": p_del_val,
                            "nodes": num_nodes,
                            "edges": total_edges,
                            "avg_degree": avg_deg,
                            "max_degree": max_deg,
                            "density": density
                        }

                    if "algorithms" not in prep_log[short_name]:
                        prep_log[short_name]["algorithms"] = {}

                    prep_log[short_name]["algorithms"][algo_type] = {
                        "timestamp": datetime.now().isoformat(),
                        "deleted_edges": deleted,
                        "file_path": str(formatted_path)
                    }

                    with open(log_file, "w", encoding="utf-8") as f:
                        json.dump(prep_log, f, indent=4)

                else:
                    meta_entry = prep_log.get(short_name, {}).get("dataset_metadata", {})
                    algo_entry = prep_log.get(short_name, {}).get("algorithms", {}).get(algo_type, {})
                    num_nodes = meta_entry.get("nodes", 0)
                    total_edges = meta_entry.get("edges", 0)
                    deleted = algo_entry.get("deleted_edges", 0)

                # Overwrite Config in Memory
                if "meta" not in ds: ds["meta"] = {}
                # ... (rest of the code remains the same)
                if "meta" not in DATASETS.get(short_name, {}): DATASETS[short_name]["meta"] = {}

                for meta_target in [ds["meta"], DATASETS[short_name]["meta"]]:
                    meta_target["nodes"] = num_nodes
                    meta_target["edges"] = total_edges
                    meta_target["deleted_edges"] = deleted

                prepared_paths[short_name][algo_type] = str(formatted_path)

            if temp_clean.exists():
                temp_clean.unlink()

    return prepared_paths
