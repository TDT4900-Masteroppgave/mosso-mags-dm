import gzip
import json
import shutil
import random
import zipfile
import tarfile
import subprocess
import urllib.error
import urllib.request
import networkx as nx
from pathlib import Path
from collections import defaultdict, Counter

from scripts.config import DATASETS_DIR, DATASETS


def clean_and_write(src: str, dst: str, algo_type: str) -> None:
    is_text_format = src.endswith((".txt", ".edges"))
    is_mtx = src.endswith(".mtx")

    seen = set()
    valid_edges = []
    header_skipped = False

    with open(src, "r", encoding="utf-8") as fin:
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
                if parts[3] in ("0", "0.0", "\\N", "-1"):
                    continue

            try:
                u, v = int(parts[0]), int(parts[1])
                if u != v:
                    edge = (min(u, v), max(u, v))
                    if edge not in seen:
                        seen.add(edge)
                        valid_edges.append(edge)
            except ValueError:
                continue

    random.seed(42)
    random.shuffle(valid_edges)

    with open(dst, "w", encoding="utf-8") as fout:
        for u, v in valid_edges:
            if algo_type == "mosso":
                fout.write(f"{u}\t{v}\t1\n")
            else:
                fout.write(f"{u}\t{v}\n")


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


def download_dataset(url: str, filename: str, timeout: int = 60) -> str:
    archive_path = Path(DATASETS_DIR) / Path(url).name
    txt_path = Path(DATASETS_DIR) / filename

    if txt_path.exists():
        return str(txt_path)

    try:
        if not archive_path.exists():
            with urllib.request.urlopen(url, timeout=timeout) as response, open(archive_path, "wb") as out:
                out.write(response.read())

        if url.endswith(".zip"):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                target_ext = Path(filename).suffix
                member = next(m for m in zip_ref.namelist() if m.endswith(target_ext))

                with zip_ref.open(member) as f_in, open(txt_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

        elif url.endswith(".tar.bz2"):
            with tarfile.open(archive_path, "r:bz2") as tar:
                member_info = next(m for m in tar.getmembers() if "out." in m.name)
                f_in = tar.extractfile(member_info)
                if f_in is not None:
                    with open(txt_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

        elif url.endswith(".tar.gz"):
            with tarfile.open(archive_path, "r:gz") as tar:
                member_info = next(m for m in tar.getmembers() if m.name.endswith(".mtx"))
                f_in = tar.extractfile(member_info)
                if f_in is not None:
                    with open(txt_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

        else:
            with gzip.open(archive_path, "rt") as f_in, open(txt_path, "w") as f_out:
                shutil.copyfileobj(f_in, f_out)

        archive_path.unlink()
        return str(txt_path)
    except Exception as e:
        if txt_path.exists(): txt_path.unlink()
        raise e


def generate_master_stream(src: str, dst: str, p_delete: float = 0.1) -> int:
    events = []
    removed_count = 0
    stream_rng = random.Random(42)

    with open(src, "r", encoding="utf-8") as fin:
        for line in fin:
            parts = line.split()
            if len(parts) >= 2:
                u, v = parts[0], parts[1]

                t_insert = stream_rng.random()
                events.append((t_insert, u, v, "1"))

                if stream_rng.random() < p_delete:
                    t_delete = stream_rng.uniform(t_insert, 1.0)
                    events.append((t_delete, u, v, "-1"))

    events.sort(key=lambda x: x[0])
    adj = defaultdict(set)

    with open(dst, "w", encoding="utf-8") as fout:
        for _, u, v, indicator in events:
            if indicator == "1":
                adj[u].add(v)
                adj[v].add(u)
                fout.write(f"{u}\t{v}\t1\n")
            elif indicator == "-1":
                if adj[u].intersection(adj[v]):
                    adj[u].remove(v)
                    adj[v].remove(u)
                    fout.write(f"{u}\t{v}\t-1\n")
                    removed_count += 1

    return removed_count


def extract_batch_snapshot(stream_src: str, batch_dst: str) -> None:
    active_edges = {}

    with open(stream_src, "r", encoding="utf-8") as fin:
        for line in fin:
            parts = line.split()
            if len(parts) >= 3:
                u, v, indicator = parts[0], parts[1], parts[2]
                if indicator == "1":
                    active_edges[(u, v)] = True
                elif indicator == "-1":
                    active_edges.pop((u, v), None)

    with open(batch_dst, "w", encoding="utf-8") as fout:
        for u, v in active_edges.keys():
            fout.write(f"{u}\t{v}\n")


def calculate_topology(file_path: str) -> tuple[int, int, float, int, float, dict]:
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
    density = (2.0 * edge_count) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0

    degree_frequencies = Counter(degrees.values())
    degree_counts_json = {str(deg): count for deg, count in degree_frequencies.items()}

    return num_nodes, edge_count, round(avg_deg, 2), max_deg, density, degree_counts_json


def calculate_advanced_topology(file_path: str, logger) -> tuple[float, float, int]:
    """Calculates NetworkX topological features necessary for tradeoff analysis."""
    logger.print(f"    [dim]└ Loading graph into NetworkX...[/dim]")
    G = nx.read_edgelist(file_path, nodetype=int)

    logger.print(f"    [dim]└ Calculating degree assortativity...[/dim]")
    try:
        assortativity = nx.degree_assortativity_coefficient(G)
        if str(assortativity) == "nan": assortativity = 0.0
    except Exception:
        assortativity = 0.0

    logger.print(f"    [dim]└ Calculating approximate clustering coefficient...[/dim]")
    try:
        clustering = nx.approximation.average_clustering(G, trials=50000, seed=42)
    except Exception:
        clustering = 0.0

    logger.print(f"    [dim]└ Calculating maximum k-core...[/dim]")
    try:
        G.remove_edges_from(nx.selfloop_edges(G))
        core_numbers = nx.core_number(G)
        max_k_core = max(core_numbers.values()) if core_numbers else 0
    except Exception:
        max_k_core = 0

    return round(assortativity, 4), round(clustering, 4), max_k_core


def _write_algorithm_format(algo_type: str, stream_type: str, temp_clean: Path, master_stream: Path, out_path: Path):
    if stream_type == "IO":
        if algo_type == "mosso":
            with open(temp_clean, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
                for line in fin:
                    fout.write(f"{line.strip()}\t1\n")
        else:
            shutil.copy(temp_clean, out_path)
    elif stream_type == "FD":
        if algo_type == "mosso":
            shutil.copy(master_stream, out_path)
        elif algo_type == "mags":
            extract_batch_snapshot(str(master_stream), str(out_path))


def _process_dataset_pipeline(ds: dict, required_algos: set, logger, prep_log: dict, dynamic_datasets: list) -> dict:
    short_name = ds.get("short_name", "N/A")
    raw_path = Path(download_dataset(ds.get("url"), ds.get("filename")))

    is_dynamic = short_name in dynamic_datasets
    stream_type = "FD" if is_dynamic else "IO"
    p_del_val = 0.1 if stream_type == "FD" else 0.0

    temp_clean = raw_path.parent / f"{raw_path.stem}_temp_clean.txt"
    master_stream = raw_path.parent / f"{raw_path.stem}_master_stream.txt"

    formatted_paths = {}
    needs_building = False
    for algo in required_algos:
        save_dir = raw_path.parent / algo.capitalize()
        save_dir.mkdir(exist_ok=True)
        fmt_path = save_dir / f"{algo}_{raw_path.stem}_{stream_type}.txt"
        formatted_paths[algo] = fmt_path
        if not fmt_path.exists():
            needs_building = True

    if short_name not in prep_log:
        prep_log[short_name] = {"metadata": {}, "algorithms": {}}

    if needs_building:
        needs_temp_clean = not prep_log[short_name].get("metadata") or (stream_type == "IO") or (
                stream_type == "FD" and not master_stream.exists())

        if needs_temp_clean and not temp_clean.exists():
            clean_and_write(str(raw_path), str(temp_clean), "mags")

        if not prep_log[short_name].get("metadata"):
            logger.status(f"[bold cyan]Calculating basic topology for {short_name}[/bold cyan]")
            num_nodes, total_edges, avg_deg, max_deg, density, degree_counts = calculate_topology(str(temp_clean))

            logger.status(f"[bold cyan]Calculating advanced topology for {short_name}[/bold cyan]")
            assort, cluster, kcore = calculate_advanced_topology(str(temp_clean), logger)

            prep_log[short_name]["metadata"] = {
                "stream_type": stream_type, "p_delete": p_del_val,
                "nodes": num_nodes, "edges": total_edges, "deleted_edges": 0,
                "avg_degree": avg_deg, "max_degree": max_deg, "density": density,
                "assortativity": assort, "clustering": cluster, "max_k_core": kcore,
                "degree_counts": degree_counts
            }

        if stream_type == "FD" and not master_stream.exists():
            logger.status(f"[bold cyan]Generating Ground Truth Timeline for {short_name}[/bold cyan]")
            deleted_edges = generate_master_stream(str(temp_clean), str(master_stream), p_delete=p_del_val)
            prep_log[short_name]["metadata"]["deleted_edges"] = deleted_edges

        for algo, fmt_path in formatted_paths.items():
            if not fmt_path.exists():
                logger.status(f"[bold cyan]Preprocessing {short_name} for {algo} ({stream_type})[/bold cyan]")
                _write_algorithm_format(algo, stream_type, temp_clean, master_stream, fmt_path)
                prep_log[short_name]["algorithms"][algo] = {"file_path": str(fmt_path)}

    meta = prep_log[short_name]["metadata"]
    for target in [ds.setdefault("meta", {}), DATASETS.setdefault(short_name, {}).setdefault("meta", {})]:
        target.update({
            "nodes": meta.get("nodes", 0),
            "edges": meta.get("edges", 0),
        })

    if temp_clean.exists():
        temp_clean.unlink()

    return {algo: str(path) for algo, path in formatted_paths.items()}


def prepare_datasets(datasets_to_run, active_algos, logger, dynamic_datasets: list = None) -> dict:
    if dynamic_datasets is None: dynamic_datasets = []
    required_algo_types = {config.get("type") for config in active_algos.values() if config.get("type")}
    log_file = DATASETS_DIR / "preprocessing_log.json"

    prep_log = {}
    if log_file.exists():
        with open(log_file, "r", encoding="utf-8") as f:
            prep_log = json.load(f)

    prepared_paths = {}
    for ds in datasets_to_run:
        with logger.status(f"[bold cyan]Preprocessing Dataset: {ds.get('filename', 'N/A')} [/bold cyan]"):
            prepared_paths[ds.get("short_name")] = _process_dataset_pipeline(
                ds, required_algo_types, logger, prep_log, dynamic_datasets
            )
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(prep_log, f, indent=4)

    return prepared_paths