import os
import traceback
from tabulate import tabulate

from scripts.config import DATASETS, OUTPUT_DIR
from scripts.utils import setup_logging, download_dataset
from scripts.runners import get_runner

def analyze_graph(file_path):
    """Performs a single pass over the CLEANED file to extract core metrics."""
    nodes = set()
    edge_count = 0
    file_size_bytes = os.path.getsize(file_path)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith(('#', '%', ' ')):
                continue

            parts = line.split()
            if len(parts) >= 2:
                u, v = parts[0], parts[1]
                nodes.add(u)
                nodes.add(v)
                edge_count += 1

    num_nodes = len(nodes)

    # Undirected Average Degree = (2 * Edges) / Nodes
    avg_degree = (2 * edge_count) / num_nodes if num_nodes > 0 else 0

    # Undirected Density = (2 * Edges) / (Nodes * (Nodes - 1))
    density = (2 * edge_count) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0

    return {
        "nodes": num_nodes,
        "edges": edge_count,
        "avg_deg": round(avg_degree, 2),
        "density": density,
        "size": f"{file_size_bytes / (1024*1024):.2f} MB"
    }

def run_metadata_sync():
    logger = setup_logging(os.path.join(OUTPUT_DIR, "metadata_sync.log"))
    logger.info("="*10 + " CLEANED DATASET SYNC " + "="*10)

    runner = get_runner("kdd20-mosso", logger, OUTPUT_DIR)
    runner.build()

    all_results = []

    for group_name, dataset_list in DATASETS.items():
        for ds in dataset_list:
            filename = ds["filename"]
            url = ds["url"]

            try:
                raw_path = download_dataset(url, filename, logger)
                if not raw_path:
                    continue

                logger.info(f"    -> Cleaning {filename}...")
                cleaned_path = runner.format_dataset(raw_path)

                logger.info(f"    -> Analyzing cleaned graph...")
                stats = analyze_graph(cleaned_path)

                all_results.append([
                    filename,
                    stats['size'],
                    f"{stats['nodes']:,}",
                    f"{stats['edges']:,}",
                    stats['avg_deg'],
                    f"{stats['density']:.6e}"  # Formatted to scientific notation
                ])

                print(f"\n# --- Cleaned Metadata for {filename} ---")
                print(f"\"meta\": {{")
                print(f"    \"nodes\": {stats['nodes']},")
                print(f"    \"edges\": {stats['edges']},")
                print(f"    \"size\": \"{stats['size']}\",")
                print(f"    \"avg_degree\": {stats['avg_deg']},")
                print(f"    \"density\": {stats['density']:.6e}")
                print(f"}},")

            except Exception as e:
                logger.error(f"    [!] Error processing {filename}: {e}")
                logger.debug(traceback.format_exc())

    print("\n\n" + "="*20 + " CLEANED GLOBAL SUMMARY " + "="*20)
    print(tabulate(all_results, headers=["Dataset", "Clean Size", "Clean Nodes", "Clean Edges", "Avg Deg", "Density"], tablefmt="grid"))

if __name__ == "__main__":
    run_metadata_sync()