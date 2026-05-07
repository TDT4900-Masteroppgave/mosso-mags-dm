from pathlib import Path
from rich.console import Console
from rich.table import Table

from scripts.config import DATASETS, DATASETS_DIR, OUTPUT_DIR
from scripts.datasets import download_dataset


def count(file_path: str) -> tuple[int, int]:
    """Counts nodes and edges using the clean_and_write logic:
       - Ignores edge direction (by sorting u, v)
       - Removes self-loops (u != v)
       - Removes duplicate edges (using a seen set)
    """
    nodes = set()
    seen_edges = set()
    with open(file_path, "r", encoding="utf-8") as fin:
        for line in fin:
            if line.startswith(("#", "%")):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                u, v = int(parts[0]), int(parts[1])
                if u != v:  # Remove self-loops
                    # Ignore direction by always storing as (min, max)
                    edge = (min(u, v), max(u, v))

                    if edge not in seen_edges:  # Remove multiple edges
                        seen_edges.add(edge)
                        nodes.add(u)
                        nodes.add(v)
            except ValueError:
                continue

    return len(nodes), len(seen_edges)


def main():
    Path(DATASETS_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    console = Console(highlight=False)

    table = Table(title="Dataset Metrics Report", show_header=True, header_style="bold magenta")
    table.add_column("Dataset", justify="left", style="cyan", no_wrap=True)
    table.add_column("Nodes", justify="right", style="green")
    table.add_column("Edges", justify="right", style="green")

    # Iterate through all datasets in config.py
    for name, info in DATASETS.items():
        url = info["url"]
        filename = info["filename"]

        # Download the dataset using the existing function
        with console.status(f"[bold cyan]{name}[/bold cyan]"):
            txt_path = download_dataset(url, filename)

            if not txt_path:
                console.print(f"Failed to process {name}")
                continue

            nodes, edges = count(txt_path)

            table.add_row(
                name,
                f"{nodes:,}",
                f"{edges:,}"
            )

    console.print(table)
    output_path = OUTPUT_DIR / "dataset_report.txt"
    with open(output_path, "wt", encoding="utf-8") as f:
        file_console = Console(file=f, color_system=None)
        file_console.print(table)
    console.print(f"Report saved to [bold green]{output_path}[/bold green]")


if __name__ == "__main__":
    main()
