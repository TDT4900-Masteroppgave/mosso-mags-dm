from pathlib import Path
import pandas as pd
from rich.table import Table
from rich import box
from scripts.config import DATASETS
from scripts.experiments.base_experiment import Experiment

def create_partial_dataset_by_count(dataset_path: str, max_edges: int) -> str:
    path = Path(dataset_path)
    partial_path = path.parent / "partial" / f"n{max_edges}_{path.name}"
    partial_path.parent.mkdir(parents=True, exist_ok=True)

    if partial_path.exists():
        return str(partial_path)

    seen = set()
    tmp = partial_path.with_suffix('.tmp')

    with open(path, "r", encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as f_out:
        for line in fin:
            if line.startswith(("#", "%")):
                continue
            try:
                u, v = map(int, line.split()[:2])
                if u != v and (edge := (min(u, v), max(u, v))) not in seen:
                    seen.add(edge)
                    f_out.write(line)
                    if len(seen) >= max_edges:
                        break
            except ValueError:
                continue

    tmp.rename(partial_path)
    return str(partial_path)


class Scalability(Experiment):
    DEFAULT_POWERS = [20, 21, 22, 23, 24, 25, 26, 27, 28]

    def __init__(self):
        # Define powers BEFORE calling super().__init__ so the parser can use it
        self.powers = self.DEFAULT_POWERS
        super().__init__("scalability")

        # New dictionary to hold the sliced file paths
        self.scalability_datasets = {}

    def add_custom_args(self, parser):
        parser.add_argument("--powers", nargs="+", type=int, default=self.powers,
                            help="List of exponents (base 2) for edge counts")

    def post_preprocessing(self):
        """Overrides the base hook to generate partial datasets by edge count for ALL algorithms."""
        self.logger.print("\n[bold cyan]Generating Scalability Checkpoints[/bold cyan]")

        for ds in self.datasets_to_run:
            short_name = ds["short_name"]
            self.scalability_datasets[short_name] = {}
            total_edges = DATASETS.get(short_name, {}).get("meta", {}).get("edges", 0)

            if total_edges == 0:
                continue

            for algo_name, config in self.active_algos.items():
                algo_type = config.get("type")

                # Grab the 100% formatted file for this specific algorithm type
                base_file_path = self.prepared_dataset[short_name].get(algo_type)

                if not base_file_path:
                    continue

                self.scalability_datasets[short_name][algo_name] = []

                with self.logger.status(f"Slicing [cyan]{short_name}[/cyan] by powers of 2 for {algo_name}..."):
                    for p in self.args.powers:
                        max_edges = 2 ** p
                        is_max = False

                        if max_edges >= total_edges:
                            max_edges = total_edges
                            is_max = True

                        partial_path = create_partial_dataset_by_count(base_file_path, max_edges)

                        # Store the power, max edges, and the sliced path
                        self.scalability_datasets[short_name][algo_name].append((p, max_edges, partial_path))

                        if is_max:
                            break

    def process(self) -> list[dict]:
        metrics = []
        for i, ds in enumerate(self.datasets_to_run, 1):
            short_name, dataset_path = self._get_dataset(ds)
            self._print_status(i, short_name)

            for algo_name, algo_config in self.active_algos.items():
                params = self._resolve_algo_params(algo_config)

                # Fetch the array of sliced paths we prepared in the hook
                slices = self.scalability_datasets.get(short_name, {}).get(algo_name, [])

                for (p, max_edges, partial_path) in slices:
                    self.logger.print(f"--- Evaluating 2^{p} ({max_edges:,} edges) ---")

                    # Explicitly pass the overrides
                    res = self.execute_runner(
                        algo_name=algo_name,
                        dataset_short_name=short_name,
                        params=params,
                        custom_path=partial_path,
                        custom_output_name=f"{short_name}_2^{p}"
                    )

                    if res is not None:
                        t_avg, _, _, _ = res
                        metrics.append({
                            "dataset": short_name,
                            "algorithm": algo_name,
                            "edges_evaluated": max_edges,
                            "power_of_2": p,
                            "accumulated_time_sec": t_avg,
                            **params
                        })
        return metrics

    def output(self, df: pd.DataFrame):
        table = Table(title="Scalability: Accumulated Execution Time", box=box.SIMPLE, show_header=True, header_style="bold yellow")
        table.add_column("Dataset", style="cyan")
        table.add_column("Algorithm", style="green")
        table.add_column("Edges Processed", justify="right")
        table.add_column("Accum. Time (s)", justify="right")

        summary_df = df.groupby(['dataset', 'algorithm', 'power_of_2', 'edges_evaluated'], as_index=False)['accumulated_time_sec'].mean()
        summary_df = summary_df.sort_values(by=["dataset", "algorithm", "power_of_2"])

        for _, row in summary_df.iterrows():
            table.add_row(
                str(row['dataset']),
                str(row['algorithm']),
                f"2^{int(row['power_of_2'])}",
                f"{row['accumulated_time_sec']:.3f}"
            )

        self.logger.print(table)

def main():
    with Scalability() as exp:
        exp.run()

if __name__ == "__main__":
    main()