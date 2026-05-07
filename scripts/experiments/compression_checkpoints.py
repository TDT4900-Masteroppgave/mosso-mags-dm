from pathlib import Path
import pandas as pd
from rich.table import Table
from rich import box
from scripts.config import DATASETS
from scripts.experiments.base_experiment import Experiment

def create_partial_dataset(dataset_path: str, fraction: float, total_edges: int) -> str:
    path = Path(dataset_path)
    partial_path = path.parent / "partial" / f"p{int(fraction * 100)}_{path.name}"
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    if partial_path.exists(): return str(partial_path)

    seen, target = set(), int(total_edges * fraction)
    tmp = partial_path.with_suffix('.tmp')

    with open(path, "r", encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as f_out:
        for line in fin:
            if line.startswith(("#", "%")): continue
            try:
                u, v = map(int, line.split()[:2])
                if u != v and (edge := (min(u, v), max(u, v))) not in seen:
                    seen.add(edge)
                    f_out.write(line)
                    if len(seen) >= target: break
            except ValueError: continue
    tmp.rename(partial_path)
    return str(partial_path)

class CompressionCheckpoints(Experiment):
    def __init__(self):
        self.fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        super().__init__("compression_checkpoints")

        # New dictionary just for this experiment
        self.checkpoint_datasets = {}

    def post_preprocessing(self):
        """Overrides the base hook to generate partial datasets for ALL algorithms."""
        self.logger.print(f"Generating Checkpoint Datasets")
        for ds in self.datasets_to_run:

            short_name = ds["short_name"]

            # Use a dictionary to store paths per algorithm
            self.checkpoint_datasets[short_name] = {}
            total_edges = DATASETS.get(short_name, {}).get("meta", {}).get("edges", 0)

            if total_edges == 0:
                continue

            for algo_name, config in self.active_algos.items():
                algo_type = config.get("type")

                # Grab the 100% formatted file for this specific algorithm type
                base_file_path = self.prepared_dataset[short_name].get(algo_type)

                if not base_file_path:
                    continue

                self.checkpoint_datasets[short_name][algo_name] = []

                with self.logger.status(f"Slicing [cyan]{short_name}[/cyan] into {len(self.fractions)} checkpoints for {algo_name}..."):
                    for fraction in self.fractions:
                        # This safely copies the exact formatting needed (e.g., Mosso's \t1)
                        partial_path = create_partial_dataset(base_file_path, fraction, total_edges)
                        self.checkpoint_datasets[short_name][algo_name].append(partial_path)

    def add_custom_args(self, parser):
        parser.add_argument("--checkpoints", nargs="+", type=float, default=self.fractions)

    def process(self) -> list[dict]:
        metrics = []
        for i, ds in enumerate(self.datasets_to_run, 1):
            short_name, dataset_path = self._get_dataset(ds)
            self._print_status(i, short_name)

            for cp_idx, cp in enumerate(self.fractions):
                self.logger.print(f"--- Evaluating Checkpoint: {int(cp * 100)}% ---")

                for algo_name, algo_config in self.active_algos.items():
                    params = self._resolve_algo_params(algo_config)

                    partial_path = self.checkpoint_datasets[short_name][algo_name][cp_idx]

                    res = self.execute_runner(
                        algo_name=algo_name,
                        dataset_short_name=short_name, # Keeps the dictionary lookup safe
                        params=params,
                        custom_path=partial_path,
                        custom_output_name=f"{short_name}_{cp}"
                    )

                    if res is not None:
                        _, r_avg, _, _ = res
                        metrics.append({
                            "dataset": short_name,
                            "algorithm": algo_name,
                            "change_ratio": cp,
                            "ratio": r_avg,
                            **params
                        })
        return metrics

    def output(self, df: pd.DataFrame):
        table = Table(title="Compression Ratio Over Time", box=box.SIMPLE, show_header=True, header_style="bold yellow")
        table.add_column("Dataset", style="cyan")
        table.add_column("Algorithm", style="green")
        table.add_column("Checkpoint", justify="right")
        table.add_column("Avg Ratio", justify="right")

        summary_df = df.groupby(['dataset', 'algorithm', 'change_ratio'], as_index=False)['ratio'].mean()
        summary_df = summary_df.sort_values(by=["dataset", "algorithm", "change_ratio"])

        for _, row in summary_df.iterrows():
            pct = float(row['change_ratio']) * 100
            table.add_row(str(row['dataset']), str(row['algorithm']), f"{pct:.0f}%", f"{row['ratio']:.5f}")

        self.logger.print(table)

def main():
    with CompressionCheckpoints() as exp:
        exp.run()

if __name__ == "__main__":
    main()