from pathlib import Path
import pandas as pd
from rich.table import Table
from rich import box
from scripts.config import DATASETS
from scripts.experiments.base_experiment import Experiment

def create_slice(source_path: str, max_lines: int, fraction: float) -> str:
    """
    Blazing fast file slicer.
    Assumes the source file is already cleaned, deduplicated,
    and formatted correctly by datasets.py.
    """
    source = Path(source_path)

    # Save the partials in the algorithm's specific folder
    partial_path = source.parent / "partial" / f"cp{fraction}_{source.name}"
    partial_path.parent.mkdir(parents=True, exist_ok=True)

    # Force rebuild to ensure clean data
    if partial_path.exists():
        return str(partial_path)

    # Pure text copy (incredibly fast)
    with open(source, "r", encoding="utf-8") as fin, open(partial_path, "w", encoding="utf-8") as out:
        for i, line in enumerate(fin):
            if i >= max_lines:
                break
            out.write(line)

    return str(partial_path)


class CompressionCheckpoints(Experiment):
    def __init__(self):
        super().__init__("compression_checkpoints")

    def process(self) -> list[dict]:
        metrics = []
        checkpoints = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for i, short_name in enumerate(self.datasets, 1):
            self._print_status(i, short_name)
            total_edges = DATASETS.get(short_name, {}).get("meta", {}).get("edges", 1)

            for algo_name, algo_config in self.active_algos.items():
                params = self._resolve_algo_params(algo_config)
                algo_type = algo_config.get("type", algo_name)

                dataset_path = self._get_dataset_path(short_name, algo_type)
                if not dataset_path:
                    continue

                if algo_type == "mosso":
                    self.logger.print(f"[dim cyan]Evaluating {algo_name} continuously (Streaming)[/dim cyan]")

                    new_ds_name = f"{short_name}_streaming_full"
                    res = self.execute_runner(dataset_path, new_ds_name, algo_name, params)
                    if res is None: continue

                    t_avg, r_avg, _, _, intermediates = res

                    for pt in intermediates:
                        fraction = pt["edges"] / total_edges

                        if fraction >= 1.0:
                            continue

                        metrics.append({
                            "dataset": short_name,
                            "algorithm": algo_name,
                            "change_ratio": fraction,
                            "time": pt["time"],
                            "ratio": pt["ratio"],
                            "is_streaming": True,
                            **params
                        })

                    metrics.append({
                        "dataset": short_name,
                        "algorithm": algo_name,
                        "change_ratio": 1.0,
                        "time": t_avg,
                        "ratio": r_avg,
                        "is_streaming": True,
                        **params
                    })
                else:
                    for cp in checkpoints:
                        fraction = float(cp)
                        max_edges = int(total_edges * fraction)

                        self.logger.print(f"[dim cyan]Evaluating {algo_name} at {fraction*100:.0f}% ({max_edges} edges)[/dim cyan]")

                        partial_path = create_slice(dataset_path, max_edges, fraction)
                        new_ds_name = f"{short_name}_cp{fraction}"
                        res = self.execute_runner(partial_path, new_ds_name, algo_name,  params)
                        if res is None:
                            continue

                        t_avg, r_avg, _, _, _ = res
                        metrics.append({
                            "dataset": short_name,
                            "algorithm": algo_name,
                            "change_ratio": fraction,
                            "time": t_avg,
                            "ratio": r_avg,
                            "is_streaming": False,
                            **params
                        })
        return metrics

    def output(self, df: pd.DataFrame):
        table = Table(title="Compression Compactness over Time", box=box.SIMPLE, show_header=True, header_style="bold yellow")
        table.add_column("Dataset", style="cyan")
        table.add_column("Algorithm", style="green")
        table.add_column("Checkpoint", justify="right")
        table.add_column("Ratio", justify="right")

        summary_df = df.groupby(['dataset', 'algorithm', 'change_ratio'], as_index=False)['ratio'].mean()
        summary_df = summary_df.sort_values(by=["dataset", "algorithm", "change_ratio"])

        for _, row in summary_df.iterrows():
            ratio_val = f"{row['ratio']:.4f}" if pd.notna(row['ratio']) else "N/A"
            table.add_row(
                str(row['dataset']),
                str(row['algorithm']),
                f"{float(row['change_ratio']) * 100:.1f}%",
                ratio_val
            )

        self.logger.print(table)

def main():
    with CompressionCheckpoints() as exp:
        exp.run()

if __name__ == "__main__":
    main()