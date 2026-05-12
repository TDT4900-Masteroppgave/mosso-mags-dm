import math
from pathlib import Path
import pandas as pd
from rich.table import Table
from rich import box
from scripts.config import DATASETS
from scripts.experiments.base_experiment import Experiment


def create_slice(source_path: str, max_lines: int) -> str:
    """
    Blazing fast file slicer.
    Assumes the source file is already cleaned, deduplicated,
    and formatted correctly by datasets.py.
    """
    source = Path(source_path)

    partial_path = source.parent / "partial" / f"n{max_lines}_{source.name}"
    partial_path.parent.mkdir(parents=True, exist_ok=True)

    if partial_path.exists():
        return str(partial_path)

    # Pure text copy (no integer parsing, no sets, no logic = incredibly fast)
    with open(source, "r", encoding="utf-8") as fin, open(partial_path, "w", encoding="utf-8") as out:
        for i, line in enumerate(fin):
            if i >= max_lines:
                break
            out.write(line)

    return str(partial_path)


class Scalability(Experiment):
    def __init__(self):
        super().__init__("scalability")

    def process(self) -> list[dict]:
        metrics = []
        for i, short_name in enumerate(self.datasets.keys(), 1):
            self._print_status(i, short_name)

            total_edges = DATASETS.get(short_name, {}).get("meta", {}).get("edges", 1)
            max_power = int(math.log2(total_edges))
            start_power = max(10, max_power - 8)
            dynamic_powers = list(range(start_power, max_power + 1))

            for algo_name, algo_config in self.active_algos.items():
                params = self._resolve_algo_params(algo_config)

                self.logger.print(
                    f"[dim cyan]Slicing {algo_name} {short_name} (2^{start_power} to 2^{max_power} edges)[/dim cyan]")

                for p in dynamic_powers:
                    max_edges = int(2 ** p)

                    algo_type = algo_config.get("type", None)
                    dataset_path = self._get_dataset_path(short_name, algo_type)
                    if not dataset_path:
                        continue

                    partial_path = create_slice(dataset_path, max_edges)
                    new_ds_name = f"{short_name}_n{max_edges}"
                    res = self.execute_runner(partial_path, new_ds_name, algo_name,  params)
                    if res is None:
                        continue

                    t_avg, _, _, _, _ = res
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
        table = Table(title="Scalability: Accumulated Execution Time", box=box.SIMPLE, show_header=True,
                      header_style="bold yellow")
        table.add_column("Dataset", style="cyan")
        table.add_column("Algorithm", style="green")
        table.add_column("Edges Processed", justify="right")
        table.add_column("Accum. Time (s)", justify="right")

        summary_df = df.groupby(['dataset', 'algorithm', 'power_of_2', 'edges_evaluated'], as_index=False)[
            'accumulated_time_sec'].mean()
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
