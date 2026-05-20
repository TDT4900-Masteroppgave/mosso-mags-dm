import math
from pathlib import Path
import pandas as pd
from rich.table import Table
from rich import box
from scripts.config import DATASETS, DATASETS_DIR
from scripts.datasets import extract_batch_snapshot
from scripts.experiments.base_experiment import Experiment


def count_lines(source_path: str) -> int:
    with open(source_path, "r", encoding="utf-8") as fin:
        return sum(1 for _ in fin)


def resolve_stream_path(short_name: str, prepared_paths: dict[str, str]) -> str | None:
    if prepared_paths.get("mosso"):
        return prepared_paths["mosso"]

    filename = DATASETS.get(short_name, {}).get("filename")
    if not filename:
        return None

    master_stream = DATASETS_DIR / f"{Path(filename).stem}_master_stream.txt"
    return str(master_stream) if master_stream.exists() else None


def create_slice(source_path: str, max_lines: int) -> str:
    """
    Blazing fast file slicer.
    Assumes the source file is already cleaned, deduplicated,
    and formatted correctly by datasets.py.
    """
    source = Path(source_path)

    partial_path = source.parent / "partial" / f"n{max_lines}_{source.name}"
    partial_path.parent.mkdir(parents=True, exist_ok=True)

    if partial_path.exists() and partial_path.stat().st_mtime >= source.stat().st_mtime:
        return str(partial_path)

    # Pure text copy (no integer parsing, no sets, no logic = incredibly fast)
    with open(source, "r", encoding="utf-8") as fin, open(partial_path, "w", encoding="utf-8") as out:
        for i, line in enumerate(fin):
            if i >= max_lines:
                break
            out.write(line)

    return str(partial_path)


def create_batch_snapshot(prefix_stream_path: str, max_changes: int) -> str:
    """
    Materialize the static graph snapshot after the first max_changes stream events.
    This lets batch algorithms be evaluated against the same point in the stream
    as incremental algorithms.
    """
    prefix_stream = Path(create_slice(prefix_stream_path, max_changes))
    snapshot_path = prefix_stream.parent / f"snapshot_{prefix_stream.stem}.txt"

    if snapshot_path.exists() and snapshot_path.stat().st_mtime >= prefix_stream.stat().st_mtime:
        return str(snapshot_path)

    extract_batch_snapshot(str(prefix_stream), str(snapshot_path))
    return str(snapshot_path)


def get_checkpoints(total_changes: int) -> list[int]:
    if total_changes <= 0:
        return []

    max_power = int(math.log2(total_changes))
    start_power = max(10, max_power - 8)
    powers = [2 ** p for p in range(start_power, max_power + 1)]

    return sorted({p for p in powers if p <= total_changes} | {total_changes})


def log2_slope(group: pd.DataFrame) -> float | None:
    valid = group[(group["edges_evaluated"] > 0) & (group["accumulated_time_sec"] > 0)]
    if len(valid) < 2:
        return None

    x = valid["edges_evaluated"].apply(math.log2)
    y = valid["accumulated_time_sec"].apply(math.log2)
    return float(y.cov(x) / x.var()) if x.var() else None


def compute_slopes(df: pd.DataFrame) -> dict[tuple[str, str], float]:
    slopes = {}
    for key, group in df.groupby(["dataset", "algorithm"]):
        slope = log2_slope(group)
        if slope is not None:
            slopes[key] = slope
    return slopes


class Scalability(Experiment):
    def __init__(self):
        super().__init__("scalability")

    def process(self) -> None:
        for i, short_name in enumerate(self.datasets.keys(), 1):
            self._print_status(i, short_name)

            is_dynamic = short_name in self.args.dynamic
            stream_path = resolve_stream_path(short_name, self.datasets[short_name])
            reference_path = stream_path if is_dynamic else next(iter(self.datasets[short_name].values()))
            if not reference_path:
                self.logger.warning(f"[!] No reference stream found for {short_name}. Skipping.")
                continue

            total_changes = count_lines(reference_path)
            checkpoints = get_checkpoints(total_changes)

            if not checkpoints:
                self.logger.warning(f"[!] No changes found for {short_name}. Skipping.")
                continue

            for algo_name, algo_config in self.active_algos.items():
                params = self._resolve_algo_params(algo_config)
                algo_type = algo_config.get("type", None)
                dataset_path = self._get_dataset_path(short_name, algo_type)
                if not dataset_path:
                    continue

                self.logger.print(
                    f"[dim cyan]Slicing {algo_name} {short_name} ({len(checkpoints)} checkpoints up to {total_changes:,} changes)[/dim cyan]")

                for changes in checkpoints:
                    if algo_type == "mags" and is_dynamic:
                        if not stream_path:
                            self.logger.warning(
                                f"[!] Cannot build dynamic snapshots for {algo_name} on {short_name} without a stream file. Skipping."
                            )
                            continue
                        experiment_path = create_batch_snapshot(stream_path, changes)
                    else:
                        experiment_path = create_slice(dataset_path, changes)

                    new_ds_name = f"{short_name}_n{changes}"
                    res = self.execute_runner(experiment_path, new_ds_name, algo_name,  params)
                    if res is None:
                        continue

                    t_avg, _, _, _, _ = res
                    self.record_result({
                        "dataset": short_name,
                        "algorithm": algo_name,
                        "changes_evaluated": changes,
                        "edges_evaluated": changes,
                        "power_of_2": math.log2(changes),
                        "accumulated_time_sec": t_avg,
                        **params
                    })

    def output(self, df: pd.DataFrame):
        table = Table(title="Scalability: Accumulated Execution Time", box=box.SIMPLE, show_header=True,
                      header_style="bold yellow")
        table.add_column("Dataset", style="cyan")
        table.add_column("Algorithm", style="green")
        table.add_column("Changes Processed", justify="right")
        table.add_column("Accum. Time (s)", justify="right")
        table.add_column("Log-log Slope", justify="right")

        summary_df = df.groupby(['dataset', 'algorithm', 'power_of_2', 'edges_evaluated'], as_index=False)[
            'accumulated_time_sec'].mean()
        summary_df = summary_df.sort_values(by=["dataset", "algorithm", "power_of_2"])
        slopes = compute_slopes(summary_df)

        for _, row in summary_df.iterrows():
            slope = slopes.get((row["dataset"], row["algorithm"]))
            table.add_row(
                str(row['dataset']),
                str(row['algorithm']),
                f"{int(row['edges_evaluated']):,}",
                f"{row['accumulated_time_sec']:.3f}",
                f"{slope:.2f}" if slope is not None else "N/A"
            )

        self.logger.print(table)


def main():
    with Scalability() as exp:
        exp.run()


if __name__ == "__main__":
    main()
