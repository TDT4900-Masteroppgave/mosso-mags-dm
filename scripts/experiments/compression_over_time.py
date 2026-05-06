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

class CompressionOverTime(Experiment):
    DEFAULT_CHECKPOINTS = [0.2, 0.4, 0.6, 0.8, 1.0]

    def __init__(self):
        super().__init__("cot")

    def add_custom_args(self, parser):
        parser.add_argument("--checkpoints", nargs="+", type=float, default=self.DEFAULT_CHECKPOINTS)

    def process(self) -> list[dict]:
        metrics = []
        for i, ds in enumerate(self.datasets_to_run, 1):
            short_name, dataset_path = self._get_dataset(ds)
            self._print_status(i, short_name)
            total_edges = DATASETS[short_name]["meta"]["edges"]
            checkpoints = getattr(self.args, "checkpoints", self.DEFAULT_CHECKPOINTS)

            for cp in checkpoints:
                self.logger.print(f"[bold yellow]--- Evaluating Checkpoint: {cp * 100:.0f}% ---[/bold yellow]")

                partial_path = create_partial_dataset(dataset_path, cp, total_edges)
                if not partial_path:
                    continue

                for algo_name, algo_config in self.active_algos.items():
                    params = self._resolve_algo_params(algo_config)
                    res = self.execute_runner(algo_name, partial_path, f"{short_name}_{cp}", params)

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
    with CompressionOverTime() as exp:
        exp.run()

if __name__ == "__main__":
    main()