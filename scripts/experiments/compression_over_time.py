import json
from scripts.config import DATASETS
from scripts.experiments.base_experiment import Experiment
from pathlib import Path

def create_partial_dataset(
        dataset_path: str,
        fraction: float,
        total_edges: int,
        logger=None,
) -> str:
    """Write a file containing the first (fraction * total_edges) unique edges.

    Cached: if the partial file already exists it is returned immediately.
    Writes are atomic to prevent corrupted cache files if interrupted.
    """
    target_edges = int(total_edges * fraction)

    dataset_file = Path(dataset_path)
    partial_dir = dataset_file.parent / "partial"
    partial_dir.mkdir(parents=True, exist_ok=True)

    partial_path = partial_dir / f"p{int(fraction * 100)}_{dataset_file.name}"

    if partial_path.exists():
        if logger:
            logger.debug(f"Using cached partial dataset: {partial_path}")
        return str(partial_path)

    if logger:
        logger.info(f"Generating partial dataset ({fraction*100:.0f}%): {partial_path}")

    edges_written = 0
    seen: set[tuple[int, int]] = set()

    tmp_path = partial_path.with_suffix('.tmp')

    with open(dataset_file, "r", encoding="utf-8") as f_in, \
            open(tmp_path, "w", encoding="utf-8") as f_out:

        for line in f_in:
            if line.startswith(("#", "%")):
                continue

            # 3. Use split() instead of Regex for a massive speed boost
            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                u, v = int(parts[0]), int(parts[1])
            except ValueError:
                continue  # skip lines that don't start with two integers

            if u == v:
                continue # self loop

            # undirected graph, normalize edge
            edge = (min(u, v), max(u, v))

            if edge not in seen:
                seen.add(edge)
                f_out.write(line)
                edges_written += 1

                if edges_written >= target_edges:
                    break

    tmp_path.rename(partial_path)

    return str(partial_path)

class CompressionOverTime(Experiment):
    DEFAULT_CHECKPOINTS = [0.2, 0.4, 0.6, 0.8, 1.0]

    def __init__(self):
        super().__init__("cot")

    def add_custom_args(self, parser):
        parser.add_argument(
            "--checkpoints", nargs="+", type=float,
            default=self.DEFAULT_CHECKPOINTS,
            help="Edge stream fractions to evaluate algorithms at (default: 0.2 0.4 0.6 0.8 1.0)",
        )

    def process(self, dataset_path: str, dataset_short_name: str) -> list[dict] | None:
        metrics: list[dict] = []
        checkpoints = getattr(self.args, "checkpoints", self.DEFAULT_CHECKPOINTS)
        total_edges = DATASETS[dataset_short_name]["meta"]["edges"]

        for checkpoint in checkpoints:
            self.console.print(f"[bold yellow]--- Evaluating Checkpoint: {checkpoint*100:.0f}% ---[/bold yellow]")

            partial_dataset_path = create_partial_dataset(dataset_path, checkpoint, total_edges)
            if not partial_dataset_path:
                self.logger.error(f"Failed to create partial dataset for {checkpoint}")
                continue

            for algo_name, algo_config in self.active_algos.items():
                params = self._resolve_algo_params(algo_config)

                t_avg, r_avg, t_list, r_list = self.execute_runner(
                    algo_name=algo_name,
                    dataset_path=partial_dataset_path,
                    params=params,
                    dataset_short_name=f"{dataset_short_name}_{checkpoint}",
                )

                if r_avg is not None:
                    metrics.append({
                        "dataset": dataset_short_name,
                        "algorithm": algo_name,
                        "change_ratio": checkpoint,
                        "ratio": r_avg,
                        "parameters": json.dumps(params),
                    })

        return metrics

def main():
    CompressionOverTime().run()

if __name__ == "__main__":
    main()