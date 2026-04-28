import json

from scripts.config import DATASETS
from scripts.experiments.base_experiment import Experiment


class IncrementalVsBatch(Experiment):
    def __init__(self):
        super().__init__("ivb")

    def process(self, dataset_path: str, dataset_short_name: str) -> list[dict] | None:
        metrics: list[dict] = []

        meta = DATASETS.get(dataset_short_name, {}).get("meta", {})
        total_edges = meta.get("edges", 1)

        for algo_name, algo_config in self.active_algos.items():
            params = self._resolve_algo_params(algo_config)

            t_avg, r_avg, t_list, r_list = self.execute_runner(
                algo_name=algo_name,
                dataset_path=dataset_path,
                params=params,
                dataset_short_name=dataset_short_name,
            )

            if t_list is None or r_list is None:
                continue

            for i, (t, r) in enumerate(zip(t_list, r_list)):
                t_micros = t * 1_000_000

                metrics.append({
                    "dataset": dataset_short_name,
                    "algorithm": algo_name,
                    "run": i + 1,
                    # Plot this column for Batch Algorithms
                    "time_micros": t_micros,
                    # Plot this column for Streaming Algorithms
                    "time_per_change": t_micros / total_edges,
                    "ratio": r,
                    "parameters": json.dumps(params),
                })

        return metrics


def main():
    IncrementalVsBatch().run()


if __name__ == "__main__":
    main()