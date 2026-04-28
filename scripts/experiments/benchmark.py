import json

from scripts.config import DATASETS
from scripts.experiments.base_experiment import Experiment


class Benchmark(Experiment):
    def __init__(self):
        super().__init__("compare")

    def process(self, dataset_path: str, dataset_short_name: str) -> list[dict] | None:
        total_edges = DATASETS.get(dataset_short_name, {}).get("meta", {}).get("edges", 1)
        metrics : list[dict] = []
        for algo_name, algo_config in self.active_algos.items():
            params = self._resolve_algo_params(algo_config)

            t_avg, r_avg, t_list, r_list = self.execute_runner(
                algo_name=algo_name,
                dataset_path=dataset_path,
                params=params,
                dataset_short_name=dataset_short_name,
            )

            for i, (t, r) in enumerate(zip(t_list, r_list)):
                metrics.append({
                    "dataset": dataset_short_name,
                    "algorithm": algo_name,
                    "run": i + 1,
                    "time": t,
                    "ratio": r,
                    "time_per_edge": t / total_edges * 1_000_000,
                    "parameters": json.dumps(params),
                })

        return metrics


def main():
    Benchmark().run()


if __name__ == "__main__":
    main()
