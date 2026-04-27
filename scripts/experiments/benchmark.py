from scripts.config import DATASETS
from scripts.experiments.base_experiment import Experiment


class Benchmark(Experiment):
    def __init__(self):
        super().__init__("compare")

    def process(self, dataset_path: str, dataset_short_name: str) -> None:
        total_edges = DATASETS.get(dataset_short_name, {}).get("meta", {}).get("edges", 1)
        for algo_name, algo_config in self.active_algos.items():
            params = self._resolve_algo_params(algo_config)

            raw_runs = self.execute_runner(
                algo_name=algo_name,
                dataset_path=dataset_path,
                params=params,
                dataset_short_name=dataset_short_name,
            )
            if not raw_runs:
                continue

            for row in raw_runs:
                row["time_per_edge"] = (row["time"] / total_edges) * 1_000_000 if row["time"] else None
                self.results.append(row)

def main():
    Benchmark().run()


if __name__ == "__main__":
    main()
