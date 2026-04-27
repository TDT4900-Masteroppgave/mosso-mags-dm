import json
from typing import Any

from scripts.benchmark import Benchmark

class CompareBenchmark(Benchmark):
    def __init__(self):
        super().__init__("compare")

    def process(self, dataset_path: str, dataset_short_name: str) -> list[dict[str, Any]] | None:
        runs = []
        for algo_name, algo_config in self.active_algos.items():
            params = self._resolve_algo_params(algo_config)

            t_avg, r_avg, t_list, r_list = self.execute_runner(
                algo_name=algo_name,
                dataset_path=dataset_path,
                parameters=params
            )

            if t_list is None or r_list is None:
                continue

            for i, (t, r) in enumerate(zip(t_list, r_list)):
                runs.append({
                    "dataset": dataset_short_name,
                    "algorithm": algo_name,
                    "run": i + 1,
                    "time": t,
                    "ratio": r,
                    "parameters": params,
                })

        return runs

def main():
    CompareBenchmark().run()


if __name__ == "__main__":
    main()
