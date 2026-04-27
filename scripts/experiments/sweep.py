from scripts.config import PARAM_CONFIG
from scripts.experiments.base_experiment import Experiment


class ParameterSweep(Experiment):
    def __init__(self):
        super().__init__("sweep")
        self.all_times_dict, self.all_ratios_dict = {}, {}
        config = PARAM_CONFIG[self.args.param]
        self.sweep_values = list(range(*self.args.range)) if self.args.range else (
            self.args.values if self.args.values else config["bounds"])

    def add_custom_args(self, parser):
        parser.add_argument("--param", choices=list(PARAM_CONFIG.keys()), required=True)
        parser.add_argument("--range", type=int, nargs=3, required=True)

    def process(self, dataset_path: str, dataset_short_name: str) -> None:
        for val in self.sweep_values:
            self.logger.info(f"{self.args.param} = {val}:")
            for algo_name, algo_config in self.active_algos.items():
                if self.args.param not in algo_config.get('template', {}):
                    self.logger.info(f"=> Skipping {algo_name}: No such parameter {self.args.param}")
                    continue

                params = self._resolve_algo_params(algo_config)
                params.update({self.args.param: str(val)})

                raw_runs = self.execute_runner(
                    algo_name=algo_name,
                    dataset_path=dataset_path,
                    params=params,
                    dataset_short_name=dataset_short_name,
                )
                if not raw_runs:
                    continue

                for row in raw_runs:
                    self.results.append(row)

def main():
    ParameterSweep().run()


if __name__ == "__main__":
    ParameterSweep().run()
