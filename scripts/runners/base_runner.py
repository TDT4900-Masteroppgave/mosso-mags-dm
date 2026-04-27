"""Algorithm runners: build orchestration, execution, and output parsing for Mosso and MAGS."""
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from scripts.config import ALGORITHMS, ALGORITHMS_DIR
from scripts.datasets import clean_and_write, retrieve_github_code


class AlgorithmRunner(ABC):
    edge_format_string = "{u}\t{v}\n"

    def __init__(self, algo_name: str, config: dict, logger, session_dir: str):
        self.algo_name = algo_name
        self.config = config
        self.logger = logger
        self.session_dir = Path(session_dir)
        self.target_dir = Path(self.config.get("target_dir", ALGORITHMS_DIR / algo_name))
        self.is_local = str(self.target_dir) == "."

    @abstractmethod
    def get_binary_path(self) -> str:
        pass

    @abstractmethod
    def compile_logic(self) -> None:
        pass

    def _run_cmd(self, cmd: list[str], cwd=None, env=None):
        try:
            return subprocess.run(
                cmd, cwd=cwd, env=env, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"[!] Command failed: {' '.join(cmd)}\nSTDERR: {e.stderr}")
            raise

    def binary_exists(self) -> bool:
        return os.path.exists(self.get_binary_path())

    def build(self) -> None:
        try:
            if not self.is_local:
                repo_url = str(self.config["repo"])
                branch = str(self.config["branch"])
                self.logger.debug(
                    f"[{self.algo_name}] Source: {repo_url.split('/')[-1]} (Branch: {branch})"
                )
                retrieve_github_code(str(self.target_dir), self.algo_name, repo_url, branch, self.logger)

            self.logger.debug(f"\t-> [{self.algo_name}] Compiling binaries...")
            self.compile_logic()
            self.logger.debug(f"\t-> [{self.algo_name}] [OK] Build successful.")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"\t-> [{self.algo_name}] [!] Compilation failed. Code {e.returncode}")
            if e.stdout:
                self.logger.debug(f"STDOUT:\n{e.stdout.strip()}")
            if e.stderr:
                self.logger.debug(f"STDERR:\n{e.stderr.strip()}")
            if not self.is_local and os.path.exists(self.target_dir):
                shutil.rmtree(self.target_dir)
            raise RuntimeError(f"Compilation failed for {self.algo_name}.") from e

    @abstractmethod
    def build_command(self, dataset_path: str, graph_output_path: str, parameters: list[str]) -> list[str]:
        pass

    @abstractmethod
    def parse_output(self, stdout: str) -> tuple[Optional[float], Optional[float]]:
        pass

    def run_single(
            self,
            format_dataset_path: str,
            output_name: str,
            parameters: list
    ) -> tuple[Optional[float], Optional[float]]:
        """Execute one trial. Returns (time, ratio, peak_rss_mb)."""
        runs_dir = self.session_dir / "runs"
        graph_output_path = self.session_dir / output_name

        cmd = self.build_command(format_dataset_path, str(graph_output_path), parameters)
        self.logger.debug(f"-> [{self.algo_name}] Running command: {' '.join(cmd)}")

        try:
            result = self._run_cmd(cmd)
            parsed_time, parsed_ratio = self.parse_output(result.stdout)

            run_log_file = runs_dir / f"{output_name}.log"
            with open(run_log_file, "w", encoding="utf-8") as f:
                f.write(f"EXECUTION COMMAND:\n{' '.join(cmd)}\n\n")
                f.write("=" * 20 + " STDOUT " + "=" * 20 + "\n")
                f.write(result.stdout)
                f.write("\n" + "=" * 20 + " STDERR " + "=" * 20 + "\n")
                f.write(result.stderr)


            if graph_output_path.exists():
                try:
                    os.remove(graph_output_path)
                except OSError as e:
                    self.logger.debug(f"[!] Cleanup failed for {graph_output_path}: {e}")

            return parsed_time, parsed_ratio

        except subprocess.CalledProcessError as e:
            self.logger.error(f"[!] Execution crashed for {output_name}: {e}")
            self.logger.debug(f"[!] Command was: {' '.join(cmd)}")
            self.logger.debug(f"[!] Error Output:\n{e.stderr}")
            return None, None

        except Exception as e:
            self.logger.error(f"[!] Unexpected error for {output_name}: {e}")
            return None, None

    def format_dataset(self, original_dataset_path: str) -> str:
        """Return path to cleaned dataset, creating it on first call (cached)."""
        basename = os.path.basename(original_dataset_path)
        base_dir = os.path.dirname(original_dataset_path)
        save_dir = os.path.join(base_dir, self.__class__.__name__)
        os.makedirs(save_dir, exist_ok=True)
        formatted_path = os.path.join(save_dir, f"{self.__class__.__name__}_{basename}")
        if os.path.exists(formatted_path):
            return formatted_path
        self.logger.debug(f"\t[*] Cleaning dataset for {self.__class__.__name__}: {basename}")
        clean_and_write(original_dataset_path, formatted_path, self.edge_format_string)
        return formatted_path

    def _warmup(self, format_dataset_path: str, base_output_name: str, parameters: list[str]) -> None:
        self.logger.debug(f"\t=> Executing Warmup Run for {self.algo_name}...")
        self.run_single(
            format_dataset_path, f"{base_output_name}_warmup",
            parameters,
        )

    def _iterate_trials(
            self, format_dataset_path: str, base_output_name: str, runs: int,
            parameters: list[str],
    ) -> tuple[list, list]:
        times, ratios = [], []
        for i in range(runs):
            self.logger.debug(f"Iter {i + 1}/{runs} for {base_output_name}...")
            t, r = self.run_single(format_dataset_path, f"{base_output_name}_run{i + 1}", parameters)
            if t is not None and r is not None:
                times.append(t)
                ratios.append(r)
        return times, ratios

    @staticmethod
    def _aggregate(times: list, ratios: list) -> tuple[Optional[float], Optional[float]]:
        avg_t = sum(times) / len(times) if times else None
        avg_r = sum(ratios) / len(ratios) if ratios else None
        return avg_t, avg_r

    def run_multiple(
            self,
            dataset_path: str,
            base_output_name: str,
            runs: int,
            parameters: list,
    ) -> tuple[float | None, float | None, list, list]:
        """Execute multiple trials."""
        format_dataset_path = self.format_dataset(dataset_path)
        #if runs > 1:
        #    self._warmup(format_dataset_path, base_output_name, parameters)
        times, ratios = self._iterate_trials(format_dataset_path, base_output_name, runs, parameters)
        avg_t, avg_r = self._aggregate(times, ratios)
        return avg_t, avg_r, times, ratios


def get_runner(algo_name: str, logger, session_dir: str) -> AlgorithmRunner:
    from scripts.runners.mags_runner import MagsRunner
    from scripts.runners.mosso_runner import MossoRunner

    config = ALGORITHMS.get(algo_name)
    if not config:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    algo_type = config.get("type")
    if algo_type == "mosso":
        return MossoRunner(algo_name, config, logger, session_dir)
    if algo_type == "mags":
        return MagsRunner(algo_name, config, logger, session_dir)
    raise ValueError(f"Unknown algorithm type: {algo_type} for {algo_name}")
