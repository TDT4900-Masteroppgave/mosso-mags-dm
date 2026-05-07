import sys
import shutil
import subprocess
from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path

from scripts.config import ALGORITHMS, ALGORITHMS_DIR
from scripts.datasets import retrieve_github_code
from scripts.datasets import clean_and_write, generate_dynamic_stream_graph, generate_dynamic_batch_graph
from scripts.config import DATASET_GROUP


class AlgorithmRunner(ABC):
    edge_format_string = "{u}\t{v}\n"

    def __init__(self, algo_name: str, config: dict, logger, session_dir: str):
        self.algo_name = algo_name
        self.config = config
        self.logger = logger
        self.session_dir = Path(session_dir)
        self.run_log_dir = self.session_dir / "run_log"
        self.run_log_dir.mkdir(parents=True, exist_ok=True)

        self.target_dir = Path(self.config.get("target_dir", ALGORITHMS_DIR / algo_name))
        self.is_local = str(self.target_dir) == "."

    @abstractmethod
    def get_binary_path(self) -> Path:
        pass

    @abstractmethod
    def compile_logic(self) -> None:
        pass

    @abstractmethod
    def build_command(self, dataset_path: str, graph_output_path: str, parameters: list[str]) -> list[str]:
        pass

    @abstractmethod
    def parse_output(self, stdout: str) -> tuple[Optional[float], Optional[float]]:
        pass

    def _execute_cmd(self, cmd: list[str], cwd=None, env=None, log_msg: Optional[str] = None):
        if log_msg:
            self.logger.debug(log_msg)
        try:
            return subprocess.run(
                cmd, cwd=cwd, env=env, check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"[!] Command failed: {' '.join(cmd)}\nSTDERR: {e.stderr}")
            raise

    def binary_exists(self) -> bool:
        return self.get_binary_path().exists()

    def build(self) -> None:
        try:
            if not self.is_local:
                repo_url, branch = str(self.config["repo"]), str(self.config["branch"])
                self.logger.debug(f"[{self.algo_name}] Source: {repo_url.split('/')[-1]} (Branch: {branch})")
                retrieve_github_code(str(self.target_dir), self.algo_name, repo_url, branch, self.logger)

            self.logger.debug(f"\t-> [{self.algo_name}] Compiling binaries")
            self.compile_logic()
            self.logger.debug(f"\t-> [{self.algo_name}] [OK] Build successful")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"\t-> [{self.algo_name}] [!] Compilation failed. Code {e.returncode}")
            if e.stdout: self.logger.debug(f"STDOUT:\n{e.stdout.strip()}")
            if e.stderr: self.logger.debug(f"STDERR:\n{e.stderr.strip()}")
            if not self.is_local and self.target_dir.exists():
                shutil.rmtree(self.target_dir)
            raise RuntimeError(f"Compilation failed for {self.algo_name}.") from e

    def run_single(self, format_dataset_path: str, output_name: str, parameters: list) -> tuple[
        Optional[float], Optional[float]]:
        graph_output_path = self.session_dir / output_name
        cmd = self.build_command(format_dataset_path, str(graph_output_path), parameters)
        self.logger.debug(f"[{self.algo_name}] Running command: {' '.join(cmd)}")

        try:
            result = self._execute_cmd(cmd)
            parsed_time, parsed_ratio = self.parse_output(result.stdout)

            log_file = self.run_log_dir / f"{output_name}.log"
            log_file.write_text(
                f"EXECUTION COMMAND:\n{' '.join(cmd)}\n\n"
                f"{'=' * 20} STDOUT {'=' * 20}\n{result.stdout}\n"
                f"{'=' * 20} STDERR {'=' * 20}\n{result.stderr}",
                encoding="utf-8"
            )

            if graph_output_path.exists():
                graph_output_path.unlink(missing_ok=True)

            return parsed_time, parsed_ratio

        except subprocess.CalledProcessError as e:
            self.logger.error(f"[!] Execution crashed for {output_name}: {e}\nSTDERR:\n{e.stderr}")
        except Exception as e:
            self.logger.error(f"[!] Unexpected error for {output_name}: {e}")
        return None, None

    def format_dataset(self, dataset_path: str, short_name: str) -> str:
        orig = Path(dataset_path)

        # Create algorithm-specific directory
        save_dir = orig.parent / self.__class__.__name__
        save_dir.mkdir(exist_ok=True)

        # Determine stream type
        is_dynamic = "dynamic" in sys.argv and short_name in DATASET_GROUP.get("dynamic", [])
        stream_type = "FD" if is_dynamic else "IO"

        # Safe filename combining class name, dataset, and stream type
        formatted_path = save_dir / f"{self.__class__.__name__}_{orig.stem}_{stream_type}.txt"

        if not formatted_path.exists():
            self.logger.debug(f"Formatting dataset for {self.algo_name} ({stream_type}): {orig.name}")

            if stream_type == "IO":
                # For Insertion-Only, just apply the runner's specific format
                clean_and_write(str(orig), str(formatted_path), self.edge_format_string)

            elif stream_type == "FD":
                # Clean first with a STRICT 2-column format to ensure no duplicates
                temp_clean = orig.parent / f"{orig.stem}_temp_clean.txt"
                if not temp_clean.exists():
                    clean_and_write(str(orig), str(temp_clean), self.edge_format_string)

                # Branch logic based on algorithm type
                algo_type = self.config.get("type", None)

                if algo_type == "mosso":
                    # Creates the timeline with +1 and -1
                    generate_dynamic_stream_graph(str(temp_clean), str(formatted_path), p_delete=0.1)

                elif algo_type in ["mags"]:
                    # Creates the final state graph (dropping 10% of edges)
                    generate_dynamic_batch_graph(str(temp_clean), str(formatted_path), p_delete=0.1)

        return str(formatted_path)

    def run_multiple(self, dataset_path: str, short_name: str, output_name: str, n_runs: int, parameters: list) \
            -> tuple[Optional[float], Optional[float], list, list]:
        format_dataset_path = self.format_dataset(dataset_path, short_name)
        times, ratios = [], []

        for i in range(n_runs):
            self.logger.debug(f"Iter {i + 1}/{n_runs} for {output_name}")
            t, r = self.run_single(format_dataset_path, f"{output_name}_run{i + 1}", parameters)
            if t is not None and r is not None:
                times.append(t)
                ratios.append(r)

        avg_t = sum(times) / len(times) if times else None
        avg_r = sum(ratios) / len(ratios) if ratios else None
        return avg_t, avg_r, times, ratios


def get_runner(algo_name: str, logger, session_dir: str) -> AlgorithmRunner:
    from scripts.runners.mags_runner import MagsRunner
    from scripts.runners.mosso_runner import MossoRunner

    config = ALGORITHMS.get(algo_name)
    if not config:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    algo_type = config.get("type")
    if algo_type == "mosso": return MossoRunner(algo_name, config, logger, session_dir)
    if algo_type == "mags":  return MagsRunner(algo_name, config, logger, session_dir)

    raise ValueError(f"Unknown algorithm type: {algo_type} for {algo_name}")
