"""Algorithm runners: build orchestration, execution, and output parsing for Mosso and MAGS."""
import os
import platform
import re
import resource
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from scripts.config import ALGORITHMS, VERSIONS_DIR
from scripts.datasets import clean_and_write, retrieve_github_code
from scripts.utils import get_fastutil_path


# ---------------------------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------------------------

def _peak_rss_mb() -> float:
    """Peak RSS of child processes in MB (macOS bytes, Linux KB)."""
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    if platform.system() == "Darwin":
        return ru.ru_maxrss / 1024 / 1024
    return ru.ru_maxrss / 1024


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def _run_build(cmd: list[str], cwd=None, env=None):
    return subprocess.run(
        cmd, cwd=cwd, env=env, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def _mags_build_env() -> dict:
    env = os.environ.copy()
    env["CXXFLAGS"] = f"{env.get('CXXFLAGS', '')} -O3"
    if platform.system() == "Darwin":
        for prefix in ["/opt/homebrew", "/usr/local"]:
            llvm = f"{prefix}/opt/llvm/bin"
            if os.path.exists(f"{llvm}/clang++"):
                env["PATH"] = f"{llvm}:{env.get('PATH', '')}"
                env["CC"] = f"{llvm}/clang"
                env["CXX"] = f"{llvm}/clang++"
                break
    return env


def _apply_source_hotfixes(target_dir: Path) -> None:
    """Patch known upstream typos in MAGS source."""
    pgsum = target_dir / "src" / "pgsum.cpp"
    if pgsum.exists():
        content = pgsum.read_text(encoding="utf-8")
        if "#pragma omp barier" in content:
            pgsum.write_text(
                content.replace("#pragma omp barier", "#pragma omp barrier"),
                encoding="utf-8",
            )


def _generate_cmake_lists(target_dir: Path, algo_name: str, algo_config: dict) -> None:
    binary_file = algo_config.get("binary_file", algo_name)
    run_file = f"run/run_{binary_file}.cpp"
    cmake_content = f"""\
cmake_minimum_required(VERSION 3.10)
project({algo_name} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenMP REQUIRED)

add_executable({binary_file}
    src/util.cpp src/graph.cpp src/gsum.cpp src/pgsum.cpp {run_file}
)

target_include_directories({binary_file} PRIVATE src src/parallel_hashmap)
target_link_libraries({binary_file} PRIVATE OpenMP::OpenMP_CXX)

if(MSVC)
    target_compile_options({binary_file} PRIVATE /W0 /openmp:llvm /O2)
else()
    target_compile_options({binary_file} PRIVATE -w -O3)
endif()
"""
    (target_dir / "CMakeLists.txt").write_text(cmake_content, encoding="utf-8")


def _move_compiled_binary(build_dir: Path, binary_path: str) -> None:
    binary_name = os.path.basename(binary_path)
    compiled_unix = build_dir / binary_name
    compiled_win = build_dir / "Release" / binary_name
    if compiled_win.exists():
        shutil.move(str(compiled_win), binary_path)
    elif compiled_unix.exists():
        shutil.move(str(compiled_unix), binary_path)
    else:
        raise FileNotFoundError(f"Binary not found at {compiled_unix} or {compiled_win}")


def build_mosso(target_dir: Path, binary_path: str, fastutil_path: str, is_local: bool) -> None:
    """Compile Mosso via bash compile.sh; rename output jar to binary_path."""
    if not is_local:
        shutil.copy(fastutil_path, os.path.join(target_dir, os.path.basename(fastutil_path)))
    _run_build(["bash", "compile.sh"], cwd=target_dir)
    compiled_jar = os.path.join(target_dir, "mosso-1.0.jar")
    if os.path.exists(compiled_jar) and os.path.abspath(compiled_jar) != os.path.abspath(binary_path):
        shutil.move(compiled_jar, binary_path)


def build_mags(target_dir: Path, binary_path: str, algo_name: str, algo_config: dict) -> None:
    """CMake build for MAGS: hotfix source, generate CMakeLists, compile, move binary."""
    _apply_source_hotfixes(target_dir)
    _generate_cmake_lists(target_dir, algo_name, algo_config)
    build_dir = target_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    env = _mags_build_env()
    _run_build(["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"], cwd=build_dir, env=env)
    _run_build(["cmake", "--build", ".", "--config", "Release"], cwd=build_dir, env=env)
    _move_compiled_binary(build_dir, binary_path)


# ---------------------------------------------------------------------------
# Abstract base runner
# ---------------------------------------------------------------------------

class AlgorithmRunner(ABC):
    edge_format_string = "{u}\t{v}\n"

    def __init__(self, algo_name: str, config: dict, logger, session_dir: str):
        self.algo_name = algo_name
        self.config = config
        self.logger = logger
        self.session_dir = session_dir
        self.target_dir = Path(self.config.get("target_dir", os.path.join(VERSIONS_DIR, algo_name)))
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
                    f"    -> [{self.algo_name}] Source: {repo_url.split('/')[-1]} (Branch: {branch})"
                )
                retrieve_github_code(self.target_dir, self.algo_name, repo_url, branch, self.logger)

            self.logger.debug(f"    -> [{self.algo_name}] Compiling binaries...")
            self.compile_logic()
            self.logger.debug(f"    -> [{self.algo_name}] [OK] Build successful.")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"    -> [{self.algo_name}] [!] Compilation failed. Code {e.returncode}")
            if e.stdout:
                self.logger.debug(f"STDOUT:\n{e.stdout.strip()}")
            if e.stderr:
                self.logger.debug(f"STDERR:\n{e.stderr.strip()}")
            if not self.is_local and os.path.exists(self.target_dir):
                shutil.rmtree(self.target_dir)
            raise RuntimeError(f"Compilation failed for {self.algo_name}.") from e

    @abstractmethod
    def build_command(
        self, dataset_path: str, graph_output_path: str, parameters: dict, template: list
    ) -> list[str]:
        pass

    @abstractmethod
    def parse_output(self, stdout: str) -> tuple[Optional[float], Optional[float]]:
        pass

    def run_single(
        self,
        format_dataset_path: str,
        output_name: str,
        parameters: dict,
        template: list,
        keep_summaries: bool = False,
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Execute one trial. Returns (time, ratio, peak_rss_mb)."""
        summary_dir = os.path.join(self.session_dir, "summarized_graphs")
        runs_dir = Path(self.session_dir) / "runs"
        graph_output_path = os.path.join(summary_dir, output_name)

        cmd = self.build_command(format_dataset_path, graph_output_path, parameters, template)
        self.logger.debug(f"-> [{self.algo_name}] Running command: {' '.join(cmd)}")

        try:
            rss_before = _peak_rss_mb()
            result = self._run_cmd(cmd)
            peak_rss = max(0.0, _peak_rss_mb() - rss_before)

            parsed_time, parsed_ratio = self.parse_output(result.stdout)

            run_log_file = runs_dir / f"{output_name}.log"
            with open(run_log_file, "w", encoding="utf-8") as f:
                f.write(f"EXECUTION COMMAND:\n{' '.join(cmd)}\n\n")
                f.write("=" * 20 + " STDOUT " + "=" * 20 + "\n")
                f.write(result.stdout)
                f.write("\n" + "=" * 20 + " STDERR " + "=" * 20 + "\n")
                f.write(result.stderr)

            if not keep_summaries:
                for fname in os.listdir(summary_dir):
                    if fname.startswith(output_name):
                        try:
                            os.remove(os.path.join(summary_dir, fname))
                        except OSError as e:
                            self.logger.debug(f"[!] Cleanup failed for {fname}: {e}")

            return parsed_time, parsed_ratio, peak_rss

        except subprocess.CalledProcessError as e:
            self.logger.error(f"[!] Execution crashed for {output_name}: {e}")
            self.logger.debug(f"[!] Command was: {' '.join(cmd)}")
            self.logger.debug(f"[!] Error Output:\n{e.stderr}")
            return None, None, None

        except Exception as e:
            self.logger.error(f"[!] Unexpected error for {output_name}: {e}")
            return None, None, None

    def format_dataset(self, original_dataset_path: str) -> str:
        """Return path to cleaned dataset, creating it on first call (cached)."""
        basename = os.path.basename(original_dataset_path)
        base_dir = os.path.dirname(original_dataset_path)
        save_dir = os.path.join(base_dir, self.__class__.__name__)
        os.makedirs(save_dir, exist_ok=True)
        formatted_path = os.path.join(save_dir, f"{self.__class__.__name__}_{basename}")
        if os.path.exists(formatted_path):
            return formatted_path
        self.logger.info(f"\t[*] Cleaning dataset for {self.__class__.__name__}: {basename}")
        clean_and_write(original_dataset_path, formatted_path, self.edge_format_string)
        return formatted_path

    def _warmup(self, format_dataset_path: str, base_output_name: str,
                parameters: dict, template: list) -> None:
        self.logger.debug(f"\t[*] Executing Warmup Run for {self.algo_name}...")
        self.run_single(
            format_dataset_path, f"{base_output_name}_warmup",
            parameters, template, keep_summaries=False,
        )

    def _iterate_trials(
        self, format_dataset_path: str, base_output_name: str, runs: int,
        parameters: dict, template: list, keep_summaries: bool,
    ) -> tuple[list, list, list]:
        times, ratios, rsss = [], [], []
        for i in range(runs):
            self.logger.debug(f"Iter {i + 1}/{runs} for {base_output_name}...")
            t, r, m = self.run_single(
                format_dataset_path, f"{base_output_name}_run{i + 1}",
                parameters, template, keep_summaries,
            )
            if t is not None and r is not None:
                times.append(t)
                ratios.append(r)
            if m is not None:
                rsss.append(m)
        return times, ratios, rsss

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
        parameters: dict,
        template: list,
        keep_summaries: bool = False,
    ) -> tuple[Optional[float], Optional[float], list, list, list]:
        """Execute multiple trials. Returns (avg_time, avg_ratio, times, ratios, peak_rsss)."""
        format_dataset_path = self.format_dataset(dataset_path)
        if runs > 1:
            self._warmup(format_dataset_path, base_output_name, parameters, template)
        times, ratios, rsss = self._iterate_trials(
            format_dataset_path, base_output_name, runs, parameters, template, keep_summaries
        )
        avg_t, avg_r = self._aggregate(times, ratios)
        return avg_t, avg_r, times, ratios, rsss


# ---------------------------------------------------------------------------
# Mosso runner
# ---------------------------------------------------------------------------

class MossoRunner(AlgorithmRunner):
    edge_format_string = "{u}\t{v}\t1\n"

    _TIME_REGEX = re.compile(r"Execution time:\s*([\d.]+)s", re.IGNORECASE)
    _RATIO_REGEX = re.compile(r"Expected Compression Ratio:\s*([\d.]+)", re.IGNORECASE)

    def __init__(self, algo_name: str, config: dict, logger, session_dir: str):
        super().__init__(algo_name, config, logger, session_dir)
        self.fastutil_path = get_fastutil_path()

    def get_binary_path(self) -> str:
        binary_file = self.config.get("binary_file", f"mosso-{self.algo_name}.jar")
        return os.path.join(self.target_dir, binary_file)

    def compile_logic(self) -> None:
        build_mosso(self.target_dir, self.get_binary_path(), self.fastutil_path, self.is_local)

    def build_command(
        self, dataset_path: str, graph_output_path: str, parameters: dict, template: list
    ) -> list[str]:
        classpath = f"{self.fastutil_path}{os.pathsep}{self.get_binary_path()}"
        java_path = graph_output_path
        if java_path.startswith("output/") or java_path.startswith("output\\"):
            java_path = java_path[7:]
        cmd = ["java", "-cp", classpath, "mosso.Run", dataset_path, java_path, "mosso"]
        for param_key in template:
            cmd.append(str(parameters.get(param_key, "")))
        return cmd

    def parse_output(self, stdout: str) -> tuple[Optional[float], Optional[float]]:
        time_m = self._TIME_REGEX.search(stdout)
        ratio_m = self._RATIO_REGEX.search(stdout)
        if not time_m:
            self.logger.warning(
                f"[!] [{self.algo_name}] Could not parse execution time from output: {stdout[:200]!r}"
            )
        if not ratio_m:
            self.logger.warning(
                f"[!] [{self.algo_name}] Could not parse compression ratio from output: {stdout[:200]!r}"
            )
        return (
            float(time_m.group(1)) if time_m else None,
            float(ratio_m.group(1)) if ratio_m else None,
        )


# ---------------------------------------------------------------------------
# MAGS runner
# ---------------------------------------------------------------------------

class MagsRunner(AlgorithmRunner):
    _READ_REGEX = re.compile(r"read:\s*([\d.]+)\(s\)", re.IGNORECASE)
    _MERGE_REGEX = re.compile(r"merge:\s*([\d.]+)\(s\)", re.IGNORECASE)
    _ENCODING_REGEX = re.compile(r"encoding:\s*([\d.]+)\(s\)", re.IGNORECASE)
    _RATIO_REGEX = re.compile(r"relative size:\s*\d+/\d+\s*=\s*([\d.]+)", re.IGNORECASE)

    def get_binary_path(self) -> str:
        binary_file = self.config.get("binary_file", self.algo_name)
        if platform.system() == "Windows" and not binary_file.endswith(".exe"):
            binary_file += ".exe"
        return os.path.join(self.target_dir, binary_file)

    def compile_logic(self) -> None:
        build_mags(self.target_dir, self.get_binary_path(), self.algo_name, self.config)

    def parse_output(self, stdout: str) -> tuple[Optional[float], Optional[float]]:
        read = self._READ_REGEX.search(stdout)
        merge = self._MERGE_REGEX.search(stdout)
        encoding = self._ENCODING_REGEX.search(stdout)
        ratio = self._RATIO_REGEX.search(stdout)

        time_val: Optional[float] = None
        if read and merge and encoding:
            time_val = float(read.group(1)) + float(merge.group(1)) + float(encoding.group(1))
        else:
            self.logger.warning(
                f"[!] [{self.algo_name}] Could not parse execution time components from output: {stdout[:200]!r}"
            )

        ratio_val: Optional[float] = None
        if ratio:
            ratio_val = float(ratio.group(1))
        else:
            self.logger.warning(
                f"[!] [{self.algo_name}] Could not parse compression ratio from output: {stdout[:200]!r}"
            )

        return time_val, ratio_val

    def build_command(
        self, dataset_path: str, graph_output_path: str, parameters: dict, template: list
    ) -> list[str]:
        cmd = [os.path.abspath(self.get_binary_path()), dataset_path]
        for param_key in template:
            cmd.append(str(parameters.get(param_key, "")))
        return cmd


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_runner(algo_name: str, logger, session_dir: str) -> AlgorithmRunner:
    config = ALGORITHMS.get(algo_name)
    if not config:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    algo_type = config.get("type")
    if algo_type == "mosso":
        return MossoRunner(algo_name, config, logger, session_dir)
    if algo_type == "mags":
        return MagsRunner(algo_name, config, logger, session_dir)
    raise ValueError(f"Unknown algorithm type: {algo_type} for {algo_name}")
