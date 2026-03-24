import os
import subprocess
import shutil
import re
import platform
from abc import ABC, abstractmethod
from pathlib import Path

from scripts.config import ALGORITHMS, VERSIONS_DIR
from scripts.utils import retrieve_github_code, get_fastutil_path


class AlgorithmRunner(ABC):
    edge_format_string = "{u}\t{v}\n"
    _EDGE_PATTERN = re.compile(r"^\s*(\d+)\s+(\d+)")

    def __init__(self, algo_name, config, logger, session_dir):
        self.algo_name = algo_name
        self.config = config
        self.logger = logger
        self.session_dir = session_dir

        self.target_dir = Path(self.config.get("target_dir", os.path.join(VERSIONS_DIR, algo_name)))
        self.is_local = self.target_dir == "."

    @abstractmethod
    def get_binary_path(self):
        """Hook for subclasses to resolve their exact binary file name."""
        pass

    @abstractmethod
    def compile_logic(self):
        """Hook for subclasses to define specific build steps (CMake vs. Bash)."""
        pass

    def _run_cmd(self, cmd: list[str], cwd=None, env=None, timeout=None):
        """Standardized wrapper for executing shell commands."""
        try:
            return subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"[!] Command failed: {' '.join(cmd)}\nSTDERR: {e.stderr}")
            raise

    def binary_exists(self):
        return os.path.exists(self.get_binary_path())

    def build(self):
        """Template method for building the algorithm."""
        try:
            if not self.is_local:
                repo_url = str(self.config['repo'])
                branch = str(self.config['branch'])
                self.logger.info(f"    -> [{self.algo_name}] Source: {repo_url.split('/')[-1]} (Branch: {branch})")
                retrieve_github_code(self.target_dir, self.algo_name, repo_url, branch, self.logger)

            self.logger.info(f"    -> [{self.algo_name}] Compiling binaries...")
            self.compile_logic()
            self.logger.info(f"    -> [{self.algo_name}] [OK] Build successful.")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"    -> [{self.algo_name}] [!] Compilation failed. Code {e.returncode}")
            if e.stdout: self.logger.debug(f"STDOUT:\n{e.stdout.strip()}")
            if e.stderr: self.logger.debug(f"STDERR:\n{e.stderr.strip()}")
            if not self.is_local and os.path.exists(self.target_dir):
                shutil.rmtree(self.target_dir)  # Clean up broken clones
            raise RuntimeError(f"Compilation failed for {self.algo_name}.") from e

    @abstractmethod
    def build_command(self, dataset_path, graph_output_path, parameters, template):
        pass

    @abstractmethod
    def parse_output(self, stdout: str):
        pass

    def run_single(self, format_dataset_path, output_name, parameters, template, keep_summaries=False, timeout=600):
        summary_dir = os.path.join(self.session_dir, "summarized_graphs")
        runs_dir = Path(self.session_dir) / "runs"
        graph_output_path = os.path.join(summary_dir, output_name)

        cmd = self.build_command(format_dataset_path, graph_output_path, parameters, template)
        self.logger.debug(f"-> [{self.algo_name}] Running command: {' '.join(cmd)}")

        try:
            result = self._run_cmd(cmd, timeout=timeout)
            parsed_time, parsed_ratio = self.parse_output(result.stdout)

            run_log_file = runs_dir / f"{output_name}.log"
            with open(run_log_file, "w", encoding="utf-8") as f:
                f.write(f"EXECUTION COMMAND:\n{' '.join(cmd)}\n\n")
                f.write("="*20 + " STDOUT " + "="*20 + "\n")
                f.write(result.stdout)
                f.write("\n" + "="*20 + " STDERR " + "="*20 + "\n")
                f.write(result.stderr)

            if not keep_summaries:
                for f in os.listdir(summary_dir):
                    if f.startswith(output_name):
                        try:
                            os.remove(os.path.join(summary_dir, f))
                        except OSError as e:
                            self.logger.debug(f"[!] Cleanup failed for {f.name}: {e}")

            return parsed_time, parsed_ratio

        except subprocess.TimeoutExpired:
            self.logger.error(f"[!] Execution TIMED OUT for {output_name} after {timeout} seconds. (Infinite loop?)")
            self.logger.debug(f"[!] Command was: {' '.join(cmd)}")
            return None, None

        except subprocess.CalledProcessError as e:
            self.logger.error(f"[!] Execution crashed for {output_name}: {e}")
            self.logger.debug(f"[!] Command was: {' '.join(cmd)}")
            self.logger.debug(f"[!] Error Output:\n{e.stderr}")
            return None, None

        except Exception as e:
            self.logger.error(f"[!] Unexpected error for {output_name}: {e}")
            return None, None

    def format_dataset(self, original_dataset_path: str) -> str:
        """Universal dataset formatter using regex and template strings."""
        basename = os.path.basename(original_dataset_path)
        base_dir = os.path.dirname(original_dataset_path)

        save_dir = os.path.join(base_dir, self.__class__.__name__)
        os.makedirs(save_dir, exist_ok=True)
        formatted_path = os.path.join(save_dir, f"{self.__class__.__name__}_{basename}")

        # Cache check
        if os.path.exists(formatted_path):
            return formatted_path

        self.logger.info(f"\t[*] Cleaning dataset for {self.algo_name}: {basename}")
        seen_edges = set()
        with open(original_dataset_path, 'r', encoding='utf-8') as f_in, open(formatted_path, 'w',
                                                                              encoding='utf-8') as f_out:
            for line in f_in:
                if line.startswith(('#', '%')): continue

                match = self._EDGE_PATTERN.search(line)
                if match:
                    u, v = int(match.group(1)), int(match.group(2))
                    if u == v: continue  # Remove self-loops

                    edge = tuple(sorted((u, v)))
                    if edge in seen_edges: continue
                    seen_edges.add(edge)

                    f_out.write(self.edge_format_string.format(u=u, v=v))

        return formatted_path

    def run_multiple(self, dataset_path, base_output_name, runs, parameters, template, keep_summaries=False,
                     timeout=600):
        format_dataset_path = self.format_dataset(dataset_path)

        if runs > 1:
            self.logger.info(f"\t[*] Executing Warmup Run for {self.algo_name}...")
            self.run_single(format_dataset_path, f"{base_output_name}_warmup", parameters, template,
                            keep_summaries=False, timeout=timeout)

        times, ratios = [], []
        for i in range(runs):
            self.logger.debug(f"Iter {i + 1}/{runs} for {base_output_name}...")
            t, r = self.run_single(format_dataset_path, f"{base_output_name}_run{i + 1}", parameters, template,
                                   keep_summaries, timeout)
            if t is not None and r is not None:
                times.append(t)
                ratios.append(r)

        return (sum(times) / len(times) if times else None), (
            sum(ratios) / len(ratios) if ratios else None), times, ratios


class MossoRunner(AlgorithmRunner):
    edge_format_string = "{u}\t{v}\t1\n"

    _TIME_REGEX = re.compile(r"Execution time:\s*([\d.]+)s", re.IGNORECASE)
    _RATIO_REGEX = re.compile(r"Expected Compression Ratio:\s*([\d.]+)", re.IGNORECASE)

    def __init__(self, algo_name, config, logger, session_dir):
        super().__init__(algo_name, config, logger, session_dir)
        self.fastutil_path = get_fastutil_path()

    def get_binary_path(self):
        default_name = f"mosso-{self.algo_name}.jar"
        binary_file = self.config.get('binary_file', default_name)
        return os.path.join(self.target_dir, binary_file)

    def compile_logic(self):
        if not self.is_local:
            shutil.copy(self.fastutil_path, os.path.join(self.target_dir, os.path.basename(self.fastutil_path)))

        self._run_cmd(["bash", "compile.sh"], cwd=self.target_dir)

        shutil.move(os.path.join(self.target_dir, "mosso-1.0.jar"), self.get_binary_path())

    def build_command(self, dataset_path, graph_output_path, parameters, template):
        classpath = f"{self.fastutil_path}{os.pathsep}{self.get_binary_path()}"

        java_path = graph_output_path
        if java_path.startswith("output/") or java_path.startswith("output\\"):
            java_path = java_path[7:]

        cmd = ["java", "-cp", classpath, "mosso.Run", dataset_path, java_path, "mosso"]
        for param_key in template:
            cmd.append(str(parameters.get(param_key, "")))
        return cmd

    def parse_output(self, stdout: str):
        time_m = self._TIME_REGEX.search(stdout)
        ratio_m = self._RATIO_REGEX.search(stdout)
        return float(time_m.group(1)) if time_m else None, float(ratio_m.group(1)) if ratio_m else None


class MagsRunner(AlgorithmRunner):
    _READ_REGEX = re.compile(r"read:\s*([\d.]+)\(s\)", re.IGNORECASE)
    _MERGE_REGEX = re.compile(r"merge:\s*([\d.]+)\(s\)", re.IGNORECASE)
    _ENCODING_REGEX = re.compile(r"encoding:\s*([\d.]+)\(s\)", re.IGNORECASE)
    _RATIO_REGEX = re.compile(r"relative size:\s*\d+/\d+\s*=\s*([\d.]+)", re.IGNORECASE)

    def get_binary_path(self):
        binary_file = self.config.get('binary_file', self.algo_name)

        if platform.system() == "Windows" and not binary_file.endswith(".exe"):
            binary_file += ".exe"
        return os.path.join(self.target_dir, binary_file)

    def parse_output(self, stdout: str):

        read = self._READ_REGEX.search(stdout)
        merge = self._MERGE_REGEX.search(stdout)
        encoding = self._ENCODING_REGEX.search(stdout)
        ratio = self._RATIO_REGEX.search(stdout)

        time_m = None
        if read and merge and encoding:
            time_m = float(read.group(1)) + float(merge.group(1)) + float(encoding.group(1))

        ratio_m = None
        if ratio:
            ratio_m = float(ratio.group(1))

        return time_m, ratio_m

    def _apply_source_hotfixes(self):
        """Patches known upstream typos in the MAGS source code."""
        pgsum_path = os.path.join(self.target_dir, "src", "pgsum.cpp")
        if os.path.exists(pgsum_path):
            with open(pgsum_path, "r", encoding="utf-8") as f:
                content = f.read()
            if "#pragma omp barier" in content:
                with open(pgsum_path, "w", encoding="utf-8") as f:
                    f.write(content.replace("#pragma omp barier", "#pragma omp barrier"))

    def _generate_cmake_lists(self):
        """Dynamically creates the CMakeLists.txt file for the targeted algorithm."""
        binary_file = self.config.get('binary_file', self.algo_name)
        run_file = f"run/run_{binary_file}.cpp"

        cmake_content = f"""
cmake_minimum_required(VERSION 3.10)
project({self.algo_name} LANGUAGES CXX)

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
        cmake_path = self.target_dir / "CMakeLists.txt"
        with open(cmake_path, "w", encoding="utf-8") as f:
            f.write(cmake_content.strip() + "\n")

    def _move_compiled_binary(self, build_dir):
        """Locates the compiled binary across different OS builds and moves it."""
        binary_name = os.path.basename(self.get_binary_path())
        compiled_unix = os.path.join(build_dir, binary_name)
        compiled_win = os.path.join(build_dir, "Release", binary_name)

        if os.path.exists(compiled_win):
            shutil.move(compiled_win, self.get_binary_path())
        elif os.path.exists(compiled_unix):
            shutil.move(compiled_unix, self.get_binary_path())
        else:
            raise FileNotFoundError(f"Binary not found at {compiled_unix} or {compiled_win}")

    def compile_logic(self):
        """Orchestrates the CMake build process."""
        build_dir = self.target_dir / "build"
        build_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["CXXFLAGS"] = f"{env.get('CXXFLAGS', '')} -O3"

        if platform.system() == "Darwin":
            prefixes = ["/opt/homebrew", "/usr/local"]
            for prefix in prefixes:
                if os.path.exists(f"{prefix}/opt/llvm/bin/clang++"):
                    llvm_bin = f"{prefix}/opt/llvm/bin"
                    env["PATH"] = f"{llvm_bin}:{env.get('PATH', '')}"
                    env["CC"] = f"{llvm_bin}/clang"
                    env["CXX"] = f"{llvm_bin}/clang++"

        self._apply_source_hotfixes()
        self._generate_cmake_lists()

        self._run_cmd(["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"], cwd=build_dir, env=env)

        self._run_cmd(["cmake", "--build", ".", "--config", "Release"], cwd=build_dir, env=env)

        self._move_compiled_binary(build_dir)

    def build_command(self, dataset_path, graph_output_path, parameters, template):
        executable = os.path.abspath(self.get_binary_path())

        cmd = [executable, dataset_path]
        for param_key in template:
            cmd.append(str(parameters.get(param_key, "")))
        return cmd


def get_runner(algo_name: str, logger, session_dir: str) -> AlgorithmRunner:
    config = ALGORITHMS.get(algo_name)
    if not config:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    algo_type = config.get("type")
    if algo_type == "mosso":
        return MossoRunner(algo_name, config, logger, session_dir)
    elif algo_type == "mags":
        return MagsRunner(algo_name, config, logger, session_dir)
    else:
        raise ValueError(f"Unknown algorithm type: {algo_type} for {algo_name}")
