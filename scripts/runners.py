import os
import subprocess
import shutil
import re
import platform
from abc import ABC, abstractmethod

from config import VERSIONS_DIR
from utils import get_fastutil_path, retrieve_github_code

class AlgorithmRunner(ABC):
    """Autonomous execution and compilation strategy for algorithms."""
    def __init__(self, algo_name, config, logger, runs_dir):
        self.algo_name = algo_name
        self.config = config
        self.logger = logger
        self.runs_dir = runs_dir

        self.target_dir = self.config.get("target_dir", os.path.join(VERSIONS_DIR, algo_name))
        self.is_local = self.target_dir == "."

    @abstractmethod
    def get_binary_path(self):
        """Hook for subclasses to resolve their exact binary file name."""
        pass

    @abstractmethod
    def compile_logic(self):
        """Hook for subclasses to define specific build steps (CMake vs Bash)."""
        pass

    def binary_exists(self):
        return os.path.exists(self.get_binary_path())

    def build(self):
        """Template method for building the algorithm."""
        try:
            if not self.is_local:
                repo_url = str(self.config['repo'])
                branch = str(self.config['branch'])
                self.logger.info(f"\t(Repo: {repo_url.split('/')[-1]} | Branch: {branch})")
                retrieve_github_code(self.target_dir, self.algo_name, repo_url, branch, self.logger)
            self.compile_logic()
            self.logger.info(f"\t\t[OK] Successfully built {os.path.basename(self.get_binary_path())}")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"\t\t[!] Failed to build {self.algo_name}. Code {e.returncode}")
            if e.stdout: self.logger.debug(f"STDOUT:\n{e.stdout.strip()}")
            if e.stderr: self.logger.debug(f"STDERR:\n{e.stderr.strip()}")
            if not self.is_local and os.path.exists(self.target_dir):
                shutil.rmtree(self.target_dir) # Clean up broken clones
            raise RuntimeError(f"Compilation failed for {self.algo_name}.") from e

    @abstractmethod
    def build_command(self, dataset_path, output_name, parameters, template):
        pass

    def run_single(self, dataset_path, output_name, parameters, template):
        """Executes a single run internally resolving the correct binary path."""
        # 1. The exact path where the algorithm should save its graph
        graph_output_path = os.path.join(self.runs_dir, output_name)

        # 2. The exact path where Python will save the console logs
        log_file_path = f"{graph_output_path}.log"

        cmd = self.build_command(dataset_path, graph_output_path, parameters, template)
        self.logger.debug(f"Running: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            output_lines = []

            with open(log_file_path, 'w') as log_f:
                for line in process.stdout:
                    log_f.write(line)
                    output_lines.append(line)

            process.wait()
            if process.returncode != 0:
                return None, None

            output = "".join(output_lines)
            time_m = re.search(r"Execution time:\s*([\d.]+)s", output, re.IGNORECASE)
            ratio_m = re.search(r"Expected Compression Ratio:\s*([\d.]+)", output, re.IGNORECASE)

            return float(time_m.group(1)) if time_m else None, float(ratio_m.group(1)) if ratio_m else None

        except Exception as e:
            self.logger.error(f"Execution failed for {output_name}: {e}")
            return None, None

    def run_multiple(self, dataset_path, base_output_name, runs, parameters, template):
        times, ratios = [], []
        for i in range(runs):
            self.logger.debug(f"Iter {i+1}/{runs} for {base_output_name}...")
            t, r = self.run_single(dataset_path, f"{base_output_name}_run{i+1}", parameters, template)
            if t is not None and r is not None:
                times.append(t)
                ratios.append(r)

        return (sum(times)/len(times) if times else None), (sum(ratios)/len(ratios) if ratios else None), times, ratios

class MoSSoRunner(AlgorithmRunner):
    def __init__(self, algo_name, config, logger, runs_dir):
        super().__init__(algo_name, config, logger, runs_dir)
        self.fastutil_path = get_fastutil_path()

    def get_binary_path(self):
        default_name = f"mosso-{self.algo_name}.jar"
        binary_file = self.config.get('binary_file', default_name)
        return os.path.join(self.target_dir, binary_file)

    def compile_logic(self):
        if not self.is_local:
            shutil.copy(self.fastutil_path, os.path.join(self.target_dir, os.path.basename(self.fastutil_path)))
        subprocess.run(["bash", "compile.sh"], cwd=self.target_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        shutil.move(os.path.join(self.target_dir, "mosso-1.0.jar"), self.get_binary_path())

    def build_command(self, dataset_path, output_name, parameters, template):
        classpath = f"{self.fastutil_path}{os.pathsep}{self.get_binary_path()}"
        cmd = ["java", "-cp", classpath, "mosso.Run", dataset_path, output_name, "mosso"]
        for param_key in template:
            cmd.append(str(parameters.get(param_key, "")))
        return cmd

class MagsRunner(AlgorithmRunner):
    def get_binary_path(self):
        binary_file = self.config.get('binary_file', self.algo_name)
        return os.path.join(self.target_dir, binary_file)

    def compile_logic(self):
        build_dir = os.path.join(self.target_dir, "build")
        os.makedirs(build_dir, exist_ok=True)
        env = os.environ.copy()

        if platform.system() == "Darwin":
            if os.path.exists("/opt/homebrew/opt/llvm/bin/clang++"): brew_prefix = "/opt/homebrew"
            elif os.path.exists("/usr/local/opt/llvm/bin/clang++"): brew_prefix = "/usr/local"
            else: brew_prefix = None
            if brew_prefix:
                env["CC"], env["CXX"] = f"{brew_prefix}/opt/llvm/bin/clang", f"{brew_prefix}/opt/llvm/bin/clang++"
                env["LDFLAGS"], env["CPPFLAGS"] = f"-L{brew_prefix}/opt/llvm/lib -L{brew_prefix}/opt/libomp/lib", f"-I{brew_prefix}/opt/llvm/include -I{brew_prefix}/opt/libomp/include"

        pgsum_path = os.path.join(self.target_dir, "src", "pgsum.cpp")
        if os.path.exists(pgsum_path):
            with open(pgsum_path, "r", encoding="utf-8") as f: content = f.read()
            if "#pragma omp barier" in content:
                with open(pgsum_path, "w", encoding="utf-8") as f: f.write(content.replace("#pragma omp barier", "#pragma omp barrier"))

        subprocess.run(["cmake", "..", "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"], cwd=build_dir, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        subprocess.run(["make"], cwd=build_dir, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        compiled_binary = os.path.join(build_dir, "mags")
        if os.path.exists(compiled_binary):
            shutil.move(compiled_binary, self.get_binary_path())
        else:
            raise FileNotFoundError(f"Expected compiled binary not found at {compiled_binary}")

    def build_command(self, dataset_path, output_name, parameters, template):
        executable = self.get_binary_path()
        if not executable.startswith("/") and not executable.startswith("./"):
            executable = f"./{executable}"
        cmd = [executable, dataset_path, output_name]
        for param_key in template:
            cmd.append(str(parameters.get(param_key, "")))
        return cmd

def get_runner(algo_name, config, logger, runs_dir):
    """Instantiates the correct runner based on the config's language tag."""
    if config.get("lang", "java") == "cpp":
        return MagsRunner(algo_name, config, logger, runs_dir)
    elif config.get("lang", "java") == "java":
        return MoSSoRunner(algo_name, config, logger, runs_dir)
    return None