import os
import re
import shutil
from pathlib import Path
from typing import Optional

from scripts.utils import get_fastutil_path
from scripts.runners.base_runner import AlgorithmRunner


class MossoRunner(AlgorithmRunner):
    edge_format_string = "{u}\t{v}\t1\n"
    _TIME_REGEX = re.compile(r"Execution time:\s*([\d.]+)s", re.IGNORECASE)
    _RATIO_REGEX = re.compile(r"Expected Compression Ratio:\s*([a-zA-Z\d.]+)", re.IGNORECASE)
    _INTERMEDIATE_REGEX = re.compile(r"(\d+)\s*:\s*Elapsed time\s*:\s*([\d.]+)\s*:\s*ratio\s*:\s*([\d.]+)", re.IGNORECASE)

    def __init__(self, algo_name: str, config: dict, logger, session_dir: str):
        super().__init__(algo_name, config, logger, session_dir)
        self.fastutil_path = Path(get_fastutil_path())

    def get_binary_path(self) -> Path:
        return self.target_dir / self.config.get("binary_file", f"mosso-{self.algo_name}.jar")

    def compile_logic(self) -> None:
        if not self.is_local:
            shutil.copy(self.fastutil_path, self.target_dir / self.fastutil_path.name)

        self._execute_cmd(["bash", "compile.sh"], cwd=self.target_dir, log_msg=f"[{self.algo_name}] Compiling via bash")

        compiled_jar = self.target_dir / "mosso-1.0.jar"
        target_jar = self.get_binary_path()
        if compiled_jar.exists() and compiled_jar.absolute() != target_jar.absolute():
            shutil.move(str(compiled_jar), str(target_jar))

    def build_command(self, dataset_path: str, graph_output_path: str, parameters: list[str]) -> list[str]:
        classpath = f"{self.fastutil_path}{os.pathsep}{self.get_binary_path()}"

        java_path = graph_output_path.replace("output/", "").replace("output\\", "")

        return ["java", "-cp", classpath, "mosso.Run", dataset_path, java_path, "mosso"] + parameters

    def parse_output(self, stdout: str) -> tuple[Optional[float], Optional[float], list[dict]]:
        time_m = self._TIME_REGEX.search(stdout)
        ratio_m = self._RATIO_REGEX.search(stdout)

        if not time_m: self.logger.warning(f"[!] [{self.algo_name}] Could not parse execution time.")
        if not ratio_m: self.logger.warning(f"[!] [{self.algo_name}] Could not parse compression ratio.")

        intermediates = []
        for match in self._INTERMEDIATE_REGEX.finditer(stdout):
            edges, t, r = match.groups()
            intermediates.append({"edges": int(edges), "time": float(t), "ratio": float(r)})

        return (
            float(time_m.group(1)) if time_m else None,
            float(ratio_m.group(1)) if ratio_m else None,
            intermediates
        )
