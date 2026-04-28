import os
import re
import shutil
from typing import Optional

from scripts.utils import get_fastutil_path
from scripts.runners.base_runner import AlgorithmRunner

class MossoRunner(AlgorithmRunner):
    edge_format_string = "{u}\t{v}\t1\n"

    _TIME_REGEX = re.compile(r"Execution time:\s*([\d.]+)s", re.IGNORECASE)
    _RATIO_REGEX = re.compile(r"Expected Compression Ratio:\s*([\d.]+)", re.IGNORECASE)

    def __init__(self, algo_name: str, config: dict, logger, session_dir: str):
        super().__init__(algo_name, config, logger, session_dir)
        self.fastutil_path = get_fastutil_path()

    def get_binary_path(self) -> str:
        binary_file = self.config.get("binary_file", f"mosso-{self.algo_name}.jar")
        return self.target_dir / binary_file

    def compile_logic(self) -> None:
        if not self.is_local:
            shutil.copy(self.fastutil_path, os.path.join(self.target_dir, os.path.basename(self.fastutil_path)))
        self.run_build(["bash", "compile.sh"], cwd=self.target_dir)
        compiled_jar = os.path.join(self.target_dir, "mosso-1.0.jar")
        if os.path.exists(compiled_jar) and os.path.abspath(compiled_jar) != os.path.abspath(self.get_binary_path()):
            shutil.move(compiled_jar, self.get_binary_path())

    def build_command(self, dataset_path: str, graph_output_path: str, parameters: list[str]) -> list[str]:
        classpath = f"{self.fastutil_path}{os.pathsep}{self.get_binary_path()}"
        java_path = graph_output_path
        if java_path.startswith("output/") or java_path.startswith("output\\"):
            java_path = java_path[7:]
        cmd = ["java", "-cp", classpath, "mosso.Run", dataset_path, java_path, "mosso"]
        for param in parameters:
            cmd.append(param)
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

