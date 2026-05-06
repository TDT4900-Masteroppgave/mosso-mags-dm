import os
import platform
import re
import shutil
from pathlib import Path
from typing import Optional

from scripts.runners.base_runner import AlgorithmRunner


def _mags_build_env() -> dict:
    env = os.environ.copy()
    env["CXXFLAGS"] = f"{env.get('CXXFLAGS', '')} -O3"
    if platform.system() == "Darwin":
        for prefix in ["/opt/homebrew", "/usr/local"]:
            llvm = Path(prefix) / "opt" / "llvm" / "bin"
            if (llvm / "clang++").exists():
                env["PATH"] = f"{llvm}:{env.get('PATH', '')}"
                env["CC"] = str(llvm / "clang")
                env["CXX"] = str(llvm / "clang++")
                break
    return env


def _apply_source_hotfixes(target_dir: Path) -> None:
    pgsum = target_dir / "src" / "pgsum.cpp"
    if pgsum.exists():
        content = pgsum.read_text(encoding="utf-8")
        if "#pragma omp barier" in content:
            pgsum.write_text(content.replace("#pragma omp barier", "#pragma omp barrier"), encoding="utf-8")


def _generate_mags_cmake_lists(target_dir: Path, algo_name: str, algo_config: dict) -> None:
    binary_file = algo_config.get("binary_file", algo_name)
    cmake_content = f"""\
cmake_minimum_required(VERSION 3.10)
project(core-sum)
set(CMAKE_CXX_FLAGS "${{CMAKE_CXX_FLAGS}} -std=c++14 -O3 -fopenmp")
find_package(OpenMP REQUIRED)
include_directories(./src)
aux_source_directory(./src INCLUDES)
add_library(INCLUDES_LIB ${{INCLUDES}})
add_executable(mags ./run/run_mags.cpp)
target_link_libraries(mags INCLUDES_LIB)
add_executable(mags_dm ./run/run_mags_dm.cpp)
target_link_libraries(mags_dm INCLUDES_LIB)
add_executable(pmags_dm ./run/run_para_mags_dm.cpp)
target_link_libraries(pmags_dm INCLUDES_LIB)
add_executable(pmags ./run/run_para_mags.cpp)
target_link_libraries(pmags INCLUDES_LIB)
"""
    (target_dir / "CMakeLists.txt").write_text(cmake_content, encoding="utf-8")


def _move_compiled_binary(build_dir: Path, binary_path: Path) -> None:
    compiled_win = build_dir / "Release" / binary_path.name
    compiled_unix = build_dir / binary_path.name

    if compiled_win.exists():
        shutil.move(str(compiled_win), str(binary_path))
    elif compiled_unix.exists():
        shutil.move(str(compiled_unix), str(binary_path))
    else:
        raise FileNotFoundError(f"Binary not found at {compiled_unix} or {compiled_win}")


class MagsRunner(AlgorithmRunner):
    _READ_REGEX = re.compile(r"read:\s*([\d.]+)\(s\)", re.IGNORECASE)
    _MERGE_REGEX = re.compile(r"merge:\s*([\d.]+)\(s\)", re.IGNORECASE)
    _ENCODING_REGEX = re.compile(r"encoding:\s*([\d.]+)\(s\)", re.IGNORECASE)
    _RATIO_REGEX = re.compile(r"relative size:\s*\d+/\d+[\s=]+([\d.]+)", re.IGNORECASE)

    def get_binary_path(self) -> Path:
        binary_file = self.config.get("binary_file", self.algo_name)
        if platform.system() == "Windows" and not binary_file.endswith(".exe"):
            binary_file += ".exe"
        return self.target_dir / binary_file

    def compile_logic(self) -> None:
        _apply_source_hotfixes(self.target_dir)
        _generate_mags_cmake_lists(self.target_dir, self.algo_name, self.config)

        build_dir = self.target_dir / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        env = _mags_build_env()

        cmake_cmd = ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"]
        if platform.system() == "Darwin":
            cmake_cmd.extend([
                "-DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang",
                "-DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++",
                "-DOpenMP_ROOT=/opt/homebrew/opt/libomp",
            ])

        self._execute_cmd(cmake_cmd, cwd=build_dir, env=env, log_msg=f"[{self.algo_name}] Running cmake")
        self._execute_cmd(["cmake", "--build", "."], cwd=build_dir, env=env, log_msg=f"[{self.algo_name}] Building")
        _move_compiled_binary(build_dir, self.get_binary_path())

    def parse_output(self, stdout: str) -> tuple[Optional[float], Optional[float]]:
        r_match = self._READ_REGEX.search(stdout)
        m_match = self._MERGE_REGEX.search(stdout)
        e_match = self._ENCODING_REGEX.search(stdout)
        ratio_match = self._RATIO_REGEX.search(stdout)

        time_val = float(r_match.group(1)) + float(m_match.group(1)) + float(e_match.group(1)) if (
                    r_match and m_match and e_match) else None
        ratio_val = float(ratio_match.group(1)) if ratio_match else None

        if time_val is None: self.logger.warning(
            f"[!] [{self.algo_name}] Could not parse execution time components from output.")
        if ratio_val is None: self.logger.warning(
            f"[!] [{self.algo_name}] Could not parse compression ratio from output.")

        return time_val, ratio_val

    def build_command(self, dataset_path: str, graph_output_path: str, parameters: list[str]) -> list[str]:
        return [str(self.get_binary_path().absolute()), dataset_path] + parameters
