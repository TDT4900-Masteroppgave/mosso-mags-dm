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


def _generate_mags_cmake_lists(target_dir: Path, algo_name: str, algo_config: dict) -> None:
    binary_file = algo_config.get("binary_file", algo_name)
    run_file = f"run/run_{binary_file}.cpp"
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

class MagsRunner(AlgorithmRunner):
    _READ_REGEX = re.compile(r"read:\s*([\d.]+)\(s\)", re.IGNORECASE)
    _MERGE_REGEX = re.compile(r"merge:\s*([\d.]+)\(s\)", re.IGNORECASE)
    _ENCODING_REGEX = re.compile(r"encoding:\s*([\d.]+)\(s\)", re.IGNORECASE)
    _RATIO_REGEX = re.compile(r"relative size:\s*\d+/\d+[\s=]+([\d.]+)", re.IGNORECASE)

    def get_binary_path(self) -> str:
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

        cmake_cmd = [
            "cmake", "..",
            "-DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang",
            "-DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++",
            "-DOpenMP_ROOT=/opt/homebrew/opt/libomp",
        ]

        self.run_build(cmake_cmd, cwd=build_dir, env=env)
        self.run_build(["cmake", "--build", "."], cwd=build_dir, env=env)
        _move_compiled_binary(build_dir, self.get_binary_path())

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

    def build_command(self, dataset_path: str, graph_output_path: str, parameters: list[str]) -> list[str]:
        cmd = [os.path.abspath(self.get_binary_path()), dataset_path]
        cmd.extend(parameters)
        return cmd