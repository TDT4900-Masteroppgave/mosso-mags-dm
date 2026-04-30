import os
import subprocess
import re
import shutil
from config import RUNS_DIR, SUMMARIZED_DIR
from utils import get_fastutil_path

def run_mosso(jar_file, dataset_path, output_name, discard_summaries, logger, parameters, template):
    classpath = f"{get_fastutil_path()}{os.pathsep}{jar_file}"
    out_file = os.path.join(RUNS_DIR, output_name)
    log_file = f"{out_file}.log"

    # Base Java command
    cmd = ["java", "-cp", classpath, "mosso.Run", dataset_path, output_name, "mosso"]

    for param_key in template:
        cmd.append(str(parameters[param_key]))

    logger.debug(f"Executing: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        output_lines = []

        with open(log_file, 'w') as log_f:
            for line in process.stdout:
                log_f.write(line)
                output_lines.append(line)

        process.wait()

        if process.returncode != 0:
            logger.error(f"[!] Java error for {output_name}: Code {process.returncode}")
            return None, None

        java_output_file = os.path.join("output", output_name)
        if os.path.exists(java_output_file):
            if discard_summaries:
                os.remove(java_output_file)
            else:
                shutil.move(java_output_file, os.path.join(SUMMARIZED_DIR, output_name))

        output = "".join(output_lines)
        time_m = re.search(r"Execution time:\s*([\d.]+)s", output)
        ratio_m = re.search(r"Expected Compression Ratio:\s*([\d.]+)", output, re.IGNORECASE)

        t = float(time_m.group(1)) if time_m else None
        r = float(ratio_m.group(1)) if ratio_m else None
        return t, r

    except Exception as e:
        logger.error(f"Execution failed for {output_name}: {e}")
        return None, None

def run_multiple_mosso(jar_file, dataset_path, output_name, runs, discard_summaries, logger, parameters, template):
    times, ratios = [], []
    for i in range(runs):
        logger.debug(f"Iter {i+1}/{runs} for {output_name}...")
        t, r = run_mosso(jar_file, dataset_path, f"{output_name}_run{i+1}", discard_summaries, logger, parameters, template)
        if t is not None and r is not None:
            times.append(t)
            ratios.append(r)

    t_avg = sum(times) / len(times) if times else None
    r_avg = sum(ratios) / len(ratios) if ratios else None
    return t_avg, r_avg, times, ratios