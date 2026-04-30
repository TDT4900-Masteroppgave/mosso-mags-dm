import os
import pandas as pd
from config import ALGORITHMS, BENCHMARK_DIR, RUNS_DIR
from utils import setup_logging, setup_directories, build_jars, download_and_prepare_dataset, prepare_dataset, \
    parse_and_filter_args, get_datasets_to_run, print_benchmark_table
from run_mosso import run_multiple_mosso
from plotter import plot_results, plot_runs_variance


def run_suite(args, datasets_to_run, logger, timestamp):
    results = []

    for i, (url, filename) in enumerate(datasets_to_run, 1):
        dataset_name = filename.replace(".txt", "").replace(".csv", "")
        path = prepare_dataset(filename, logger) if url == "local" else download_and_prepare_dataset(url, filename,
                                                                                                     logger)
        if not path: continue

        logger.info(f"[{i}/{len(datasets_to_run)}] Benchmarking [{dataset_name}] ({args.runs} runs) ...")
        current_result = {"Dataset": dataset_name}
        all_times_dict, all_ratios_dict = {}, {}

        for algo_name, algo_config in ALGORITHMS.items():
            jar_file = f"mosso-{algo_name}.jar"
            if not os.path.exists(jar_file):
                continue

            template = algo_config.get('template')
            params = algo_config.get('params', {})
            resolved_params = {
                "samples": params.get('samples', args.samples),
                "escape": params.get('escape', args.escape),
                "interval": params.get('interval', args.interval),
            }

            t_avg, r_avg, t_list, r_list = run_multiple_mosso(
                jar_file, path, f"{algo_name}_{dataset_name}_{timestamp}",
                args.runs, not args.keep_summaries, logger, resolved_params, template
            )

            if t_avg is not None:
                current_result[f"Time_{algo_name}"], current_result[f"Ratio_{algo_name}"] = t_avg, r_avg
                logger.info(f"\t=> {algo_name: <12} Time: {t_avg:.3f}s | Ratio: {r_avg:.5f}")
                if args.runs > 1:
                    all_times_dict[algo_name], all_ratios_dict[algo_name] = t_list, r_list

        results.append(current_result)
        if args.runs > 1:
            plot_runs_variance(f"{dataset_name}_{timestamp}", all_times_dict, all_ratios_dict, RUNS_DIR)

    return results


def main():
    args = parse_and_filter_args(script_type="benchmark")
    logger, timestamp = setup_logging("benchmark")

    logger.info("=" * 10 + f"{' STAGE 1: SETUP & COMPILATION ':^10}" + "=" * 10)
    setup_directories()
    build_jars(args.skip_build, args.local, logger)

    logger.info("=" * 10 + f"{' STAGE 2: PROCESSING ':^10}" + "=" * 10)
    datasets_to_run = get_datasets_to_run(args)

    results = run_suite(args, datasets_to_run, logger, timestamp)
    if results:
        print_benchmark_table(results, logger, title="BENCHMARK SUMMARY", baseline_algo=args.baseline)
        csv_file = os.path.join(BENCHMARK_DIR, f"results_{timestamp}.csv")
        pd.DataFrame(results).to_csv(csv_file, index=False)
        plot_results(csv_file, os.path.join(BENCHMARK_DIR, f"comparison_{timestamp}.pdf"), logger)
        logger.info(f"[*] Artifacts saved to: {BENCHMARK_DIR}")


if __name__ == "__main__":
    main()