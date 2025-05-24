import argparse
from cs336_data.parse_html import warc_text_iterator
from pathlib import Path
from submitit import AutoExecutor
import json
import os
import gzip
import multiprocessing

def process_negative_file(args):
    warc_file, out_path = args
    NUM_NEGATIVES_PER_WARC = 100
    with gzip.open(warc_file, 'rb') as unzipped_warc, open(out_path, 'w') as out_file:
        num_negatives = 0
        for negative_text in warc_text_iterator(unzipped_warc):
            out_file.write(json.dumps({"label": "__label__negative", "text": negative_text}) + "\n")
            num_negatives += 1
            if num_negatives >= NUM_NEGATIVES_PER_WARC:
                break

    return out_path

def process_all_negatives(warc_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    # executor = AutoExecutor(folder="/data/c-aalag/executor_logs")
    # executor.update_parameters(
    #     timeout_min=30, cpus_per_task=1, mem_gb=100,
    #     slurm_partition="a4-cpu", slurm_qos="a4-cpu-qos",
    #     slurm_python="/usr/bin/python3"
    # )

    warcs = sorted(Path(warc_folder).glob("CC-MAIN-*.warc.wet.gz"))
    jobs = []

    # let's process 100 warc files
    num_warcs = 100
    for i, warc in enumerate(warcs):
        if i >= num_warcs:
            break

        out_file = Path(output_folder)/f"neg_{i}.jsonl"
        jobs.append((warc, out_file))

    with multiprocessing.Pool(16) as pool:
        for out in pool.imap_unordered(process_negative_file, jobs):
            print(f"Completed: {out}")

    # results = []
    # for job in jobs:
    #     try:
    #         result = job.result()
    #         results.append(result)
    #         print(f"Completed: {result}")
    #     except Exception as e:
    #         print(f"job failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wet_folder', type=str, default='/data/CC')
    parser.add_argument('--out_file', type=str, default='/data/c-aalag/processed_classifier_data/negatives')
    args = parser.parse_args()
    process_all_negatives(args.wet_folder, args.out_file)