from cs336_data.classify_data import GopherQualityClassifier, LanguageClassifier, NSFWClassifier, ToxicClassifier, QualityClassifier
from cs336_data.parse_html import warc_text_iterator
import argparse
from collections import Counter
import concurrent.futures
import os
from tqdm import tqdm
import pathlib
import random
from pathlib import Path
import gzip
import os, signal
import submitit
from more_itertools import chunked
import time
import sys

def worker_batch(batch_paths, output_dir, rejected_output_dir):
    batch_stats = Counter()

    language_filter = LanguageClassifier()
    nsfw_filter = NSFWClassifier()
    toxic_filter = ToxicClassifier()
    gopher_quality_filter = GopherQualityClassifier()
    quality_filter = QualityClassifier()

    for path in batch_paths:
        per_file_stats = filter_warc_file(
            path,
            os.path.join(output_dir, path.name),
            os.path.join(rejected_output_dir, path.name),
            language_filter,
            nsfw_filter,
            toxic_filter,
            gopher_quality_filter,
            quality_filter
        )

        batch_stats.update(per_file_stats)
    return batch_stats

# taken from example code
def process_warc_files(input_path, output_dir, rejected_output_dir):
    executor = submitit.AutoExecutor(folder="slurm_logs")
    max_simultaneous_jobs = 16
    shards_per_job = 10
    executor.update_parameters(
        slurm_array_parallelism=max_simultaneous_jobs,
        timeout_min=15,
        mem_gb=10,
        cpus_per_task=1,
        slurm_account="student",
        slurm_partition="a4-cpu",
        slurm_qos="a4-cpu-qos",
    )

    wet_paths = list(Path(input_path).glob("CC-*.warc.wet.gz"))
    jobs = list(chunked(wet_paths, shards_per_job))

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(rejected_output_dir, exist_ok=True)

    futures = []
    with executor.batch():
        for filepath_chunk in jobs:
            future = executor.submit(
                worker_batch,
                filepath_chunk,
                output_dir,
                rejected_output_dir
            )
            futures.append(future)

    aggregate_num_after_x = Counter()
    for future in tqdm(submitit.helpers.as_completed(futures),
                       desc=f"[{os.getpid()}] jobs",
                       unit="job",
                       position=1,
                       leave=False,
                       file=sys.stdout):
        num_after_x = future.result()
        aggregate_num_after_x.update(num_after_x)

    return aggregate_num_after_x

def filter_warc_file(warc_file_path, output_file_path, rejected_file_path, language_filter, nsfw_filter, toxic_filter, gopher_quality_filter, quality_filter):
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(rejected_file_path), exist_ok=True)

    print(f"Filtering {warc_file_path}")
    with gzip.open(warc_file_path, "rb") as in_file, \
         gzip.open(output_file_path, "wt") as f_out, \
         open(rejected_file_path, "w") as f_rejected:

        def maybe_write_to_rejected(sample):
            if random.random() < 0.001:
                f_rejected.write(sample + "\n")

        num_after_x = Counter()
        num_samples = 0
        try:
            for sample in warc_text_iterator(in_file):
                num_samples += 1
                # if num_samples % 10000 == 0:
                #     print(f"Processed {num_samples} samples")

                num_after_x["total"] += 1
                language, language_score = language_filter.classify(sample)
                if language != "en" or language_score < 0.9:
                    maybe_write_to_rejected(sample)
                    continue

                num_after_x["language"] += 1

                should_keep = gopher_quality_filter.classify(sample)
                if not should_keep:
                    maybe_write_to_rejected(sample)
                    continue

                num_after_x["quality"] += 1

                nsfw_label, nsfw_score = nsfw_filter.classify(sample)
                if nsfw_label == "nsfw" or nsfw_score < 0.98:
                    maybe_write_to_rejected(sample)
                    continue

                num_after_x["nsfw"] += 1

                toxic_label, toxic_score = toxic_filter.classify(sample)
                if toxic_label == "toxic" or toxic_score < 0.98:
                    maybe_write_to_rejected(sample)
                    continue

                num_after_x["toxic"] += 1

                # quality_label, quality_score = quality_filter.classify(sample)
                # if quality_label == "low" or quality_score < 0.7:
                #     maybe_write_to_rejected(sample)
                #     continue

                # num_after_x["quality"] += 1

                f_out.write(sample + "<|endoftext|>")

            return num_after_x
        except Exception as e:
            print(f"Error processing {warc_file_path}: {e}")
            return num_after_x

if __name__ == "__main__":
    os.setpgrp()

    def _die(signum, frame):
        os.killpg(0, signal.SIGTERM)

    signal.signal(signal.SIGINT,  _die)
    signal.signal(signal.SIGQUIT, _die)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", type=str, default="/data/CC")
    parser.add_argument("--output_dir", type=str, default="/data/c-aalag/filtered_cc3")
    parser.add_argument("--rejected_output_dir", type=str, default="/data/c-aalag/rejected_cc3")
    args = parser.parse_args()

    start_time = time.time()
    aggregate_num_after_x = process_warc_files(args.input_files, args.output_dir, args.rejected_output_dir)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(aggregate_num_after_x)