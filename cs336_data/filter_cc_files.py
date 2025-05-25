from classify_data import GopherQualityClassifier, LanguageClassifier, NSFWClassifier, ToxicClassifier
from deduplication import minhash_deduplication
from parse_html import warc_text_iterator
import argparse
from collections import defaultdict
import concurrent.futures
import os
from tqdm import tqdm
import pathlib
import random
from pathlib import Path
import gzip

# sample perplexity from paloma and use as a filter?
def sample_paloma_perplexity(text):
    pass

# taken from example code
def process_warc_files(input_path, output_dir, rejected_output_dir):
    wet_paths = list(Path(input_path).glob("CC-*.warc.wet.gz"))
    num_cpus = len(os.sched_getaffinity(0))
    print(f"Using {num_cpus} CPUs")
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(rejected_output_dir, exist_ok=True)

    wet_paths = wet_paths[:10]

    futures = []
    for wet_filepath in wet_paths:
        # For each warc.wet.gz filepath, submit a job to the executor and get a future back
        wet_filename = str(pathlib.Path(wet_filepath).name)
        future = executor.submit(
            filter_warc_file,
            wet_filepath,
            os.path.join(output_dir, wet_filename),
            os.path.join(rejected_output_dir, wet_filename)
        )
        futures.append(future)

    aggregate_num_after_x = defaultdict(int)
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(wet_paths)):
        try:
            num_after_x = future.result(timeout=60)  # 1 minute timeout per result
            for key, value in num_after_x.items():
                aggregate_num_after_x[key] += value
        except concurrent.futures.TimeoutError:
            print(f"Timeout processing file: {futures[future]}")
        except Exception as e:
            print(f"Error processing file {futures[future]}: {e}")

    return aggregate_num_after_x

def filter_warc_file(warc_file_path, output_file_path, rejected_file_path):
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(rejected_file_path), exist_ok=True)

    print(f"Filtering {warc_file_path}")
    with gzip.open(warc_file_path, "rb") as in_file, \
         gzip.open(output_file_path, "wt") as f_out, \
         open(rejected_file_path, "w") as f_rejected:

        def maybe_write_to_rejected(sample):
            if random.random() < 0.001:
                f_rejected.write(sample + "\n")

        num_after_x = defaultdict(int)
        for sample in warc_text_iterator(in_file):
            num_after_x["total"] += 1
            language, language_score = LanguageClassifier().classify(sample)
            if language != "en" or language_score < 0.9:
                maybe_write_to_rejected(sample)
                continue

            num_after_x["language"] += 1

            nsfw_label, nsfw_score = NSFWClassifier().classify(sample)
            if nsfw_label == "nsfw" or nsfw_score < 0.95:
                maybe_write_to_rejected(sample)
                continue

            num_after_x["nsfw"] += 1

            toxic_label, toxic_score = ToxicClassifier().classify(sample)
            if toxic_label == "toxic" or toxic_score < 0.95:
                maybe_write_to_rejected(sample)
                continue

            num_after_x["toxic"] += 1

            should_keep = GopherQualityClassifier().classify(sample)
            if not should_keep:
                maybe_write_to_rejected(sample)
                continue

            num_after_x["quality"] += 1

            f_out.write(sample + "\n")

        return num_after_x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", type=str, default="/data/CC")
    parser.add_argument("--output_dir", type=str, default="/data/c-aalag/filtered_cc")
    parser.add_argument("--rejected_output_dir", type=str, default="/data/c-aalag/rejected_cc")
    args = parser.parse_args()

    process_warc_files(args.input_files, args.output_dir, args.rejected_output_dir)