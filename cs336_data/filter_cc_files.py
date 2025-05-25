from classify_data import gopher_quality_filter, classify_language, classify_nsfw, classify_toxic
from deduplication import minhash_deduplication
from parse_html import warc_text_iterator
import argparse
from collections import defaultdict
import concurrent.futures
import os
from tqdm import tqdm
import pathlib

# sample perplexity from paloma
def sample_paloma_perplexity(text):
    # text is a string
    # return a float
    # use the paloma api to sample perplexity
    # return the perplexity
    pass

# taken from example code
def process_warc_files(input_path, output_path):
    num_cpus = len(os.sched_getaffinity(0))
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus)

    futures = []
    for wet_filepath in input_path:
        # For each warc.wet.gz filepath, submit a job to the executor and get a future back
        wet_filename = str(pathlib.Path(wet_filepath).name)
        future = executor.submit(
            filter_warc_file,
            wet_filepath,
            os.path.join(output_path, wet_filename)
        )
        futures.append(future)

    for future in tqdm(concurrent.futures.as_completed(futures), total=len(input_path)):
        output_file = future.result()
        print(f"Output file written: {output_file}")

def filter_warc_file(warc_file_path, output_file_path):
    failure_reasons = defaultdict(int)
    for sample in warc_text_iterator(warc_file_path):
        language, language_score = classify_language(sample)
        if language != "en" or language_score < 0.95:
            failure_reasons["language"] += 1
            continue

        should_keep = gopher_quality_filter(sample)
        if not should_keep:
            failure_reasons["quality"] += 1
            continue

        nsfw_label, nsfw_score = classify_nsfw(sample)
        if nsfw_label == "nsfw" or nsfw_score < 0.95:
            failure_reasons["nsfw"] += 1
            continue

        toxic_label, toxic_score = classify_toxic(sample)
        if toxic_label == "toxic" or toxic_score < 0.95:
            failure_reasons["toxic"] += 1
            continue

        with open(output_file_path, "a") as f:
            f.write(sample)

    return failure_reasons

def filter_cc_files(input_files, output_dir, num_hashes, num_bands, ngrams, jaccard_threshold):
    minhash_deduplication(input_files, num_hashes, num_bands, ngrams, jaccard_threshold, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)