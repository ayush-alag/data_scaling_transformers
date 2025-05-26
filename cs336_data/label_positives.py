import argparse
from cs336_data.parse_html import warc_text_iterator
from pathlib import Path
from submitit import AutoExecutor
import json
import os
import gzip
import multiprocessing

def process_positive_file(args):
    warc_file, out_path = args
    len_threshold = 30
    with gzip.open(warc_file, 'rb') as unzipped_warc, open(out_path, 'w') as out_file:
        for positive_text in warc_text_iterator(unzipped_warc):
            if len(positive_text) > len_threshold:
                out_file.write(json.dumps({"label": "__label__positive", "text": positive_text}) + "\n")

    return out_path

def process_all_positives(warc_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    warcs = sorted(Path(warc_folder).glob("chunk_*.warc.gz"))
    jobs = []
    for i, warc in enumerate(warcs):
        out_file = Path(output_folder)/f"pos_{i}.jsonl"
        jobs.append((warc, out_file))

    with multiprocessing.Pool(16) as pool:
        for out in pool.imap_unordered(process_positive_file, jobs):
            print(f"done with {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--warc_folder', type=str, default='/data/c-aalag/sampled_url_data')
    parser.add_argument('--out_file', type=str, default='/data/c-aalag/processed_classifier_data/positives')
    args = parser.parse_args()
    process_all_positives(args.warc_folder, args.out_file)