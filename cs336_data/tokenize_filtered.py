#!/usr/bin/env python
import argparse
import gzip
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import submitit
from tqdm import tqdm
from transformers import AutoTokenizer

tokenizer = None

def _init_tokenizer():
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize_example(text: str):
    if not text.strip():
        return []
    ids = tokenizer.encode(text)
    ids.append(tokenizer.eos_token_id)
    return ids

def process_wet(path: Path):
    _init_tokenizer()
    with gzip.open(path, "rt") as f:
        examples = f.read().split("<|endoftext|>")
    tokens = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as tp:
        for tks in tp.map(tokenize_example, examples):
            tokens.extend(tks)
    return tokens

def process_chunk(chunk_files: list[str], out_bin: str):
    _init_tokenizer()
    all_tokens = []
    for fp in chunk_files:
        all_tokens.extend(process_wet(Path(fp)))
    np.asarray(all_tokens, dtype=np.uint16).tofile(out_bin)
    return len(all_tokens)

def chunkify(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wet_files = sorted(input_dir.glob("CC-*.warc.wet.gz"))
    if not wet_files:
        return

    executor = submitit.AutoExecutor(folder=output_dir / "logs")
    executor.update_parameters(
        slurm_array_parallelism=16,
        timeout_min=30,
        mem_gb=10,
        cpus_per_task=1,
        slurm_partition="a4-cpu",
        slurm_qos="a4-cpu-qos",
        slurm_account="student",
        name="tokenise_chunk",
    )

    chunk_size = 10
    chunks = list(chunkify(wet_files, chunk_size))
    jobs = executor.map_array(
        process_chunk,
        [[str(f) for f in c] for c in chunks],
        [str(output_dir / f"chunk_{i:04d}.bin") for i in range(len(chunks))],
    )

    from time import sleep
    bar = tqdm(total=len(jobs), desc="chunks")
    pending = set(jobs)

    while pending:
        done_now = [j for j in pending if j.done()]
        for j in done_now:
            bar.update(1)
            pending.remove(j)
        sleep(10)

    totals = [j.result() for j in jobs]
    print(f"tokens total: {sum(totals):,}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/data/c-aalag/filtered_cc3")
    parser.add_argument("--output_dir", default="/data/c-aalag/tokenized_cc")
    args = parser.parse_args()
    main(args)