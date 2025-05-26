import argparse
import gzip
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import submitit
from tqdm import tqdm
import re

from transformers import AutoTokenizer
from time import sleep

tokenizer = None

def init_tokenizer():
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize_example(text: str):
    if not text.strip():
        return []
    ids = tokenizer(text, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
    ids.append(tokenizer.eos_token_id)
    return ids

def process_wet(path: Path):
    END_TOKEN = "<|endoftext|>"
    buf = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if END_TOKEN in line:
                left, _, right = line.partition(END_TOKEN)
                buf.append(left)
                yield "".join(buf)
                buf = [right]
            else:
                buf.append(line)

        # ending
        if buf and any(s.strip() for s in buf):
            yield "".join(buf)

def process_chunk(chunk_files: list[str], out_bin: str):
    init_tokenizer()
    with open(out_bin, "wb") as out_file:
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as tp:
            for chunk_file in chunk_files:
                wet_file = process_wet(Path(chunk_file))
                for sample_toks in tp.map(tokenize_example, wet_file):
                    out_file.write(np.asarray(sample_toks, dtype=np.uint16).tobytes())
    return len(chunk_files)

def get_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def tokenize_filtered(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wet_files = sorted(input_dir.glob("CC-*.warc.wet.gz"))
    print(f"{len(wet_files)} wet files")
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
    chunks = list(get_chunks(wet_files, chunk_size))

    # some chunks failed initially
    failed_chunks = set([int(re.match(r"\d+", x).group(0)) for x in open("/data/c-aalag/tokenized_cc2/failed.txt").read().splitlines()])
    chunks = [c for i, c in enumerate(chunks) if i in failed_chunks]
    print(len(chunks))
    print(f"{len(chunks)} chunks")
    jobs = executor.map_array(
        process_chunk,
        [[str(f) for f in c] for c in chunks],
        [str(output_dir / f"chunk_{i:04d}.bin") for i in range(len(chunks))],
    )

    bar = tqdm(total=len(jobs), desc="chunks")
    pending = set(jobs)

    while pending:
        done_now = [j for j in pending if j.done()]
        for j in done_now:
            bar.update(1)
            pending.remove(j)
        sleep(10)

    totals = [j.result() for j in jobs]
    print(f"tokens total: {sum(totals)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/data/c-aalag/filtered_cc3")
    parser.add_argument("--output_dir", default="/data/c-aalag/tokenized_cc2")
    args = parser.parse_args()
    tokenize_filtered(args)