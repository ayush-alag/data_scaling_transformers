import numpy as np
import argparse
from transformers import AutoTokenizer
import glob

def collect_tokens(args):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    chunk_files = sorted(glob.glob(f"{args.dir}/chunk_*.bin"))
    print(f"{len(chunk_files)} chunk files")

    output_file = f"{args.dir}/{args.output_suffix}"
    total_tokens = 0
    with open(output_file, "wb") as output_file:
        for chunk_file in chunk_files:
            data = np.fromfile(chunk_file, dtype=np.uint16)
            output_file.write(data.tobytes())
            total_tokens += data.shape[0]

    print(f"{total_tokens} total tokens")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="/data/c-aalag/tokenized_cc")
    parser.add_argument("--output_suffix", type=str, default="full_data.bin")
    args = parser.parse_args()
    collect_tokens(args)