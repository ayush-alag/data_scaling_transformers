import os
import hashlib
from collections import defaultdict
import mmh3
import ctypes

def exact_line_deduplication(input_files, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    hash_line_to_count = defaultdict(int)
    for file in input_files:
        with open(file, "r") as f:
            for line in f:
                line_hash = hashlib.md5(line.encode('utf-8')).hexdigest()
                hash_line_to_count[line_hash] += 1

    for file in input_files:
        output_file = os.path.join(output_dir, os.path.basename(file))
        with open(output_file, "w", encoding="utf-8") as output_f:
            with open(file, "r", encoding="utf-8") as input_f:
                for line in input_f:
                    line_hash = hashlib.md5(line.encode('utf-8')).hexdigest()
                    if hash_line_to_count[line_hash] > 1:
                        continue
                    output_f.write(line)

    return hash_line_to_count

def create_hash_functions(num_functions, seed=0):
    hash_functions = []

    for i in range(num_functions):
        def hash_func(data, seed=i+seed):
            return ctypes.c_uint32(mmh3.hash(data, seed)).value

        hash_functions.append(hash_func)

    return hash_functions

def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    hash_functions = create_hash_functions(num_hashes)

    for file in input_files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                for hash_function in hash_functions:
                    hash_value = hash_function(line.encode('utf-8')).hexdigest()

    raise NotImplementedError