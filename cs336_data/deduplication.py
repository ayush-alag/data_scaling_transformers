import os
import hashlib
from collections import defaultdict
import mmh3
import ctypes
import unicodedata
import re
import networkx as nx
import random
import itertools
import string

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

def get_documents(input_files):
    documents = []
    for file in input_files:
        with open(file, "r", encoding="utf-8") as f:
            documents.append(f.read())
    return documents

def normalize_document(document):
    normalized = unicodedata.normalize('NFD', document.lower())
    normalized = ''.join(c for c in normalized if not unicodedata.combining(c))
    punct_pattern = r"[{}]".format(re.escape(string.punctuation))
    normalized = re.sub(punct_pattern, '', normalized)

    return normalized

def get_ngrams(document, ngrams):
    # we want word n-grams, not character n-grams
    words = document.split()
    return set([' '.join(words[i : i + ngrams]) for i in range(len(words) - ngrams + 1)])

def minhash_signature(ngram_doc, hash_functions):
    return [min([hash_function(ngram) for ngram in ngram_doc]) for hash_function in hash_functions]

def split_signature(signature, num_bands):
    band_size = len(signature) // num_bands
    return [signature[i : i + band_size] for i in range(0, len(signature), band_size)]

def get_candidate_duplicates(bands_all_docs, jaccard_threshold):
    buckets = defaultdict(list)
    for doc_index, bands in enumerate(bands_all_docs):
        for band_idx, band in enumerate(bands):
            buckets[(band_idx, tuple(band))].append(doc_index)

    candidate_duplicates = set()
    for bucket, docs in buckets.items():
        if len(docs) > 1:
            for pair in itertools.combinations(docs, 2):
                candidate_duplicates.add(pair)

    return candidate_duplicates

def get_jaccard_similarity(set1, set2):
    return len(set1.intersection(set2)) / len(set1.union(set2))

def get_true_fuzzy_duplicates(ngram_docs, candidate_duplicates, jaccard_threshold):
    true_fuzzy_duplicates = []
    for doc_idx_pair in candidate_duplicates:
        doc1 = ngram_docs[doc_idx_pair[0]]
        doc2 = ngram_docs[doc_idx_pair[1]]
        jaccard_similarity = get_jaccard_similarity(doc1, doc2)
        if jaccard_similarity >= jaccard_threshold:
            true_fuzzy_duplicates.append(doc_idx_pair)

    return true_fuzzy_duplicates

def get_clusters(documents, true_fuzzy_duplicates):
    G = nx.Graph()
    G.add_edges_from(true_fuzzy_duplicates)
    G.add_nodes_from(range(len(documents)))
    return list(nx.connected_components(G))

def get_representative_documents(documents, clusters, input_files):
    deduplicated_docs = []
    filenames = []
    for cluster in clusters:
        rep_idx = random.choice(list(cluster))
        deduplicated_docs.append(documents[rep_idx])
        filenames.append(input_files[rep_idx])
    return deduplicated_docs, filenames

def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    # get a set of documents from the input files
    # for each document, get the set of ngrams
    # create k hash functions
    # for each hash function, hash each ngram and take the min hash value over the document
    # we have k min hash values per document
    # divide the signature into b bands of r rows each
    # candidate duplicates have matching signatures in at least one band
    # compute jaccard similarity between the candidate duplicates
    # output the candidate duplicates with jaccard similarity greater than the threshold
    documents = get_documents(input_files)
    normalized_documents = [normalize_document(doc) for doc in documents]
    ngram_docs = [get_ngrams(doc, ngrams) for doc in normalized_documents]

    hash_functions = create_hash_functions(num_hashes)
    minhash_signatures_all_docs = [minhash_signature(ngram_doc, hash_functions) for ngram_doc in ngram_docs]
    bands_all_docs = [split_signature(signature, num_bands) for signature in minhash_signatures_all_docs]
    candidate_duplicates = get_candidate_duplicates(bands_all_docs, jaccard_threshold)

    # need to compute the true jaccard similarity for the candidate duplicates
    true_fuzzy_duplicates = get_true_fuzzy_duplicates(ngram_docs, candidate_duplicates, jaccard_threshold)
    # use the similar pairs to create clusters
    clusters = get_clusters(documents, true_fuzzy_duplicates)
    deduplicated_docs, filenames = get_representative_documents(documents, clusters, input_files)

    os.makedirs(output_directory, exist_ok=True)
    for doc, filename in zip(deduplicated_docs, filenames):
        output_file = os.path.join(output_directory, os.path.basename(filename))
        with open(output_file, "w") as f:
            f.write(doc)