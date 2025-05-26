import numpy as np
from parse_html import warc_text_iterator
import argparse
from classify_data import LanguageClassifier

def get_indexed_samples(stream, positions):
    positions = sorted(positions)
    pos_idx = 0
    for curr_pos, item in enumerate(stream):
        if curr_pos == positions[pos_idx]:
            yield item
            pos_idx += 1
            if pos_idx >= len(positions):
                return

def random_warc_samples(warc_file, num_samples):
    np.random.seed(2)
    sample_indices = list(np.random.choice(1000, size=num_samples, replace=False))

    warc_stream = warc_text_iterator(warc_file)
    return get_indexed_samples(warc_stream, sample_indices)

def run_language_classification(warc_file, num_samples):
    language_classifier = LanguageClassifier()
    for sample in random_warc_samples(warc_file, num_samples):
        language = language_classifier.classify(sample)
        print(sample, "\n", language, "\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--warc_file', type=str, default='/data/CC/example.warc.wet.gz')
    parser.add_argument("--num_samples", type=int, default=20)
    args = parser.parse_args()
    run_language_classification(args.warc_file, args.num_samples)
