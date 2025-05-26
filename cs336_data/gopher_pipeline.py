from classify_data import GopherQualityClassifier
from parse_html import warc_text_iterator
import argparse

def run_gopher(warc_file, num_samples):
    gopher_quality_classifier = GopherQualityClassifier()
    for sample in warc_text_iterator(warc_file):
        should_keep = gopher_quality_classifier.classify(sample)
        print("Should Keep:", should_keep)
        print("Sample:\n", sample[:1000], "\n\n")

        num_samples -= 1
        if num_samples <= 0:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--warc_file', type=str, default='/data/CC/example.warc.wet.gz')
    parser.add_argument("--num_samples", type=int, default=20)
    args = parser.parse_args()
    run_gopher(args.warc_file, args.num_samples)