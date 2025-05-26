from classify_data import NSFWClassifier, ToxicClassifier
from parse_html import warc_text_iterator
import argparse

def run_nsfw(warc_file, num_samples):
    nsfw_classifier = NSFWClassifier()
    toxic_classifier = ToxicClassifier()
    for sample in warc_text_iterator(warc_file):
        nsfw_label, nsfw_score = nsfw_classifier.classify(sample)
        toxic_label, toxic_score = toxic_classifier.classify(sample)
        # if nsfw_label == "nsfw" or toxic_label == "toxic":
        print("NSFW Label:", nsfw_label, "NSFW Score:", nsfw_score)
        print("Toxic Label:", toxic_label, "Toxic Score:", toxic_score)
        print("Sample:\n", sample[:1000], "\n\n")

        num_samples -= 1
        if num_samples <= 0:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--warc_file', type=str, default='/data/CC/example.warc.wet.gz')
    parser.add_argument("--num_samples", type=int, default=20)
    args = parser.parse_args()
    run_nsfw(args.warc_file, args.num_samples)