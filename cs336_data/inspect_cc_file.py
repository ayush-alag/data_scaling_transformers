import argparse
from parse_html import warc_text_iterator
import gzip

def view_filtered_cc(filtered_text_file, num_records):
    with gzip.open(filtered_text_file, "rb") as in_file:
        for i, record in enumerate(in_file):
            text = record.decode("utf-8", errors="replace")
            if i >= num_records:
                break

            print(text[:1000], "\n")

def view_rejected_cc(rejected_cc_file, num_records):
    with open(rejected_cc_file, "r") as in_file:
        for i, record in enumerate(in_file):
            text = record
            if i >= num_records:
                break

            print(text[:1000], "\n")

if __name__ == '__main__':
    # take in warc file via argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--warc_file', type=str, default='/data/c-aalag/filtered_cc/CC-MAIN-20250424022518-20250424052518-00711.warc.wet.gz')
    parser.add_argument('--warc_file', type=str, default='/data/c-aalag/rejected_cc/CC-MAIN-20250424053916-20250424083916-00049.warc.wet.gz')
    parser.add_argument('--num_records', type=int, default=10)
    args = parser.parse_args()
    view_rejected_cc(args.warc_file, args.num_records)