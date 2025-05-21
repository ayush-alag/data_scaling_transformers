from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text
from fastwarc import ArchiveIterator
from pathlib import Path
import argparse

def extract_text_from_html(bytes_str):
    encoding = detect_encoding(bytes_str)
    decoded_str = bytes_str.decode(encoding)
    return extract_plain_text(decoded_str)

def iterate_warc(warc_file):
    if isinstance(warc_file, (str, Path)):
        warc_file = open(warc_file, 'rb')
    for record in ArchiveIterator(warc_file):
        if record.headers['WARC-Type'] == 'conversion':
            yield record

def extract_text_from_warc(warc_file, num_records=10):
    extracted_text = []
    for record in iterate_warc(warc_file):
        if len(extracted_text) >= num_records:
            break

        extracted_text.append(extract_text_from_html(record.reader.read()))
    return extracted_text

def main(warc_file, num_records):
    extracted_text = extract_text_from_warc(warc_file, num_records)
    print("\n\n".join(extracted_text))

if __name__ == '__main__':
    # take in warc file via argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--warc_file', type=str, nargs='?', default='/data/CC/example.warc.wet.gz')
    parser.add_argument('--num_records', type=int, default=10)
    args = parser.parse_args()
    main(args.warc_file, args.num_records)