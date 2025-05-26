from mask_data import mask_emails, mask_phone_numbers, mask_ips
from parse_html import warc_text_iterator
import argparse

def mask_pii(text):
    text, _ = mask_emails(text)
    text, _ = mask_phone_numbers(text)
    text, _ = mask_ips(text)
    return text

def run_mask_pii(warc_file, num_samples):
    for sample in warc_text_iterator(warc_file):
        masked_text = mask_pii(sample)
        # print if masked_text is not the same as sample
        if masked_text != sample:
            num_samples -= 1
            print("Before:\n", sample, "\n\n")
            print("After:\n", masked_text, "\n\n")

            if num_samples <= 0:
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--warc_file', type=str, default='/data/CC/example.warc.wet.gz')
    parser.add_argument("--num_samples", type=int, default=2)
    args = parser.parse_args()
    run_mask_pii(args.warc_file, args.num_samples)