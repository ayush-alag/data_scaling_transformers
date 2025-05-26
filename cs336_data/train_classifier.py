import json
import argparse
import subprocess
from pathlib import Path
import fasttext

# prepends __label__ to the text
def build_train_file(pos_dir, neg_dir, train_file):
    def process_file(in_file, out_file):
        for line in in_file.open(encoding="utf-8"):
            obj = json.loads(line)
            text = obj["text"].replace("\n", " ")
            out_file.write(f"{obj['label']} {text}\n")

    with train_file.open("w", encoding="utf-8") as out:
        for pos_file in pos_dir.glob("pos_*.jsonl"):
            process_file(pos_file, out)

        for neg_file in neg_dir.glob("neg_*.jsonl"):
            process_file(neg_file, out)

def run_classifier(args):
    train_file = args.out_dir / "train.txt"
    build_train_file(args.pos_dir, args.neg_dir, train_file)

    model = fasttext.train_supervised(
        input=str(train_file),
        epoch=args.epochs,
        thread=16
    )

    model_path = args.out_dir / "quality_classifier.bin"
    model.save_model(str(model_path))

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pos_dir", type=Path, default="/data/c-aalag/processed_classifier_data/positives")
    p.add_argument("--neg_dir", type=Path, default="/data/c-aalag/processed_classifier_data/negatives")
    p.add_argument("--out_dir", type=Path, default="/data/c-aalag/processed_classifier_data/quality_classifier")
    p.add_argument("--epochs", type=int, default=100)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_classifier(args)
