from pathlib import Path
from argparse import ArgumentParser
import csv


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--dataset_folder", type=Path, required=True)
    parser.add_argument("--tr_no_dev_scp", type=Path, required=True)
    parser.add_argument("--dev_scp", type=Path, required=True)
    parser.add_argument("--eval_scp", type=Path, required=True)
    parser.add_argument("--utt_prefix", type=str, default="fsd50k")
    return parser


def main():
    args = get_parser().parse_args()
    Path(args.tr_no_dev_scp).parent.mkdir(parents=True, exist_ok=True)
    Path(args.dev_scp).parent.mkdir(parents=True, exist_ok=True)
    Path(args.eval_scp).parent.mkdir(parents=True, exist_ok=True)

    # tr_no_dev and dev
    with open(args.dataset_folder / "FSD50K.ground_truth/dev.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            fname, labels, mids, split = row
            if split == "train":
                with open(args.tr_no_dev_scp, "a") as f:
                    f.write(
                        f"{args.utt_prefix}_{fname} {args.dataset_folder}/FSD50K.dev_audio/{fname}.wav\n"
                    )
            elif split == "val":
                with open(args.dev_scp, "a") as f:
                    f.write(
                        f"{args.utt_prefix}_{fname} {args.dataset_folder}/FSD50K.dev_audio/{fname}.wav\n"
                    )
            else:
                raise ValueError(f"Found unexpected split in fsd50k: {split}")

    # eval
    with open(args.dataset_folder / "FSD50K.ground_truth/eval.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            fname, labels, mids = row
            with open(args.eval_scp, "a") as f:
                f.write(
                    f"{args.utt_prefix}_{fname} {args.dataset_folder}/FSD50K.eval_audio/{fname}.wav\n"
                )


if __name__ == "__main__":
    main()
