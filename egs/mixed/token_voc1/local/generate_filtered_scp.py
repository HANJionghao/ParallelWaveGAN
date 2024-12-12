import argparse
from parallel_wavegan.utils import find_files
from pathlib import Path
import re

def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate filtered scp file based on audio_query.",
    )
    parser.add_argument(
        "--src-scp",
        type=Path,
        required=True,
        help="scp file path.",
    )
    parser.add_argument(
        "--dumpdir",
        type=Path,
        required=True,
        help="dump directory path.",
    )
    parser.add_argument(
        "--audio-query",
        type=str,
        required=True,
        help="query string to identify included utterances.",
    )
    parser.add_argument(
        "--output-scp",
        type=Path,
        required=True,
        help="output scp file path.",
    )
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    filtered_files = find_files(str(args.dumpdir), args.audio_query)
    # utts should be parsed from audio_query, e.g. *.h5, where * is the utt_id
    re_pattern = re.compile(args.audio_query.replace("*", "(.*)"))
    utts = sorted([re_pattern.match(Path(file).name).group(1) for file in filtered_files])
    with open(args.src_scp, "r") as infile, open(args.output_scp, "w") as outfile:
        for line in infile:
            utt = line.split(maxsplit=1)[0]
            # print(f"{utt=}")
            if utt in utts:
                outfile.write(line)
    print(f"Filtered scp file is generated at {args.output_scp}")

if __name__ == "__main__":
    main()
