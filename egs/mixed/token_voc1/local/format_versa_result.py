import argparse
import json
from pathlib import Path
from math import sqrt


class StatisticsCalculator:
    def __init__(self):
        self.total = 0
        self.count = 0
        self.sum_of_squares = 0

    def add_value(self, value):
        self.total += value
        self.count += 1
        self.sum_of_squares += value**2

    def calculate_average(self):
        return self.total / self.count

    def calculate_std(self):
        return sqrt(self.sum_of_squares / self.count - (self.total / self.count) ** 2)


def process_versa_output(versa_output_path, metric_key, output_dir):
    utt2score_path = output_dir / f"utt2{metric_key}"
    metric_avg_path = output_dir / f"{metric_key}_avg_result.txt"
    stats_calculator = StatisticsCalculator()
    
    with open(versa_output_path, "r") as versa_file, open(utt2score_path, "w") as utt2score_file:
        for line in versa_file:
            score_info = json.loads(
                line.strip()
                .replace("'", '"')
                .replace("inf", "Infinity")
                .replace("nan", "0.0")
            )
            utterance = score_info["key"]
            score = score_info[metric_key]
            utt2score_file.write(f"{utterance} {score}\n")
            stats_calculator.add_value(score)

    with open(metric_avg_path, "w") as metric_avg_file:
        metric_avg_file.write(f"#utterances: {stats_calculator.count}\n")
        metric_avg_file.write(
            f"Average: {stats_calculator.calculate_average():.4f}"
            f" ± {stats_calculator.calculate_std():.4f}\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process versa output and calculate metric statistics."
    )
    parser.add_argument(
        "versa_output_path", type=Path, help="Path to the versa output file"
    )
    parser.add_argument(
        "metric_key", type=str, help="The metric key to extract from the versa output"
    )
    parser.add_argument(
        "output_dir", type=Path, help="The directory to save the formatted results"
    )
    args = parser.parse_args()

    versa_output_path = args.versa_output_path
    metric_key = args.metric_key
    output_dir = args.output_dir

    process_versa_output(versa_output_path, metric_key, output_dir)
