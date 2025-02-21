from __future__ import annotations

from typing import Optional
import argparse
from pathlib import Path
import pandas as pd
import yaml

BLANK_VALUE = "-"
EXPERIMENT_TAG_COLUMN = "Experiment"
IGNORED_CONFIG_KEYS = ("outdir", "config")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_path", type=Path, help="Path to the experiment directory.")
    parser.add_argument(
        "ref_exp_conf",
        type=Path,
        help="Path to the reference config file.",
    )
    parser.add_argument("results_csv", type=Path, help="Path to the results CSV")
    return parser


def update_results(
    results_pd: pd.DataFrame,
    exp_config: Path,
    result_folders: iter[Path],
    reference_config: Path,
    exp_tag="",
    result_tags: Optional[iter[str]] = None,
    ignored_config_diffs=IGNORED_CONFIG_KEYS,
):
    # parse config
    exp_config_pd = read_and_flatten_config(exp_config)
    exp_config_pd[EXPERIMENT_TAG_COLUMN] = [exp_tag]
    exp_config_pd.set_index(EXPERIMENT_TAG_COLUMN, inplace=True)
    reference_config_pd = read_and_flatten_config(reference_config)
    exp_config_diff = get_config_differences(exp_config_pd, reference_config_pd, ignored_config_diffs)
    
    # add results
    if result_tags is None:
        result_tags = [""] * len(result_folders)
    
    exp_results_dict = {}
    for result_folder, result_tag in zip(result_folders, result_tags):
        # NOTE(jhan): The later results will overwrite the previous results if they have the same key.
        exp_results_dict.update(get_results(result_folder, result_tag=result_tag))
        
    exp_results_pd = pd.DataFrame(exp_results_dict, index=[exp_tag])
    exp_results_pd = exp_config_diff.join(exp_results_pd)

    if exp_tag in results_pd.index:
        # NOTE(jhan): Drop the previous results if they exist.
        results_pd.drop(exp_tag, inplace=True)
    results_pd = combine_rows_with_default_pds(
        ((exp_results_pd, exp_config_pd), (results_pd, reference_config_pd))
    )

    results_pd.sort_index(inplace=True)

    return results_pd


def get_results(result_folder: Path, result_tag=""):
    """
    Extracts results from the experiment results directory.

    Parameters:
    result_folder (Path): Path to the experiment results directory.
    result_tag (str): Tag to prepend to each result key.

    Returns:
    dict: A dictionary with result keys and their corresponding values.
    """
    results = {}
    for result_file in result_folder.glob("*_res/*_avg_result.txt"):
        if result_file.is_file():
            with open(result_file, "r") as f:
                f.readline()
                result = f.readline().strip()[len("Average: "):]
                results[result_tag + "." + result_file.stem[:-len("_avg_result")]] = result
    return results


def get_config_differences(exp_config_pd, reference_config_pd, ignored_diffs):
    combined_df = pd.concat([exp_config_pd, reference_config_pd])
    diff = combined_df.iloc[0].ne(combined_df.iloc[1])
    exp_config_diff = combined_df.loc[exp_config_pd.index, diff]
    exp_config_diff.drop(ignored_diffs, axis=1, errors="ignore", inplace=True)
    return exp_config_diff


def read_and_flatten_config(config_path: Path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return pd.json_normalize(config)


def combine_rows_with_default_pds(pds):
    combined_df = pd.concat(next(zip(*pds)))
    for df, default in pds:
        missing_cols = combined_df.columns.difference(df.columns)
        missing_cols_with_default = missing_cols.intersection(default.columns)
        combined_df.loc[df.index, missing_cols_with_default] = default.loc[df.index, missing_cols_with_default]
    return combined_df


def main(args):
    if not args.results_csv.exists():
        results_pd = pd.DataFrame()
    else:
        with open(args.results_csv, "r") as f:
            # match from f"[Reference Experiment Config]{outfile_ref_exp_conf}""
            first_line = f.readline()
            if not first_line.startswith("[Reference Experiment Config]"):
                raise ValueError("Results CSV must start with the reference experiment config.")
            outfile_ref_exp_conf = Path(first_line[len("[Reference Experiment Config]"):].strip())
            if outfile_ref_exp_conf.resolve() != args.ref_exp_conf.resolve():
                raise ValueError(
                    f"Reference experiment config in results CSV ({outfile_ref_exp_conf}) does not match provided reference config ({args.ref_exp_conf})."
                )
            results_pd = pd.read_csv(f, index_col=EXPERIMENT_TAG_COLUMN)
    result_folders = list((args.exp_path / "wav").iterdir())
    results_pd = update_results(
        results_pd,
        args.exp_path / "config.yml",
        result_folders,
        args.ref_exp_conf,
        exp_tag=args.exp_path.stem,
        result_tags=[result_path.stem for result_path in result_folders],
    )

    with open(args.results_csv, "w") as f:
        f.write(f"[Reference Experiment Config]{args.ref_exp_conf}\n")
    results_pd.to_csv(args.results_csv, mode="a", na_rep=BLANK_VALUE, header=True)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
