from __future__ import annotations

from typing import Optional
import argparse
from pathlib import Path
import pandas as pd
import yaml

BLANK_VALUE = "-"
EXPERIMENT_TAG_COLUMN = "Experiment"
IGNORED_CONFIG_KEYS = ("outdir", "config", "resume", "save_interval_steps", "train_max_steps")


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
    ignored_config_diffs = ["conf." + key for key in ignored_config_diffs]
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
        ((results_pd, reference_config_pd), (exp_results_pd, exp_config_pd))
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
            metric = result_tag + "." + result_file.parent.stem[:-len("_res")] # NOTE: use parent folder name as metric name
            if metric in results:
                raise ValueError(f"Duplicate metric found: {metric}")
            results[metric] = result
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
    config = pd.json_normalize(config)
    config.columns = ["conf." + col for col in config.columns]
    return config


def combine_rows_with_default_pds(pds):
    """
    Combine rows of multiple DataFrames with default values for missing columns.

    Parameters:
    pds (list of tuples): A list where each element is a tuple containing two pandas DataFrames.
                          The first DataFrame in the tuple is the one to be combined.
                          The second DataFrame contains default values to fill in missing columns;
                          It is assumed to have only one row, whose values will be broadcasted to all missing columns for all rows in the first DataFrame.

    Returns:
    pd.DataFrame: A single DataFrame with rows combined from all input DataFrames, 
                  with missing columns filled with default values from the corresponding second DataFrame in each tuple.
    """
    combined_df = pd.concat([df for df, _ in pds])
    # assert no duplicate index
    assert combined_df.index.duplicated().sum() == 0, "Duplicate index found in the combined DataFrames, which is not expected. Previous result should have been dropped before calling this function."
    for df, default in pds:
        missing_cols = combined_df.columns.difference(df.columns)
        missing_cols_with_default = missing_cols.intersection(default.columns)
        if not missing_cols_with_default.empty:
            combined_df.loc[df.index, missing_cols_with_default] = default.loc[default.index, missing_cols_with_default].values
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
            results_pd = pd.read_csv(f, index_col=0, header=[0, 1], na_values=BLANK_VALUE)
    result_folders = list((args.exp_path / "wav").iterdir())
    results_pd.columns = ['.'.join([i for i in col if pd.notna(i)]) for col in results_pd.columns] # convert to flat column index
    results_pd = update_results(
        results_pd,
        args.exp_path / "config.yml",
        result_folders,
        args.ref_exp_conf,
        exp_tag=args.exp_path.name,
        result_tags=[result_path.name for result_path in result_folders],
    )

    results_pd.sort_index(axis=1, inplace=True, key=lambda columns: [f"0{col}" if col.startswith("conf.") else (col if not col.split(".")[-1].startswith("VC_") else col.replace(".VC_", ".ZVC_")) for col in columns]) # prioritize config columns, deprioritize VC results
    results_pd = results_pd.iloc[results_pd.apply(lambda row: row.to_list(), axis=1).argsort()] # sort rows by content
    results_pd.columns = pd.MultiIndex.from_tuples([col.split('.', maxsplit=1) for col in results_pd.columns]) # convert back to multi-index

    with open(args.results_csv, "w") as f:
        f.write(f"[Reference Experiment Config]{args.ref_exp_conf}\n")
    results_pd.to_csv(args.results_csv, mode="a", header=True, na_rep=BLANK_VALUE)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    # args = parser.parse_args("exp/tr_no_dev_human_hifigan_token_16k_nodp_f0_spemb_tok_embed_dropout0.2.v1_1224 exp/tr_no_dev_human_hifigan_token_16k_nodp_f0_spemb.v1_1224/config.yml summary.csv".split())
    main(args)
