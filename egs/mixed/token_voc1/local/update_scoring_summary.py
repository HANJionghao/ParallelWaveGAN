import argparse
from pathlib import Path
import pandas as pd
import yaml

BLANK_VALUE = "-"


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
    exp_results: Path,
    reference_config: Path,
    exp_tag="",
    results_tag="",
    ignore_diffs=["outdir", "config"],
):
    # add config
    exp_config_pd = read_and_flatten_config(exp_config)
    reference_config_pd = read_and_flatten_config(reference_config)
    exp_config_diff = calculate_config_diff(exp_config_pd, reference_config_pd, ignore_diffs)
    exp_config_diff.index = [exp_tag]

    # if "config" in exp_config_diff.columns:
    #     exp_base_config = Path(exp_config_diff["config"].iloc[0])
    #     reference_base_config = Path(reference_config_pd["config"].iloc[0])
    #     exp_base_config_pd = read_and_flatten_config(exp_base_config)
    #     reference_base_config_pd = read_and_flatten_config(reference_base_config)
    #     exp_base_config_diff = calculate_config_diff(
    #         exp_base_config_pd, reference_base_config_pd
    #     )
    #     exp_base_config_diff.index = [exp_tag]
    #     exp_config_diff = exp_config_diff.merge(exp_base_config_diff, how="left")
    
    align_columns_with_defaultpd(exp_config_diff, results_pd, reference_config_pd)
    align_columns_with_defaultpd(results_pd, exp_config_diff, exp_config_pd)
    
    # add results
    exp_results_pd = pd.DataFrame(get_results(exp_results, result_tag=results_tag), index=[exp_tag])
    exp_results_pd.sort_index(axis=1, inplace=True)
    # align_columns_with_default(exp_results_pd, results_pd, None)
    # align_columns_with_default(results_pd, exp_results_pd, None)
    exp_results_pd = exp_config_diff.join(exp_results_pd)

    # if results_tag in results_pd's index, update the row, else add a new row
    if exp_tag in results_pd.index:
        results_pd.loc[exp_tag] = exp_results_pd.loc[exp_tag]
    else:
        results_pd = pd.concat([results_pd, exp_config_diff])
        results_pd = pd.concat([results_pd, exp_results_pd])
    results_pd.fillna(BLANK_VALUE, inplace=True)

    return results_pd


def get_results(exp_results: Path, result_tag=""):
    """
    Extracts results from the experiment results directory.

    Parameters:
    exp_results (Path): Path to the experiment results directory.
    result_tag (str): Tag to prepend to each result key.

    Returns:
    dict: A dictionary with result keys and their corresponding values.
    """
    results = {}
    for result_folder in exp_results.glob("*_res"):
        if result_folder.is_dir():
            for result_file in result_folder.glob("*_avg_result.txt"):
                if result_file.is_file():
                    with open(result_file, "r") as f:
                        f.readline()
                        result = f.readline().strip()[len("Average: "):]
                        results[result_tag + "_" + result_file.stem[:-len("_avg_result")]] = result
    return results


def calculate_config_diff(exp_config_pd, reference_config_pd, ignore_diffs):
    align_columns(exp_config_pd, reference_config_pd)
    exp_config_pd.sort_index(axis=1, inplace=True)
    reference_config_pd.sort_index(axis=1, inplace=True)
    diff = exp_config_pd.compare(reference_config_pd)
    exp_config_diff = diff.xs("self", level=1, axis=1)
    exp_config_diff = exp_config_diff.drop(ignore_diffs, axis=1, errors="ignore")
    return exp_config_diff


def read_and_flatten_config(config_path: Path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return pd.json_normalize(config)


def align_columns(df1, df2):
    for col in df1.columns:
        if col not in df2.columns:
            df2[col] = None
    for col in df2.columns:
        if col not in df1.columns:
            df1[col] = None

def align_columns_with_defaultpd(df1, df2, default_df):
    if df2.empty:
        return
    for col in df1.columns:
        if col not in df2.columns:
            df2[col] = default_df[col].iloc[0]

def align_columns_with_default(df1, df2, default_val):
    for col in df1.columns:
        if col not in df2.columns:
            df2[col] = default_val


def main(args):
    if not args.results_csv.exists():
        results_pd = pd.DataFrame(index=["Experiment"])
    else:
        with open(args.results_csv, "r") as f:
            outfile_ref_exp_conf = Path(f.readline().strip())
        if outfile_ref_exp_conf != args.ref_exp_conf:
            raise ValueError(
                f"Reference experiment config in results CSV ({outfile_ref_exp_conf}) does not match provided reference config ({args.ref_exp_conf})."
            )
        results_pd = pd.read_csv(args.results_csv, index_col="Experiment", skiprows=1)
    for exp_results in (args.exp_path / "wav").iterdir():
        updated_results_pd = update_results(
            results_pd,
            args.exp_path / "config.yml",
            exp_results,
            args.ref_exp_conf,
            exp_tag=args.exp_path.stem,
            results_tag=exp_results.stem,
        )

    with open(args.results_csv, "w") as f:
        f.write(f"{args.ref_exp_conf}\n")
    updated_results_pd.to_csv(args.results_csv, mode="a", header=True)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
