def check_files_line_alignment(filepaths):
    assert len(filepaths) >= 2, "At least two file paths must be provided."

    file_handles = [open(fp, 'r') for fp in filepaths]

    try:
        for line_num, lines in enumerate(zip(*file_handles), 1):
            fields_list = [line.strip().split() for line in lines]
            lengths = [len(fields) for fields in fields_list]

            if len(set(lengths)) != 1:
                print(f"[Line {line_num}] Length mismatch: {lengths} in {filepaths}")
    finally:
        for f in file_handles:
            f.close()


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Check line alignment across multiple files.")
    parser.add_argument('--data_folder', type=str, help='Path to the data folder.')
    parser.add_argument('--token_files', nargs='+', help='List of token files to check for line alignment.')
    args = parser.parse_args()
    filepaths = [Path(args.data_folder) / fp for fp in args.token_files]
    check_files_line_alignment(filepaths)

# # 用法示例
# splits = ["tr_no_dev", "dev", "eval"]
# data_dir = "/ocean/projects/cis210027p/jhan7/cartoon_voice/model/ParallelWaveGAN/egs/mixed/token_voc1/data_scaleup/jvs_mini_44100Hz_0516_processed/jvs_mini"
# for split in splits:
#     file_list = [
#         f"/ocean/projects/cis210027p/jhan7/cartoon_voice/model/ParallelWaveGAN/egs/mixed/token_voc1/data_human44.1k/{split}/pseudo_labels_contentvec_5_km1024.txt",
#         f"/ocean/projects/cis210027p/jhan7/cartoon_voice/model/ParallelWaveGAN/egs/mixed/token_voc1/data_human44.1k/{split}/pseudo_labels_contentvec_8_km1024.txt",
#         # f"/ocean/projects/cis210027p/jhan7/cartoon_voice/model/ParallelWaveGAN/egs/mixed/token_voc1/data_human44.1k/{split}/pseudo_labels_contentvec_9_km1024.txt",
#         # f"/ocean/projects/cis210027p/jhan7/cartoon_voice/model/ParallelWaveGAN/egs/mixed/token_voc1/data_human44.1k/{split}/pseudo_labels_contentvec_12_km1024.txt",
#     ]
#     # file_list = [
#     #     f"{data_dir}/{split}/pseudo_labels_contentvec_5_km1024.txt",
#     #     f"{data_dir}/{split}/pseudo_labels_contentvec_8_km1024.txt",
#     #     f"{data_dir}/{split}/pseudo_labels_contentvec_9_km1024.txt",
#     #     f"{data_dir}/{split}/pseudo_labels_contentvec_12_km1024.txt",
#     #     f"{data_dir}/{split}/pseudo_labels_contentvec2_5_km1024.txt",
#     #     f"{data_dir}/{split}/pseudo_labels_contentvec2_8_km1024.txt",
#     #     f"{data_dir}/{split}/pseudo_labels_contentvec2_9_km1024.txt",
#     #     f"{data_dir}/{split}/pseudo_labels_contentvec2_12_km1024.txt",
#     #     f"{data_dir}/{split}/pseudo_labels_hubert_large_ll60k_6_km1024.txt",
#     #     f"{data_dir}/{split}/pseudo_labels_wavlm_large_6_km1024.txt",
#     #     f"{data_dir}/{split}/pseudo_labels_wavlm_large_23_km1024.txt",
#     # ]
    
#     check_files_line_alignment(file_list)

# print("Check completed.")