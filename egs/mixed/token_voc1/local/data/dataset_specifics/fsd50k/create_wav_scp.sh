dataset_folder=
tr_no_dev_scp=
dev_scp=
eval_scp=
verbose=false
utt_prefix=fsd50k

# Expected directory structure:
# <dataset_folder>
# ├── FSD50K.dev_audio
# │   └── *.wav
# ├── FSD50K.eval_audio
# │   └── *.wav
# └── FSD50K.ground_truth
#     ├── dev.csv
#     ├── eval.csv
#     └── ...


# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1

if [ -z "${dataset_folder}" ]; then
    echo "Please set the dataset_folder variable."
    exit 1
fi
if [ -z "${tr_no_dev_scp}" ] || [ -z "${dev_scp}" ] || [ -z "${eval_scp}" ]; then
    echo "Please set the tr_no_dev_scp, dev_scp and eval_scp variables."
    exit 1
fi

echo "Creating ${tr_no_dev_scp}, ${dev_scp} and ${eval_scp} from ${dataset_folder} for fsd50k dataset"

python local/data/dataset_specifics/fsd50k/create_wav_scp.py \
    --dataset_folder "${dataset_folder}" \
    --tr_no_dev_scp "${tr_no_dev_scp}" \
    --dev_scp "${dev_scp}" \
    --eval_scp "${eval_scp}"

sort -k1,1 -u "${tr_no_dev_scp}" -o "${tr_no_dev_scp}"
sort -k1,1 -u "${dev_scp}" -o "${dev_scp}"
sort -k1,1 -u "${eval_scp}" -o "${eval_scp}"

if [ "$verbose" = true ]; then
    echo "Finished creating ${tr_no_dev_scp}, ${dev_scp} and ${eval_scp} from ${dataset_folder} for fsd50k dataset"
fi
