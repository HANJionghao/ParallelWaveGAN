dataset_folder=
tr_no_dev_scp=
dev_scp=
eval_scp=
verbose=false
utt_prefix=urbansound8k

# Expected directory structure:
# <dataset_folder>
# ├── audio
# │   └── fold*/*.wav
# └── metadata
#     └── UrbanSound8K.csv

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

metadata_file="${dataset_folder}/metadata/UrbanSound8K.csv"
tr_no_dev_list=(1 2 3 4 5 6 7 8)
dev_list=(9)
eval_list=(10)

echo "Creating ${tr_no_dev_scp}, ${dev_scp} and ${eval_scp} from ${dataset_folder} for urbansound8k dataset"
mkdir -p "$(dirname "${tr_no_dev_scp}")"
mkdir -p "$(dirname "${dev_scp}")"
mkdir -p "$(dirname "${eval_scp}")"

while IFS=, read -r slice_file_name fsID start end salience fold classID class; do
    if [[ " ${tr_no_dev_list[*]} " =~ " ${fold} " ]]; then
        echo "${utt_prefix}_${fsID}_${start}_${end} ${dataset_folder}/audio/fold${fold}/${slice_file_name}" >> "${tr_no_dev_scp}"
    elif [[ " ${dev_list[*]} " =~ " ${fold} " ]]; then
        echo "${utt_prefix}_${fsID}_${start}_${end} ${dataset_folder}/audio/fold${fold}/${slice_file_name}" >> "${dev_scp}"
    elif [[ " ${eval_list[*]} " =~ " ${fold} " ]]; then
        echo "${utt_prefix}_${fsID}_${start}_${end} ${dataset_folder}/audio/fold${fold}/${slice_file_name}" >> "${eval_scp}"
    fi
done < "${metadata_file}"

sort -k1,1 -u "${tr_no_dev_scp}" -o "${tr_no_dev_scp}"
sort -k1,1 -u "${dev_scp}" -o "${dev_scp}"
sort -k1,1 -u "${eval_scp}" -o "${eval_scp}"

if [ "$verbose" = true ]; then
    echo "Finished creating ${tr_no_dev_scp}, ${dev_scp} and ${eval_scp} from ${dataset_folder} for urbansound8k dataset"
fi
