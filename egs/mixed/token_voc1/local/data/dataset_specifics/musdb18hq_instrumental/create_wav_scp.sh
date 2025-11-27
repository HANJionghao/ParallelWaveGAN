dataset_folder=
train_scp=
eval_scp=
verbose=false
utt_prefix=musdb18hq_instrumental

# Expected directory structure:
# <dataset_folder>
# ├── test
# │   └── ... -> song folders
# └── train
#     └── ... -> song folders


# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1

if [ -z "${dataset_folder}" ]; then
    echo "Please set the dataset_folder variable."
    exit 1
fi
if [ -z "${train_scp}" ] || [ -z "${eval_scp}" ]; then
    echo "Please set the train_scp and eval_scp variables."
    exit 1
fi

echo "Creating ${train_scp} and ${eval_scp} from ${dataset_folder} for MUSDB18-HQ dataset"
mkdir -p "$(dirname "${train_scp}")"
mkdir -p "$(dirname "${eval_scp}")"

# test
find "${dataset_folder}/test" -type f -name "*.wav" ! -name "vocals.wav" ! -name "mixture.wav" | while read -r audio; do
    utt_id="${audio#${dataset_folder}/}"
    utt_id="${utt_id//\//_}"
    utt_id=$(echo "$utt_id" | perl -CS -pe 'chomp; s/\p{Space}/_/g') # replace spaces with underscores
    utt_id="${utt_id%.*}"
    echo "${utt_prefix}_${utt_id} ${audio}" >> "${eval_scp}"
done

# train
find "${dataset_folder}/train" -type f -name "*.wav" ! -name "vocals.wav" ! -name "mixture.wav" | while read -r audio; do
    utt_id="${audio#${dataset_folder}/}"
    utt_id="${utt_id//\//_}"
    utt_id=$(echo "$utt_id" | perl -CS -pe 'chomp; s/\p{Space}/_/g') # replace spaces with underscores
    utt_id="${utt_id%.*}"
    echo "${utt_prefix}_${utt_id} ${audio}" >> "${train_scp}"
done

sort -k1,1 -u "${train_scp}" -o "${train_scp}"
sort -k1,1 -u "${eval_scp}" -o "${eval_scp}"

if [ "$verbose" = true ]; then
    echo "Finished creating ${train_scp} and ${eval_scp} from ${dataset_folder} for MUSDB18-HQ dataset"
fi
