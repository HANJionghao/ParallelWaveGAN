# This script iterates through all the audio files in ${dataset_folder} and creates a wav.scp file.

dataset_folder=
verbose=false
train_scp=
eval_scp=
utt_prefix=laion_audio_630k_no_overlap

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1

if [ -z "${dataset_folder}" ]; then
    echo "[ERROR] Please set the dataset_folder variable."
    exit 1
fi
if [ -z "${train_scp}" ] || [ -z "${eval_scp}" ]; then
    echo "[ERROR] Please set the train_scp and eval_scp variables."
    exit 1
fi

dataset_folder=$(realpath "${dataset_folder}")
mkdir -p "$(dirname "${train_scp}")"
mkdir -p "$(dirname "${eval_scp}")"

function create_wav_scp {
    local output_wav_scp=$1
    local dataset_folder=$2
    local utt_prefix=$3
    shift 3
    local subsets=("$@")

    echo "[INFO] Creating ${output_wav_scp} for ${subsets[@]} from ${dataset_folder} by collecting audio files with the following extensions: flac"
    > "${output_wav_scp}"

    for subset in "${subsets[@]}"; do
        find "$dataset_folder/${subset}" -type f -name "*.flac" ! -name '._*' | sort | while read -r file; do
            utt_id="${file#${dataset_folder}/}"
            utt_id="${utt_id%.*}"
            utt_id=$(echo "$utt_id" | perl -CS -pe 'chomp; s/\p{Space}/_/g')
            utt_id="${utt_id//[\/.,\']/_}"
            echo "${utt_prefix}_${utt_id} ${file}" >> "${output_wav_scp}"
        done
    done
    sort -k1,1 -u "${output_wav_scp}" -o "${output_wav_scp}"
}

create_wav_scp "${eval_scp}" "${dataset_folder}/freesound_no_overlap" "${utt_prefix}" "test"
create_wav_scp "${train_scp}" "${dataset_folder}/freesound_no_overlap" "${utt_prefix}" "train_1" "train_2"

if [ "$verbose" = true ]; then
    echo "[INFO] Finished creating ${train_scp} and ${eval_scp} from ${dataset_folder} for dataset: ${utt_prefix}"
fi
