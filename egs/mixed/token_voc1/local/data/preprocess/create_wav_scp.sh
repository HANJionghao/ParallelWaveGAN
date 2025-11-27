# This script iterates through all the audio files in ${dataset_folder} and creates a wav.scp file.

dataset_folder=
verbose=false
output_wav_scp=
dataset_tag=
supported_audio_exts=("wav" "mp3" "flac") # source audio file types that will be collected

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1

if [ -z "${dataset_folder}" ]; then
    echo "[ERROR] Please set the dataset_folder variable."
    exit 1
fi
if [ -z "${output_wav_scp}" ]; then
    echo "[ERROR] Please set the output_wav_scp variable."
    exit 1
fi
if [ -z "${dataset_tag}" ]; then
    echo "[ERROR] Please set the dataset_tag variable."
    exit 1
fi

args=()
for ext in "${supported_audio_exts[@]}"; do
    args+=(-name "*.${ext}")
    args+=(-o)
done
unset 'args[-1]'  # remove trailing -o

> "${output_wav_scp}"

dataset_folder=$(realpath "${dataset_folder}")

echo "[INFO] Creating ${output_wav_scp} from ${dataset_folder} by collecting audio files with the following extensions: ${supported_audio_exts}"
mkdir -p "$(dirname "${output_wav_scp}")"
find "$dataset_folder" -type f \( "${args[@]}" \) ! -name '._*' | sort | while read -r file; do
    utt_id="${file#${dataset_folder}/}"
    utt_id="${utt_id%.*}" # remove file extension
    utt_id=$(echo "$utt_id" | perl -CS -pe 'chomp; s/\p{Space}/_/g') # replace spaces with underscores
    utt_id="${utt_id//[\/.,\']/_}" # e.g., "Japanese/JA-Tenor-1/Vibrato/Heartful Song/Paired_Speech_Group/0003.wav" -> "Japanese_JA-Tenor-1_Vibrato_Heartful_Song_Paired_Speech_Group_0003"
    echo "${dataset_tag}_${utt_id} ${file}" >> "${output_wav_scp}"
done

LC_ALL=C sort -k1,1 -u "${output_wav_scp}" -o "${output_wav_scp}"

if [ "$verbose" = true ]; then
    echo "[INFO] Finished creating ${output_wav_scp} from ${dataset_folder} for dataset: ${dataset_tag}"
fi
