dataset_folder=
output_wav_scp=
verbose=false

# Expected directory structure:
# <dataset_folder>
# ├── FULL
# │   ├── female1
# │   │   ├── arpeggios
# │   │   │   ├── belt
# │   │   │   └── ...
# │   │   └── ...
# │   └── ...
# └── ...
# or
# <dataset_folder>
# ├── female1
# │   ├── arpeggios
# │   │   ├── belt
# │   │   └── ...
# │   └── ...
# └── ...


# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1

if [ -z "${dataset_folder}" ]; then
    echo "Please set the dataset_folder variable."
    exit 1
fi
if [ -z "${output_wav_scp}" ]; then
    echo "Please set the output_wav_scp variable."
    exit 1
fi

echo "Creating ${output_wav_scp} from ${dataset_folder} for VocalSet dataset"
find "${dataset_folder}" -type f -name "*.wav" | sort | while read -r file; do
    utt_id="vocalset_$(basename "$file" .wav)" # e.g., "<dataset_folder>/FULL/female7/long_tones/trillo/f7_ long_trillo_a.wav" -> "vocalset_f7_ long_trillo_a"
    utt_id="${utt_id// /}" # e.g., -> "vocalset_f7_long_trillo_a"
    echo "${utt_id} ${file}" >> "${output_wav_scp}"
done

sort -u "${output_wav_scp}" -o "${output_wav_scp}"

if [ "$verbose" = true ]; then
    echo "Finished creating ${output_wav_scp} from ${dataset_folder} for VocalSet dataset"
fi
