dataset_folder=
output_wav_scp=
verbose=false

# Expected directory structure:
# <dataset_folder>
# ├── bossa_nova
# │   ├── dongurikorokoro
# │   │   ├── alto.wav
# │   │   ├── bass.wav
# │   │   ├── lead_vocal.wav
# │   │   └── ...
# │   └── ...
# ├── enka
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

echo "Creating ${output_wav_scp} from ${dataset_folder} for jaCappella dataset"
mkdir -p "$(dirname "${output_wav_scp}")"
for subset_dir in "${dataset_folder}"/*; do
    if [ -d "${subset_dir}" ]; then
        subset=$(basename "${subset_dir}")
        if [ "$verbose" = true ]; then
            echo "Processing jaCappella dataset for subset: ${subset}"
        fi
        for song_dir in "${subset_dir}"/*; do
            song=$(basename "${song_dir}")
            for voice_part in lead_vocal soprano alto tenor bass; do
                audio="${song_dir}/${voice_part}.wav"
                if [ ! -f "${audio}" ]; then
                    echo "Error: ${audio} does not exist. Please check folder structure or missing files."
                    exit 1
                fi
                utt_id=jacappella_${subset}_${song}_${voice_part}
                echo "${utt_id} ${audio}" >> "${output_wav_scp}"
            done
        done
    fi
done

sort -u "${output_wav_scp}" -o "${output_wav_scp}"

if [ "$verbose" = true ]; then
    echo "Finished creating ${output_wav_scp} from ${dataset_folder} for jaCappella dataset"
fi
