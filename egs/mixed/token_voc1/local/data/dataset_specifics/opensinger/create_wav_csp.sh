dataset_folder=
output_wav_scp=
verbose=false

# Expected directory structure:
# <dataset_folder>
# ├── WomanRaw
# │   ├── 0_一如年少模样
# │   │   ├── 0_光年之外
# │   │   ├── 0_光年之外_0.lab
# │   │   ├── 0_光年之外_0.txt
# │   │   ├── 0_光年之外_0.wav
# │   │   └── ...
# │   └── ...
# ├── ManRaw
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

echo "Creating ${output_wav_scp} from ${dataset_folder} for OpenSinger dataset"
mkdir -p "$(dirname "${output_wav_scp}")"
for gender_dir in "${dataset_folder}"/*; do
    if [ -d "${gender_dir}" ]; then
        gender=$(basename "${gender_dir}")
        for song_dir in "${gender_dir}"/*; do
            if [ "$verbose" = true ]; then
                echo "Processing OpenSinger dataset for gender: ${gender}, folder: $(basename "${song_dir}")"
            fi
            for wav in "${song_dir}"/*.wav; do
                IFS='_' read -r singerid song segid <<< $(basename "$wav" .wav)
                utt="opensinger_${gender}$(printf "%02d" "$singerid")_${song}_$(printf "%02d" "$segid")"
                echo "${utt} ${wav}" >> "${output_wav_scp}"
            done
        done
    fi
done

sort -u "${output_wav_scp}" -o "${output_wav_scp}"

if [ "$verbose" = true ]; then
    echo "Finished creating ${output_wav_scp} from ${dataset_folder} for OpenSinger dataset"
fi
