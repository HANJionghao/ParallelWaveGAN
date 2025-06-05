dataset_folder=
output_wav_scp=
verbose=false

# Expected directory structure:
# <dataset_folder>
# ├── popcs-Bad
# │   ├── 0000_ph.txt
# │   ├── 0000.TextGrid
# │   ├── 0000.txt
# │   ├── 0000_wf0.wav
# │   └── ...
# ├── popcs-honey
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

echo "Creating ${output_wav_scp} from ${dataset_folder} for popcs dataset"
mkdir -p "$(dirname "${output_wav_scp}")"
for song_dir in "${dataset_folder}"/popcs-*; do
    if [ -d "${song_dir}" ]; then
        song=$(basename "${song_dir}")
        song=${song#popcs-}
        for audio in "${song_dir}"/*.wav; do
            if [ -f "${audio}" ]; then
                audioid=$(basename "${audio}" .wav)
                echo "popcs_${song}_${audioid} ${audio}" >> "${output_wav_scp}"
            fi
        done
    fi
done

sort -k1,1 -u "${output_wav_scp}" -o "${output_wav_scp}"

if [ "$verbose" = true ]; then
    echo "Finished creating ${output_wav_scp} from ${dataset_folder} for popcs dataset"
fi
