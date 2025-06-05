dataset_folder=
output_wav_scp=
verbose=false

# Expected directory structure:
# <dataset_folder>
# ├── wav_PT
# │   ├── 01.wav
# │   ├── 02.wav
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

echo "Creating ${output_wav_scp} from ${dataset_folder} for no7singing dataset"
mkdir -p "$(dirname "${output_wav_scp}")"
for song in "${dataset_folder}"/wav_PT/*.wav; do
    if [ -f "${song}" ]; then
        songid=$(basename "${song}" .wav)
        echo "no7singing_PT_${songid} ${song}" >> "${output_wav_scp}"
    fi
done

sort -k1,1 -u "${output_wav_scp}" -o "${output_wav_scp}"

if [ "$verbose" = true ]; then
    echo "Finished creating ${output_wav_scp} from ${dataset_folder} for no7singing dataset"
fi
