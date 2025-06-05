dataset_folder=
output_wav_scp=
verbose=false

# Expected directory structure:
# <dataset_folder>
# ├── jvs001/
# │   ├── song_common/
# │   │   ├── wav/
# │   │   │   ├── modified_grouped.wav
# │   │   │   ├── modified.wav
# │   │   │   └── raw.wav
# │   │   └── mpd/
# │   │       └── ...
# │   └── song_unique/
# │       └── wav/
# │           └── raw.wav
# ├── jvs002/
# │   ├── song_common/
# │   │   ├── wav/
# │   │   │   ├── modified_grouped.wav
# │   │   │   ├── modified.wav
# │   │   │   └── raw.wav
# │   │   └── mpd/
# │   │       └── ...
# │   └── song_unique/
# │       └── wav/
# │           └── raw.wav
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

echo "Creating ${output_wav_scp} from ${dataset_folder} for JVS dataset"
mkdir -p "$(dirname "${output_wav_scp}")"
for singer_dir in "${dataset_folder}"/jvs*; do
    if [ -d "${singer_dir}" ]; then
        if [ "$verbose" = true ]; then
            echo "Processing JVS dataset for singer: ${singer_dir}"
        fi
        singer=$(basename "${singer_dir}")
        echo "${singer}_common_raw ${singer_dir}/song_common/wav/modified.wav" >> "${output_wav_scp}"
        echo "${singer}_unique_raw ${singer_dir}/song_unique/wav/raw.wav" >> "${output_wav_scp}"
    fi
done

sort -k1,1 -u "${output_wav_scp}" -o "${output_wav_scp}"

if [ "$verbose" = true ]; then
    echo "Finished creating ${output_wav_scp} from ${dataset_folder} for JVS dataset"
fi
