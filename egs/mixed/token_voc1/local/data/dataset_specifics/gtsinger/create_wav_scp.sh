dataset_folder=
output_wav_scp=
verbose=false
languages="Chinese Japanese"

# Expected directory structure:
# <dataset_folder>
# ├── Chinese/
# │   ├── ZH-Alto-1/
# │   │   ├── Breathy/
# │   │   │   ├── 不再见/
# │   │   │   │   ├── Breathy_Group/
# │   │   │   │   │   ├── 0000.json
# │   │   │   │   │   ├── 0000.musicxml
# │   │   │   │   │   ├── 0000.wav
# │   │   │   │   │   └── ...
# │   │   │   │   └── ...
# │   │   │   └── ...
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

echo "Creating ${output_wav_scp} from ${dataset_folder} for GTSinger dataset"
mkdir -p "$(dirname "${output_wav_scp}")"
for lang in ${languages}; do
    folder="${dataset_folder}/${lang}"
    if [ ! -d "${folder}" ]; then
        echo "Directory ${folder} does not exist. Please check if dataset is downloaded correctly."
        exit 1
    fi

    if [ "$verbose" = true ]; then
        echo "Processing GTSinger dataset for language: ${lang}"
    fi

    if ! sh local/data/checks/check_empty_dirs.sh "${folder}"; then
        echo "Directory ${folder} contain empty directories. Please check if dataset is downloaded correctly."
        echo "You may need to unzip using `UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip`."
        exit 1
    fi

    find "${folder}" -type f -name "*.wav" | sort | while read -r file; do
        group=$(basename "$(dirname "$file")") # e.g., "<dataset_folder>/Chinese/ZH-Alto-1/Breathy/不再见/Breathy_Group/0000.wav" -> "Breathy_Group"
        if [[ "${group}" == "Paired_Speech_Group" ]]; then
            continue
        fi
        utt_id="${file#${dataset_folder}/}" # e.g., "<dataset_folder>/Japanese/JA-Tenor-1/Vibrato/Heartful Song/Paired_Speech_Group/0003.wav" -> "Japanese/JA-Tenor-1/Vibrato/Heartful Song/Paired_Speech_Group/0003.wav"
        utt_id="${utt_id//\//_}" # e.g., -> "Japanese_JA-Tenor-1_Vibrato_Heartful Song_Paired_Speech_Group_0003.wav"
        utt_id="${utt_id// /_}" # e.g., -> "Japanese_JA-Tenor-1_Vibrato_Heartful_Song_Paired_Speech_Group_0003.wav"
        utt_id="${utt_id//　/_}" # e.g., -> "Japanese_JA-Tenor-1_Vibrato_Heartful_Song_Paired_Speech_Group_0003.wav"
        utt_id="${utt_id%.wav}" # e.g., -> "Japanese_JA-Tenor-1_Vibrato_Heartful_Song_Paired_Speech_Group_0003"
        utt_id="gtsinger_${utt_id//\//_}" # e.g., -> "gtsinger_Japanese_JA-Tenor-1_Vibrato_Heartful_Song_Paired_Speech_Group_0003"
        echo "${utt_id} ${file}" >> "${output_wav_scp}"
    done
done

sort -k1,1 -u "${output_wav_scp}" -o "${output_wav_scp}"

if [ "$verbose" = true ]; then
    echo "Finished creating ${output_wav_scp} from ${dataset_folder} for GTSinger dataset"
fi
