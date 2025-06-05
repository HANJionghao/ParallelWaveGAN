dataset_folder=
output_wav_scp=
languages="chinese"
verbose=false

# http://isophonics.net/SingingVoiceDataset
# Expected directory structure:
# <dataset_folder>
# ├── chinese
# │   ├── F1-九儿-ChineseFolk-LyricalPassionate
# │   │   ├── F1-九儿-ChineseFolk-抒情力量01.mid
# │   │   ├── F1-九儿-ChineseFolk-抒情力量01_score.mid
# │   │   ├── F1-九儿-ChineseFolk-抒情力量01.TextGrid
# │   │   ├── F1-九儿-ChineseFolk-抒情力量01.txt
# │   │   ├── F1-九儿-ChineseFolk-抒情力量01.wav
# │   │   └── ...
# │   ├── F1-九儿-Opera-LyricalPassionate
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

echo "Creating ${output_wav_scp} from ${dataset_folder} for SingStyle111 dataset"
mkdir -p "$(dirname "${output_wav_scp}")"
for lang in ${languages}; do
    folder="${dataset_folder}/${lang}"
    if [ ! -d "${folder}" ]; then
        echo "Directory ${folder} does not exist. Please check if dataset is downloaded correctly."
        exit 1
    fi
    if [ "$verbose" = true ]; then
        echo "Processing SingStyle111 dataset for language: ${lang}"
    fi

    if ! sh local/data/checks/check_empty_dirs.sh "${folder}"; then
        echo "Directory ${folder} contain empty directories. Please check if dataset is downloaded correctly."
        echo "You may need to unzip using `UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip`."
        exit 1
    fi

    find "${folder}" -type f -name "*.wav" | sort | while read -r file; do
        utt_id=$(basename "${file}" .wav) # "<dataset_folder>/chinese/F1-小城故事-Pop-Normal/F1-小城故事-Pop-正常01.wav" -> "F1-小城故事-Pop-正常01"
        if [[ "$utt_id" =~ [0-9]$ ]]; then
            utt_id="singstyle111_${lang}_${utt_id}" # e.g., -> "singstyle111_chinese_F1-小城故事-Pop-正常01"
            echo "${utt_id} ${file}" >> "${output_wav_scp}"
        fi
    done
done

sort -k1,1 -u "${output_wav_scp}" -o "${output_wav_scp}"

if [ "$verbose" = true ]; then
    echo "Finished creating ${output_wav_scp} from ${dataset_folder} for SingStyle111 dataset"
fi
