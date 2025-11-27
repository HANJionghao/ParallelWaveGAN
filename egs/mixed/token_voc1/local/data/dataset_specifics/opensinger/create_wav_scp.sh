dataset_folder=
tr_no_dev_scp=
dev_scp=
eval_scp=
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
if [ -z "${tr_no_dev_scp}" ] || [ -z "${dev_scp}" ] || [ -z "${eval_scp}" ]; then
    echo "Please set the tr_no_dev_scp, dev_scp and eval_scp variables."
    exit 1
fi

dev_list=(
    # gender singerid song segid
    "WomanRaw 47 左边 50"
    "WomanRaw 47 左边 51"
    "WomanRaw 47 左边 52"
    "WomanRaw 46 喜欢你 0"
    "WomanRaw 46 喜欢你 1"
    "WomanRaw 46 喜欢你 2"
    "WomanRaw 46 喜欢你 3"
    "WomanRaw 46 喜欢你 4"
    "WomanRaw 46 喜欢你 5"
    "ManRaw 25 一路向北 24"
    "ManRaw 25 一路向北 25"
    "ManRaw 25 一路向北 26"
    "ManRaw 1 鼓楼 14"
    "ManRaw 1 鼓楼 15"
    "ManRaw 1 鼓楼 16"
)
eval_list=(
    "WomanRaw 47 左边 53"
    "WomanRaw 47 左边 54"
    "WomanRaw 47 左边 55"
    "WomanRaw 46 喜欢你 6"
    "WomanRaw 46 喜欢你 7"
    "WomanRaw 46 喜欢你 8"
    "WomanRaw 46 喜欢你 9"
    "WomanRaw 46 喜欢你 10"
    "ManRaw 25 一路向北 27"
    "ManRaw 25 一路向北 28"
    "ManRaw 25 一路向北 29"
    "ManRaw 25 一路向北 30"
    "ManRaw 1 鼓楼 17"
    "ManRaw 1 鼓楼 18"
    "ManRaw 1 鼓楼 19"
)

echo "Creating ${tr_no_dev_scp}, ${dev_scp} and ${eval_scp} from ${dataset_folder} for OpenSinger dataset"
mkdir -p "$(dirname "${tr_no_dev_scp}")"
mkdir -p "$(dirname "${dev_scp}")"
mkdir -p "$(dirname "${eval_scp}")"

for gender_dir in "${dataset_folder}"/*; do
    if [ -d "${gender_dir}" ]; then
        gender=$(basename "${gender_dir}")
        for song_dir in "${gender_dir}"/*; do
            if [ "$verbose" = true ]; then
                echo "Processing OpenSinger dataset for gender: ${gender}, folder: $(basename "${song_dir}")"
            fi
            for wav in "${song_dir}"/*.wav; do
                filename=$(basename "$wav" .wav)
                IFS='_' read -r singerid song segid <<< "${filename}"
                if [[ " ${dev_list[*]} " =~ " ${gender} ${singerid} ${song} ${segid} " ]]; then
                    output_wav_scp="${dev_scp}"
                elif [[ " ${eval_list[*]} " =~ " ${gender} ${singerid} ${song} ${segid} " ]]; then
                    output_wav_scp="${eval_scp}"
                else
                    output_wav_scp="${tr_no_dev_scp}"
                fi
                utt="opensinger_${gender}$(printf "%02d" "$singerid")_${song}_$(printf "%02d" "$segid")"
                echo "${utt} ${wav}" >> "${output_wav_scp}" 
            done
        done
    fi
done

sort -k1,1 -u "${output_wav_scp}" -o "${output_wav_scp}"

if [ "$verbose" = true ]; then
    echo "Finished creating ${output_wav_scp} from ${dataset_folder} for OpenSinger dataset"
fi
