dataset_folder=
tr_no_dev_scp=
dev_scp=
eval_scp=
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
if [ -z "${tr_no_dev_scp}" ] || [ -z "${dev_scp}" ] || [ -z "${eval_scp}" ]; then
    echo "Please set the tr_no_dev_scp, dev_scp and eval_scp variables."
    exit 1
fi

dev_list=("13" "14" "26" "28" "39")
eval_list=("01" "16" "17" "27" "44")

echo "Creating ${tr_no_dev_scp}, ${dev_scp} and ${eval_scp} from ${dataset_folder} for no7singing dataset"
mkdir -p "$(dirname "${tr_no_dev_scp}")"
mkdir -p "$(dirname "${dev_scp}")"
mkdir -p "$(dirname "${eval_scp}")"

for song in "${dataset_folder}"/wav_PT/*.wav; do
    if [ -f "${song}" ]; then
        songid=$(basename "${song}" .wav)
        if [[ " ${dev_list[*]} " =~ " ${songid} " ]]; then
            output_wav_scp="${dev_scp}"
        elif [[ " ${eval_list[*]} " =~ " ${songid} " ]]; then
            output_wav_scp="${eval_scp}"
        else
            output_wav_scp="${tr_no_dev_scp}"
        fi
        echo "no7singing_PT_${songid} ${song}" >> "${output_wav_scp}"
    fi
done

sort -k1,1 -u "${tr_no_dev_scp}" -o "${tr_no_dev_scp}"
sort -k1,1 -u "${dev_scp}" -o "${dev_scp}"
sort -k1,1 -u "${eval_scp}" -o "${eval_scp}"

if [ "$verbose" = true ]; then
    echo "Finished creating ${tr_no_dev_scp}, ${dev_scp} and ${eval_scp} from ${dataset_folder} for no7singing dataset"
fi
