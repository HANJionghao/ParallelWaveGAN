dataset_folder=
tr_no_dev_scp=
dev_scp=
eval_scp=
verbose=false
utt_prefix=popcs

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
if [ -z "${tr_no_dev_scp}" ] || [ -z "${dev_scp}" ] || [ -z "${eval_scp}" ]; then
    echo "Please set the tr_no_dev_scp, dev_scp and eval_scp variables."
    exit 1
fi

dev_list=("桥边姑娘" "九张机")
eval_list=("会呼吸的痛" "我不难过")

echo "Creating ${tr_no_dev_scp}, ${dev_scp} and ${eval_scp} from ${dataset_folder} for popcs dataset"
mkdir -p "$(dirname "${tr_no_dev_scp}")"
mkdir -p "$(dirname "${dev_scp}")"
mkdir -p "$(dirname "${eval_scp}")"

for song_dir in "${dataset_folder}"/popcs-*; do
    if [ -d "${song_dir}" ]; then
        song=$(basename "${song_dir}")
        song=${song#popcs-}
        if [[ " ${dev_list[*]} " =~ " ${song} " ]]; then
            output_wav_scp="${dev_scp}"
        elif [[ " ${eval_list[*]} " =~ " ${song} " ]]; then
            output_wav_scp="${eval_scp}"
        else
            output_wav_scp="${tr_no_dev_scp}"
        fi
        for audio in "${song_dir}"/*.wav; do
            if [ -f "${audio}" ]; then
                audioid=$(basename "${audio}" .wav)
                echo "${utt_prefix}_${song}_${audioid} ${audio}" >> "${output_wav_scp}"
            fi
        done
    fi
done

LC_ALL=C sort -k1,1 -u "${tr_no_dev_scp}" -o "${tr_no_dev_scp}"
LC_ALL=C sort -k1,1 -u "${dev_scp}" -o "${dev_scp}"
LC_ALL=C sort -k1,1 -u "${eval_scp}" -o "${eval_scp}"

if [ "$verbose" = true ]; then
    echo "Finished creating ${tr_no_dev_scp}, ${dev_scp} and ${eval_scp} from ${dataset_folder} for popcs dataset"
fi
