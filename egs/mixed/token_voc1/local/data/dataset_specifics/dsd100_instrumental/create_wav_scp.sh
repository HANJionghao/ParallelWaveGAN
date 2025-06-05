dataset_folder=
output_wav_scp=
verbose=false
utt_prefix=dsd100_instrumental

# Expected directory structure:
# <dataset_folder>
# ├── Mixtures
# │   └── ...
# └── Sources
#     └── ...


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

rm -f "${output_wav_scp}"

echo "Creating ${output_wav_scp} from ${dataset_folder} for DSD100 dataset"
mkdir -p "$(dirname "${output_wav_scp}")"

find "${dataset_folder}/Sources" -type f -name "*.wav" ! -name "vocals.wav" | while read -r audio; do
    utt_id="${audio#${dataset_folder}/}"
    utt_id="${utt_id//\//_}"
    utt_id=$(echo "$utt_id" | perl -CS -pe 'chomp; s/\p{Space}/_/g') # replace spaces with underscores
    utt_id="${utt_id%.*}"
    echo "${utt_prefix}_${utt_id} ${audio}" >> "${output_wav_scp}"
done

sort -k1,1 -u "${output_wav_scp}" -o "${output_wav_scp}"

if [ "$verbose" = true ]; then
    echo "Finished creating ${output_wav_scp} from ${dataset_folder} for urmp dataset"
fi
