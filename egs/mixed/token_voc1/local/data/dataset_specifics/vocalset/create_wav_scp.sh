dataset_folder=
train_scp=
eval_scp=
verbose=false
dataset_tag=vocalset

# Expected directory structure:
# <dataset_folder>
# ├── FULL
# │   ├── female1
# │   │   ├── arpeggios
# │   │   │   ├── belt
# │   │   │   └── ...
# │   │   └── ...
# │   └── ...
# └── ...
# or
# <dataset_folder>
# ├── female1
# │   ├── arpeggios
# │   │   ├── belt
# │   │   └── ...
# │   └── ...
# └── ...

test_singers=(
    female2
    female8
    male3
    male5
    male10
)
train_singers=(
    female1
    female3
    female4
    female5
    female6
    female7
    female9
    male1
    male2
    male4
    male6
    male7
    male8
    male9
    male11
)


# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1

if [ -z "${dataset_folder}" ]; then
    echo "Please set the dataset_folder variable."
    exit 1
fi
if [ -z "${train_scp}" ] || [ -z "${eval_scp}" ]; then
    echo "Please set the train_scp and eval_scp variables."
    exit 1
fi

echo "Creating ${train_scp}, ${eval_scp} from ${dataset_folder} for VocalSet dataset"
mkdir -p "$(dirname "${train_scp}")"
mkdir -p "$(dirname "${eval_scp}")"

if [ "$(basename "${dataset_folder}")" != "FULL" ]; then
    dataset_folder="${dataset_folder}/FULL"
fi
find "${dataset_folder}" -type f -name "*.wav" | sort | while read -r file; do
    utt_id="${file#${dataset_folder}/}" # e.g., "<dataset_folder>/female7/long_tones/trillo/f7_ long_trillo_a.wav" -> "female7/long_tones/trillo/f7_ long_trillo_a.wav"
    singer="${utt_id%%/*}"
    if [[ " ${test_singers[*]} " =~ " ${singer} " ]]; then
        output_wav_scp="${eval_scp}"
    elif [[ " ${train_singers[*]} " =~ " ${singer} " ]]; then
        output_wav_scp="${train_scp}"
    else
        echo "Singer ${singer} is not in the test or train singers list. Please check if the dataset is downloaded correctly."
        exit 1
    fi
    utt_id="${utt_id%.*}" # remove file extension
    utt_id=$(echo "$utt_id" | perl -CS -pe 'chomp; s/\p{Space}/_/g') # replace spaces with underscores
    utt_id="${utt_id//[\/.,\']/_}" # e.g., "female7/long_tones/trillo/f7__long_trillo_a" -> "female7_long_tones_trillo_f7__long_trillo_a"
    echo "${dataset_tag}_${utt_id} ${file}" >> "${output_wav_scp}"
done

sort -k1,1 -u "${train_scp}" -o "${train_scp}"
sort -k1,1 -u "${eval_scp}" -o "${eval_scp}"

if [ "$verbose" = true ]; then
    echo "Finished creating ${train_scp}, ${eval_scp} from ${dataset_folder} for VocalSet dataset"
fi
