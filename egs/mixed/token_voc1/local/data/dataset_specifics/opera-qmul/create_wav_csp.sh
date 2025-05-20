dataset_folder=
output_wav_scp=
verbose=false

# http://isophonics.net/SingingVoiceDataset
# Expected directory structure:
# <dataset_folder>
# ├── monophonic
# │   ├── chinese
# │   │   ├── fem_01
# │   │   │   ├── neg_1.wav
# │   │   │   ├── pos_1.wav
# │   │   │   └── ...
# │   │   ├── fem_02
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

echo "Creating ${output_wav_scp} from ${dataset_folder} for the Singing Voice Audio Dataset from the Centre for Digital Music, Queen Mary, University of London"
mkdir -p "$(dirname "${output_wav_scp}")"
if [ "$verbose" = true ]; then
    echo "Processing opera-qmul dataset for singer: ${singer}, folder: monophonic/chinese"
fi

find "${dataset_folder}/monophonic/chinese" -type f -name "*.wav" | sort | while read -r file; do
    utt_id="${file#${dataset_folder}/}" # e.g., "<dataset_folder>/monophonic/chinese/fem_01/neg_1.wav" -> "monophonic/chinese/fem_01/neg_1.wav"
    utt_id="${utt_id//\//_}" # e.g., -> "monophonic_chinese_fem_01_neg_1.wav"
    utt_id="opera-qmul_${utt_id%.wav}" # e.g., -> "opera-qmul_monophonic_chinese_fem_01_neg_1"
    echo "${utt_id} ${file}" >> "${output_wav_scp}"
done

sort -u "${output_wav_scp}" -o "${output_wav_scp}"

if [ "$verbose" = true ]; then
    echo "Finished creating ${output_wav_scp} from ${dataset_folder} for opera-qmul dataset"
fi
