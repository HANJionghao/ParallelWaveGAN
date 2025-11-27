dataset_folder=
train_scp=
eval_scp=
verbose=false

# Expected directory structure:
# <dataset_folder>
# ├── bossa_nova
# │   ├── dongurikorokoro
# │   │   ├── alto.wav
# │   │   ├── bass.wav
# │   │   ├── lead_vocal.wav
# │   │   └── ...
# │   └── ...
# ├── enka
# │   └── ...
# └── ...

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

eval_song_list=(
    anomachikonomachi
    dongurikorokoro
    hiraitahiraita
    katatsumuri
    koinobori
    otamajakushi
    usagitokame
    nakayoshikomichi
    sunayama
    inu
)

echo "Creating ${train_scp} and ${eval_scp} from ${dataset_folder} for jaCappella dataset"
mkdir -p "$(dirname "${train_scp}")"
mkdir -p "$(dirname "${eval_scp}")"

for subset_dir in "${dataset_folder}"/*; do
    if [ -d "${subset_dir}" ]; then
        subset=$(basename "${subset_dir}")
        if [ "$verbose" = true ]; then
            echo "Processing jaCappella dataset for subset: ${subset}"
        fi
        for song_dir in "${subset_dir}"/*; do
            song=$(basename "${song_dir}")
            # check if song is in test_song_list
            if [[ ! " ${eval_song_list[*]} " =~ " ${song} " ]]; then
                output_wav_scp="${train_scp}"
            else
                output_wav_scp="${eval_scp}"
            fi
            voice_part=vocal_percussion
            audio="${song_dir}/${voice_part}.wav"
            if [ ! -f "${audio}" ]; then
                echo "Error: ${audio} does not exist. Please check folder structure or missing files."
                exit 1
            fi
            utt_id=jacappella_${subset}_${song}_${voice_part}
            echo "${utt_id} ${audio}" >> "${output_wav_scp}"
        done
    fi
done

sort -k1,1 -u "${output_wav_scp}" -o "${output_wav_scp}"

if [ "$verbose" = true ]; then
    echo "Finished creating ${output_wav_scp} from ${dataset_folder} for jaCappella dataset"
fi
