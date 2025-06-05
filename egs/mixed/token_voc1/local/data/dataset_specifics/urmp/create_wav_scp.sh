dataset_folder=
output_wav_scp=
verbose=false

# Expected directory structure:
# <dataset_folder>
# ├── 01_Jupiter_vn_vc
# │   ├── AuSep_1_vn_01_Jupiter.wav
# │   ├── AuSep_2_vc_01_Jupiter.wav
# │   └── ...
# ├── 02_Sonata_vn_vn
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

echo "Creating ${output_wav_scp} from ${dataset_folder} for urmp dataset"
mkdir -p "$(dirname "${output_wav_scp}")"
for song in "${dataset_folder}"/*; do
    if [ -d "${song}" ] && [[ $(basename "${song}") =~ ^[0-9]+ ]]; then
        songid=$(basename "${song}" .wav)
        for audio in "${song}"/AuSep_*.wav; do
            if [ -f "${audio}" ]; then
                utt_id="urmp_${songid}_$(basename "${audio}" .wav)"
                echo "${utt_id} ${audio}" >> "${output_wav_scp}"
            fi
        done
    fi
done

sort -k1,1 -u "${output_wav_scp}" -o "${output_wav_scp}"

if [ "$verbose" = true ]; then
    echo "Finished creating ${output_wav_scp} from ${dataset_folder} for urmp dataset"
fi
