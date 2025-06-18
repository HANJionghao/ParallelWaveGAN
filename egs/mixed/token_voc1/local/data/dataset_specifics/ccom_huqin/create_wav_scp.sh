dataset_folder=
output_wav_scp=
wav_dump=wav_dump/ccom_huqin
verbose=false
utt_prefix=ccom_huqin

# Expected directory structure:
# <dataset_folder>
# ├── Excerpts
# │   └── ...
# └── SinglePT
#     ├── AltoBanhu
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

echo "[INFO] Creating ${output_wav_scp} from ${dataset_folder} for the CCOM-HuQin dataset"

rm -rf "${wav_dump}"
mkdir -p ${wav_dump}

> "${output_wav_scp}"

dataset_folder=$(realpath "${dataset_folder}")

# Excerpts
mkdir -p "${wav_dump}/Excerpts"
find "${dataset_folder}/Excerpts" -type f -name "*.wav" | sort | while read -r file; do
    utt_id="${file#${dataset_folder}/}"
    utt_id="${utt_id//\//_}"
    utt_id=$(echo "$utt_id" | perl -CS -pe 'chomp; s/\p{Space}/_/g') # replace spaces with underscores
    utt_id="${utt_id%.*}"
    sox -V0 "$file" "${wav_dump}/Excerpts/${utt_id}.wav" silence 1 0.2 1% 1 0.7 1% : newfile : restart
done

for segment in "${wav_dump}/Excerpts"/*.wav; do
    audio_length=$(soxi -D "${segment}")
    # truncate further is audio length is longer than 20 seconds
    if (($(echo "$audio_length > 20" | bc -l))); then
        sox "${segment}" "${segment}" silence 1 0.1 1% 1 0.1 1% : newfile : restart
        rm "${segment}"
    else
        filename=$(basename "$segment")
        filename="${filename%.*}000.${filename##*.}"
        mv "${segment}" "${wav_dump}/Excerpts/${filename}"
    fi
done

for file in "${wav_dump}/Excerpts"/*.wav; do
    [ -f "$file" ] || continue
    # check audio is empty
    audio_length=$(soxi -D "${file}")
    if [ "$(echo "${audio_length} > 0" | bc -l)" -eq 1 ]; then
        utt_id=$(basename "${file}" .wav)
        file=$(realpath "${file}")
        echo "${utt_prefix}_${utt_id} ${file}" >> "${output_wav_scp}"
    else
        rm "${file}"
    fi
done

# SinglePT
mkdir -p "${wav_dump}/SinglePT"
for instrument_folder in "${dataset_folder}/SinglePT/"*; do
    [ -d "${instrument_folder}" ] || continue
    instrument=$(basename "${instrument_folder}")

    for skill_folder in "${instrument_folder}"/*; do
        [ -d "${skill_folder}" ] || continue
        skill=$(basename "${skill_folder}")

        for file in "${skill_folder}"/*.wav; do
            duration=$(soxi -D "$file")
            file=$(realpath "$file")
            if [ $(echo "$duration > 0.37" | bc) -eq 1 ]; then
                echo "${utt_prefix}_SinglePT_${instrument}_${skill}_$(basename "${file}" .wav) ${file}" >> "${output_wav_scp}"
            fi
        done
    done
done

LC_ALL=C sort -k1,1 -u "${output_wav_scp}" -o "${output_wav_scp}"

if [ "$verbose" = true ]; then
    echo "[INFO] Finished creating ${output_wav_scp} from ${dataset_folder} for the CCOM-HuQin dataset"
fi
