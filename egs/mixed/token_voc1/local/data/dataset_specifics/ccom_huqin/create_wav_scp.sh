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
        output_dir="${wav_dump}/SinglePT/${instrument}_${skill}"
        mkdir -p "${output_dir}"

        # Estimate average duration
        estimate_num=3
        files=("${skill_folder}"/*.wav)
        estimate_files=("${files[@]:0:$estimate_num}")
        if [ ${#estimate_files[@]} -lt "$estimate_num" ]; then
            echo "[ERROR] Not enough files in ${skill_folder}. Found ${#estimate_files[@]}. Please check if download is complete."
            exit 1
        fi
        estimated_duration=0
        for file in "${estimate_files[@]}"; do
            duration=$(soxi -D "$file")
            estimated_duration=$(echo "$estimated_duration + $duration" | bc)
        done
        avg_duration=$(echo "$estimated_duration / $estimate_num" | bc -l)
        # Check if avg_duration is a valid number before comparison
        if [[ -n "$avg_duration" && "$avg_duration" =~ ^[0-9]*\.?[0-9]+$ ]] && [ $(echo "$avg_duration > 0.5" | bc) -eq 1 ]; then
            # If average duration is greater than 0.6s, output the files directly
            for file in "${files[@]}"; do
                outfile="${output_dir}/$(basename "${file}")"
                cp "${file}" "${outfile}"
                echo "${utt_prefix}_SinglePT_${instrument}_${skill}_$(basename "${file}" .wav) ${outfile}" >> "${output_wav_scp}"
            done
        else
            # Concatenate files with random shuffle
            shuffled_files=($(shuf -e "${files[@]}"))
            concat_count=$(echo "1 / $avg_duration" | bc -l | awk '{print int($1 + 0.999)}')  # Round up
            if [ "$concat_count" -lt 1 ]; then
                concat_count=1
            fi
            for ((i = 0; i < ${#shuffled_files[@]}; i += concat_count)); do
                concat_files=("${shuffled_files[@]:i:concat_count}")
                if [ ${#concat_files[@]} -gt 0 ]; then
                    # name by concatenated files
                    concatenated=$(basename -a "${concat_files[@]}" | sed 's/\.wav$//' | paste -sd '-')
                    hash_concat=$(echo -n "${concatenated}" | md5sum | cut -c1-16)
                    output_file="${output_dir}/${hash_concat}.wav"
                    sox "${concat_files[@]}" "$output_file"
                    echo "${utt_prefix}_SinglePT_${instrument}_${skill}_${concatenated} ${output_file}" >> "${output_wav_scp}"
                fi
            done
        fi
    done
done

sort -k1,1 -u "${output_wav_scp}" -o "${output_wav_scp}"

if [ "$verbose" = true ]; then
    echo "[INFO] Finished creating ${output_wav_scp} from ${dataset_folder} for the CCOM-HuQin dataset"
fi
