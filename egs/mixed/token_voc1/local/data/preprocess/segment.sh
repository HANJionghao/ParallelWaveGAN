#!/bin/bash

set -euo pipefail

max_wav_duration=15.0
min_wav_duration=0.4
source_wav_scp=wav.scp
output_wav_scp=wav_segment.scp
wav_dump=wav_dump
append=false
verbose=false
segment_mode="multi" # multi or once
remove_short=true

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1

if [ "${append}" = false ]; then
    rm -rf "${wav_dump}"
fi

wav_dump_parent_dir=$(dirname "${wav_dump}")
mkdir -p "${wav_dump_parent_dir}"
tmp_dir="${wav_dump_parent_dir}/tmp$(uuidgen)"
mkdir -p "${tmp_dir}"

while IFS=" " read -r utt wav_path; do
    if [ "${verbose}" = true ]; then
        echo "Processing ${utt} ${wav_path}. Splitting into segments as ${tmp_dir}/${utt}_*.wav"
    fi
    suffix=${wav_path##*.}
    sox "${wav_path}" "${tmp_dir}/${utt}_.${suffix}" silence 1 0.1 1% 1 0.7 1% : newfile : restart
done <"${source_wav_scp}"

if [ "${segment_mode}" = "multi" ]; then
    # segment once more if any segment is longer than max_wav_duration
    for segment in "${tmp_dir}"/*; do
        audio_length=$(soxi -D "${segment}")
        if (($(echo "$audio_length > $max_wav_duration" | bc -l))); then
            if [ "${verbose}" = true ]; then
                echo "Segment ${segment} has length ${audio_length} longer than ${max_wav_duration} seconds. Splitting into smaller segments."
            fi
            sox "${segment}" "${tmp_dir}/$(basename ${segment})" silence 1 0.1 1% 1 0.5 1.5% : newfile : restart
            rm "${segment}"
        else
            filename=$(basename "$segment")
            filename="${filename%.*}000.${filename##*.}"
            mv "${segment}" "${tmp_dir}/${filename}"
        fi
    done

    # segment once more if any segment is longer than max_wav_duration
    for segment in "${tmp_dir}"/*; do
        audio_length=$(soxi -D "${segment}")
        if (($(echo "$audio_length > $max_wav_duration" | bc -l))); then
            if [ "${verbose}" = true ]; then
                echo "Segment ${segment} has length ${audio_length} longer than ${max_wav_duration} seconds. Splitting into smaller segments."
            fi
            sox "${segment}" "${tmp_dir}/$(basename ${segment})" silence 1 0.1 1% 1 0.25 1.5% : newfile : restart
            rm "${segment}"
        else
            filename=$(basename "$segment")
            filename="${filename%.*}000.${filename##*.}"
            mv "${segment}" "${tmp_dir}/${filename}"
        fi
    done

    # segment once more if any segment is longer than max_wav_duration
    for segment in "${tmp_dir}"/*; do
        audio_length=$(soxi -D "${segment}")
        if (($(echo "$audio_length > $max_wav_duration" | bc -l))); then
            # increase silence threshold to 3% and 5% to handle breathing sounds
            if [ "${verbose}" = true ]; then
                echo "Segment ${segment} has length ${audio_length} longer than ${max_wav_duration} seconds. Splitting into smaller segments."
            fi
            sox "${segment}" "${tmp_dir}/$(basename ${segment})" silence 1 0.1 3% 1 0.3 5% : newfile : restart
            rm "${segment}"
        else
            filename=$(basename "$segment")
            filename="${filename%.*}000.${filename##*.}"
            mv "${segment}" "${tmp_dir}/${filename}"
        fi
    done
fi

# remove short audio files
if [ "${remove_short}" = true ]; then
    if [ "${verbose}" = true ]; then
        echo "Removing short audio files less than ${min_wav_duration} seconds for ${wav_dump}."
    fi
    sh local/data/preprocess/remove_short_audio.sh "${tmp_dir}" ${min_wav_duration}
fi

# move the segments to the wav_dump
if [ "${verbose}" = true ]; then
    echo "Moving segments from ${tmp_dir} to ${wav_dump}"
fi
mkdir -p "${wav_dump}"
rsync -a --remove-source-files "${tmp_dir}/" "${wav_dump}/" # NOTE(jhan): use rsync to avoid argument list too long error 
rmdir "${tmp_dir}"

# create the output wav.scp file
if [ "${verbose}" = true ]; then
    echo "Creating ${output_wav_scp} from ${wav_dump}"
fi
mkdir -p $(dirname "${output_wav_scp}")
if [ "${append}" = false ]; then
    > "${output_wav_scp}"
fi

find "$wav_dump" -type f | sort | while IFS= read -r file; do
    filename=$(basename "${file}")
    utt="${filename%.*}"
    file=$(realpath "${file}")
    echo "${utt} ${file}" >> "${output_wav_scp}"
done
