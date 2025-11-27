#!/bin/bash

set -euo pipefail

max_wav_sox_duration=15.0
min_wav_filter_duration=0.37
max_wav_filter_duration=30.0
source_wav_scp=wav.scp
output_wav_scp=wav_segment.scp
wav_dump=wav_dump
append=false
verbose=false
segment_mode="multi" # multi or once
remove_short=true
remove_long=true

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1

if [ "${append}" = false ]; then
    rm -rf "${wav_dump}"
fi

wav_dump_parent_dir=$(dirname "${wav_dump}")

mkdir -p "${wav_dump_parent_dir}"
wav_dump_tmp_dir=$(mktemp -d "${wav_dump_parent_dir}/tmp_XXXXXX")
mkdir -p "$(dirname "${output_wav_scp}")"

if [ "${verbose}" = true ]; then
    echo "Segmenting audio files in ${source_wav_scp}"
fi

while IFS=" " read -r utt wav_path; do
    file_tmp_dir=$(mktemp -d "${wav_dump_parent_dir}/tmp_${utt}_XXXXXX")
    if [ "${verbose}" = true ]; then
        echo "Processing ${utt} ${wav_path}. Splitting into segments in ${file_tmp_dir}"
    fi
    suffix=${wav_path##*.}
    sox "${wav_path}" "${file_tmp_dir}/${utt}_.${suffix}" silence 1 0.1 1% 1 0.7 1% : newfile : restart

    if [ "${segment_mode}" = "multi" ]; then
        # segment multiple times until the audio is shorter than max_wav_sox_duration
        for pass in 1 2 3; do
            find "${file_tmp_dir}" -type f | while read -r segment; do
                audio_length=$(soxi -D "${segment}")
                if (( $(echo "$audio_length > $max_wav_sox_duration" | bc -l) )); then
                    case $pass in
                        1) sox "${segment}" "${file_tmp_dir}/$(basename "${segment}")" silence 1 0.1 1% 1 0.5 1.5% : newfile : restart ;;
                        2) sox "${segment}" "${file_tmp_dir}/$(basename "${segment}")" silence 1 0.1 1% 1 0.25 1.5% : newfile : restart ;;
                        3) sox "${segment}" "${file_tmp_dir}/$(basename "${segment}")" silence 1 0.1 3% 1 0.3 5% : newfile : restart ;;
                    esac
                    rm "${segment}"
                else
                    filename=$(basename "$segment")
                    filename="${filename%.*}000.${filename##*.}"
                    mv "${segment}" "${file_tmp_dir}/${filename}"
                fi
            done
        done
    fi

    # filter out audio by duration
    if [ "${remove_short}" = true ] || [ "${remove_long}" = true ]; then
        if [ "${remove_short}" = false ]; then
            min_wav_filter_duration="none"
        fi
        if [ "${remove_long}" = false ]; then
            max_wav_filter_duration="none"
        fi
        if [ "${verbose}" = true ]; then
            echo "Filtering out audio by duration for ${utt}: min ${min_wav_filter_duration}s to max ${max_wav_filter_duration}s"
        fi
        sh local/data/preprocess/remove_audio_by_duration.sh "${file_tmp_dir}" ${min_wav_filter_duration} ${max_wav_filter_duration}
    fi

    # sync the result to wav_dump_tmp_dir
    if [ "${verbose}" = true ]; then
        echo "Moving segments from ${file_tmp_dir} to ${wav_dump_tmp_dir}"
    fi
    rsync -a --remove-source-files "${file_tmp_dir}/" "${wav_dump_tmp_dir}/"  # NOTE(jhan): use rsync to avoid argument list too long error
    rmdir "${file_tmp_dir}"
done <"${source_wav_scp}"

# rename wav_dump_tmp_dir to wav_dump
mv "${wav_dump_tmp_dir}" "${wav_dump}"

# create the output wav.scp file
if [ "${verbose}" = true ]; then
    echo "Creating ${output_wav_scp} from ${wav_dump}"
fi
mkdir -p $(dirname "${output_wav_scp}")

> "${output_wav_scp}.tmp"

find "$wav_dump" -type f | while IFS= read -r file; do
    filename=$(basename "${file}")
    uttid="${filename%.*}"
    echo "${uttid} $(realpath "${file}")" >> "${output_wav_scp}.tmp"
done

if [ "${append}" = false ]; then
    > "${output_wav_scp}"
fi
cat "${output_wav_scp}.tmp" >> "${output_wav_scp}"
rm "${output_wav_scp}.tmp"

# remove duplicates & sort
LC_ALL=C sort -k1,1 -u "${output_wav_scp}" -o "${output_wav_scp}"
