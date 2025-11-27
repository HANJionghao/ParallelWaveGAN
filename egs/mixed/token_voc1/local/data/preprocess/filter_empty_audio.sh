# This script filters out empty audio files in place from a wav.scp file and remove empty audio files.

verbose=false

if [ $# -ne 1 ]; then
    echo "Usage: $0 <wav.scp>"
    exit 1
fi

source_wav_scp=$1
# skip if the file is empty
if [ ! -s "${source_wav_scp}" ]; then
    echo "File ${source_wav_scp} is empty. Skipping."
    exit 0
fi

tmp_wav_scp="$source_wav_scp.$(uuidgen).tmp"

while read -r line; do
    utt_id="${line%% *}" # NOTE(jhan): Assuming the utt_id does not contain spaces
    wav_file="${line#* }"
    if [ ! -f "${wav_file}" ]; then
        echo "File ${wav_file} does not exist. Please check ${source_wav_scp}."
        exit 1
    fi
    audio_length=$(soxi -D "${wav_file}")
    if [ "$(echo "${audio_length} > 0" | bc -l)" -eq 1 ]; then
        echo "${utt_id} ${wav_file}" >>"${tmp_wav_scp}"
    else
        if [ "${verbose}" = true ]; then
            echo "File ${wav_file} is empty. Removing it."
        fi
        rm "${wav_file}"
    fi
done <"${source_wav_scp}"

mv "${tmp_wav_scp}" "${source_wav_scp}"
echo "Filtered empty audio files from ${source_wav_scp}."
