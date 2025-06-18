wav_scp=$1
fs=$2

while read -r line; do
    utt_id="${line%% *}" # NOTE(jhan): Assuming the utt_id does not contain spaces
    wav_file="${line#* }"
    if [ ! -f "${wav_file}" ]; then
        echo "File ${wav_file} does not exist. Please check ${wav_scp}."
        exit 1
    fi
    actual_fs=$(soxi -r "${wav_file}")
    if [ "${actual_fs}" -ne "${fs}" ]; then
        echo "${wav_scp} has incorrect sampling rate. ${wav_file} has ${actual_fs} Hz, expected ${fs} Hz."
        exit 1
    fi
done <"${wav_scp}"
echo "All files in ${wav_scp} have the correct sampling rate of ${fs} Hz."
