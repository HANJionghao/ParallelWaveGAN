source_wav_scp=wav.scp
output_wav_scp=wav16k.scp
wav_dump=wav_dump16k
fs=16000
append=false
audio_ext=flac
resample_tool=sox

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1

if [ $append = false ]; then
    echo "Emptying existing wav dump directory ${wav_dump} and output wav scp ${output_wav_scp}"
    >"${output_wav_scp}"
    rm -rf "${wav_dump}"
fi

if [ ! -f "${source_wav_scp}" ]; then
    echo "File ${source_wav_scp} does not exist. Please check the path."
    exit 1
fi

echo "Resampling and combining to mono with sox ${source_wav_scp} -> ${output_wav_scp}. Sampling rate: ${fs} Hz. Wav dump dir: ${wav_dump}"
mkdir -p "${wav_dump}"
if [ "${resample_tool}" = "sox" ]; then
    while read -r line; do
        utt_id="${line%% *}" # NOTE(jhan): Assuming the utt_id does not contain spaces
        wav_file="${line#* }"
        if [ ! -f "${wav_file}" ]; then
            echo "File ${wav_file} does not exist. Please check ${source_wav_scp}."
            exit 1
        fi
        outfile="${wav_dump}/${utt_id}.${audio_ext}"
        sox "${wav_file}" -r "${fs}" -b 16 -c 1 "${outfile}"
        echo "${utt_id} $(realpath ${outfile})" >>"${output_wav_scp}"
    done <"${source_wav_scp}"
elif [ "${resample_tool}" = "torchaudio" ]; then # TODO(jhan): not tested
    # Check if torchaudio is installed
    if ! python -c "import torchaudio" &>/dev/null; then
        echo "torchaudio is not installed. Please install it first."
        exit 1
    fi
    python ./local/data/preprocess/resample_wav_scp_torchaudio.py --source_wav_scp "${source_wav_scp}" --output_wav_scp "${output_wav_scp}" --wav_dump "${wav_dump}" --fs "${fs}" --audio_ext "${audio_ext}"
else
    echo "Unsupported resample tool: ${resample_tool}. Please use sox or torchaudio."
    exit 1
fi
