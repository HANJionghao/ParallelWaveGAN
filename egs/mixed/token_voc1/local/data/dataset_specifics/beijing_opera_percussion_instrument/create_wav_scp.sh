dataset_folder=
output_wav_scp=
wav_dump=wav_dump/beijing_opera_percussion_instrument
verbose=false
utt_prefix=beijing_opera_percussion_instrument

# Expected directory structure:
# <dataset_folder>
# ├── 205972__ajaysm__daluo-01.wav
# ├── 205973__ajaysm__bangu-06.wav
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

echo "Creating ${output_wav_scp} from ${dataset_folder} for the Beijing Opera percussion instrument dataset"

mkdir -p ${wav_dump}
tmp_dir=$(mktemp -d "${wav_dump}/tmp.XXXXXX")

for audio in "${dataset_folder}"/*.wav; do
    if [ -f "${audio}" ]; then
        filename=$(basename "${audio}" .wav)
        instrument="${filename##*__}"
        instrument="${instrument%%-*}"
        segid="${filename##*-}"
        mkdir -p "${tmp_dir}/${instrument}"
        sox -V0 "${audio}" "${tmp_dir}/${instrument}/${segid}.wav" silence 1 0.2 0.3% reverse silence 1 0.2 0.3% reverse
    fi
done

for instrument_dir in "${tmp_dir}"/*; do
    instrument=$(basename "${instrument_dir}")
    # concatenate all wav files in the instrument directory
    sox "${instrument_dir}"/*.wav "${wav_dump}/${instrument}.wav"
    echo "${utt_prefix}_${instrument} ${wav_dump}/${instrument}.wav" >> "${output_wav_scp}"
done

sort -k1,1 -u "${output_wav_scp}" -o "${output_wav_scp}"
rm -rf "${tmp_dir}"

if [ "$verbose" = true ]; then
    echo "Finished creating ${output_wav_scp} from ${dataset_folder} for the Beijing Opera percussion instrument dataset"
fi
