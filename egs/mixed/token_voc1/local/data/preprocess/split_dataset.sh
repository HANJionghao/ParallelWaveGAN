#  sh local/data/postprocess/split_dataset.sh --source_file "${datadir}/${dataset_tag}/wav_orig.scp" --output_train_file "${datadir}/${dataset_tag}/${train_set}/wav.scp" --output_test_file "${datadir}/${dataset_tag}/${eval_set}/wav.scp" --num_test 1

source_file=wav.scp
output_train_file=train/wav.scp
output_test_file=test/wav.scp
num_test=0
append=false
random=true

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1

tmp_source_file=
if [ "${source_file}" = "${output_train_file}" ] || [ "${source_file}" = "${output_test_file}" ]; then
    tmp_source_file="${source_file}.tmp"
    cp "${source_file}" "${tmp_source_file}"
    source_file="${tmp_source_file}"
fi

mkdir -p "$(dirname "${output_train_file}")"
mkdir -p "$(dirname "${output_test_file}")"

wav_count=$(wc -l < "${source_file}")
num_train=$(($wav_count - $num_test))
if [ "${random}" = true ]; then
    shuf "${source_file}" > "${source_file}.shuf"

    if [ $append = false ]; then
        head -n "${num_train}" "${source_file}.shuf" > "${output_train_file}"
        tail -n "${num_test}" "${source_file}.shuf" > "${output_test_file}"
    else
        head -n "${num_train}" "${source_file}.shuf" >> "${output_train_file}"
        tail -n "${num_test}" "${source_file}.shuf" >> "${output_test_file}"
    fi

    LC_ALL=C sort -k1,1 -u "${output_train_file}" -o "${output_train_file}"
    LC_ALL=C sort -k1,1 -u "${output_test_file}" -o "${output_test_file}"
    rm "${source_file}.shuf"
else
    if [ $append = false ]; then
        head -n "${num_train}" "${source_file}" > "${output_train_file}"
        tail -n "${num_test}" "${source_file}" > "${output_test_file}"
    else
        head -n "${num_train}" "${source_file}" >> "${output_train_file}"
        tail -n "${num_test}" "${source_file}" >> "${output_test_file}"
    fi
fi

if [ -n "${tmp_source_file}" ]; then
    echo "Removing temporary source file: ${tmp_source_file}" # TODO: remove this line
    rm "${tmp_source_file}"
fi
