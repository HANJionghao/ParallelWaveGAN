#  sh local/data/postprocess/split_dataset.sh --source_file "${datadir}/${dataset_tag}/wav_orig.scp" --output_train_file "${datadir}/${dataset_tag}/${train_set}/wav.scp" --output_test_file "${datadir}/${dataset_tag}/${eval_set}/wav.scp" --num_test 1

source_file=wav.scp
output_train_file=train/wav.scp
output_test_file=test/wav.scp
num_test=1
append=false

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1

tmp_source_file=
if [ "${source_file}" = "${output_train_file}" ] || [ "${source_file}" = "${output_test_file}" ]; then
    tmp_source_file="${source_file}.tmp"
    cp "${source_file}" "${tmp_source_file}"
    source_file="${tmp_source_file}"
fi

if [ $append = false ]; then
    > "${output_train_file}"
    > "${output_test_file}"
fi

wav_count=$(wc -l < "${source_file}")
num_train=$(($wav_count - $num_test))
head -n "${num_train}" "${source_file}" >> "${output_train_file}"
tail -n "${num_test}" "${source_file}" >> "${output_test_file}"

if [ -n "${tmp_source_file}" ]; then
    echo "Removing temporary source file: ${tmp_source_file}" # TODO: remove this line
    rm "${tmp_source_file}"
fi
