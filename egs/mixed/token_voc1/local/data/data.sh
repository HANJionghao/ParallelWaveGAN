set -e
set -u
set -o pipefail

datasets_to_extract_feats=local/data/processed_nonespnet_dataset_paths.csv # set it to a processed nonespnet dataset csv if tokens are already generated for espnet datasets
nonespnet_dataset_paths=local/data/nonespnet_dataset_paths.csv
espnet_dataset_paths=local/data/espnet_dataset_paths.csv
combined_dataset_paths=local/data/combined_dataset_paths.csv
processed_datasets_dir=data_processed
combined_datadir=data_all
wav_dump=wav_dump
spemb_dump=espnet_spk_dump
resampled_wav_dump=wav_dump_resampled
train_set="tr_no_dev"
dev_set="dev"
eval_set="eval"
audio_ext=wav
fs=44100
stage=0
stop_stage=100
append=false # whether to append to existing wav.scp files or folders. When set to false, it will overwrite the existing files and remove the existing folders.
clean_up=true

# Token and speaker embedding extraction
espnet_path= # path to espnet repository, will be used to infer token files and speaker embeddings
km_folder=espnet/egs2/mixed/svs2/exp/kmeans
kmeans_features=
RVQ_layers=1
spemb_pretrained_model=espnet/voxcelebs12_rawnet3 # pretrained model for speaker embedding extraction
spemb_toolkit=espnet
spemb_tag=espnet_spk
spemb_resample_package=torchaudio
use_gpu=true
cmd=run.pl

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1

log_info() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO]  $*"
}

log_warn() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [WARN]  $*" >&2
}

log_error() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] $*" >&2
}


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    if [ -f "${nonespnet_dataset_paths}" ]; then
        log_info "Data preprocessing Stage 0: Creating wav.scp files for non-ESPnet datasets from ${nonespnet_dataset_paths}"

        while IFS="," read -r dataset_tag dataset_folder audio_segment_mode; do
            (
                processed_data_subdir="${processed_datasets_dir}/${dataset_tag}"
                mkdir -p "${processed_data_subdir}"

                # Clear existing wav_orig.scp if not appending
                if [ "$append" = false ] && [ -f "${processed_data_subdir}/wav_orig.scp" ]; then
                    >"${processed_data_subdir}/wav_orig.scp"
                fi

                # Run the appropriate wav.scp creation script
                if [ -f "./local/data/dataset_specifics/${dataset_tag}/create_wav_scp.sh" ]; then
                    ./local/data/dataset_specifics/${dataset_tag}/create_wav_scp.sh \
                        --dataset_folder "${dataset_folder}" \
                        --output_wav_scp "${processed_data_subdir}/wav_orig.scp"
                else
                    ./local/data/preprocess/create_wav_scp.sh \
                        --dataset_folder "${dataset_folder}" \
                        --output_wav_scp "${processed_data_subdir}/wav_orig.scp" \
                        --dataset_tag "${dataset_tag}"
                fi
            ) &
        done <"${nonespnet_dataset_paths}"

        # Wait for all dataset processing to finish
        wait
        log_info "Finished creating wav.scp files for all datasets in ${nonespnet_dataset_paths}"
    else
        log_info "Non-ESPnet dataset paths file ${nonespnet_dataset_paths} does not exist. Skipped preprocessing stage 0."
    fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    if [ -f "${nonespnet_dataset_paths}" ]; then
        log_info "Data preprocessing Stage 1: Preprocessing nonespnet datasets from ${nonespnet_dataset_paths}"
        pids=()
        job_names=()
        while IFS="," read -r dataset_tag dataset_folder audio_segment_mode; do
            (
            log_info "Processing dataset: ${dataset_tag} from folder: ${dataset_folder}"
            processed_data_subdir="${processed_datasets_dir}/${dataset_tag}"

            # 1.1. Resample and combine to mono
            log_info "Resampling and combining audio files for dataset: ${dataset_tag}"
            ./local/data/preprocess/resample_to_mono.sh \
                --source_wav_scp "${processed_data_subdir}/wav_orig.scp" \
                --output_wav_scp "${processed_data_subdir}/wav_orig_fs${fs}.scp" \
                --wav_dump "${resampled_wav_dump}/${fs}/${dataset_tag}" \
                --fs "${fs}" \
                --append "${append}" \
                --audio_ext "${audio_ext}"

            # 1.2. Split into train/eval
            mkdir -p "${processed_data_subdir}/${train_set}" "${processed_data_subdir}/${train_set}_with_dev" "${processed_data_subdir}/${dev_set}" "${processed_data_subdir}/${eval_set}"
            test_percent=0.01 # 1% of the total data will be used for evaluation
            total=$(wc -l < "${processed_data_subdir}/wav_orig_fs${fs}.scp")
            num_test=$(echo "($total * $test_percent)/1" | bc)
            if [ "$num_test" -lt 1 ]; then
                num_test=1
            fi

            log_info "Splitting dataset: ${dataset_tag} into ${train_set}_with_dev and ${eval_set} sets with num_test: ${num_test}"
            ./local/data/preprocess/split_dataset.sh \
                --source_file "${processed_data_subdir}/wav_orig_fs${fs}.scp" \
                --output_train_file "${processed_data_subdir}/${train_set}_with_dev/wav.scp.tmp" \
                --output_test_file "${processed_data_subdir}/${eval_set}/wav.scp.tmp" \
                --num_test "${num_test}" \
                --append false

            # 1.3. Segment
            if [ "$audio_segment_mode" != "none" ]; then
                log_info "Segmenting audio files for dataset: ${dataset_tag} with mode: ${audio_segment_mode}"
                for split in $eval_set "${train_set}_with_dev"; do
                    ./local/data/preprocess/segment.sh \
                        --source_wav_scp "${processed_data_subdir}/${split}/wav.scp.tmp" \
                        --output_wav_scp "${processed_data_subdir}/${split}/wav.scp" \
                        --wav_dump "${wav_dump}/${dataset_tag}/${split}" \
                        --append "${append}" \
                        --segment_mode "${audio_segment_mode}" \
                        --remove_short true
                    rm "${processed_data_subdir}/${split}/wav.scp.tmp"
                done
            else
                log_info "No segmentation required for dataset: ${dataset_tag}. Using original wav.scp."
                for split in $eval_set "${train_set}_with_dev"; do
                    mkdir -p "${wav_dump}/${dataset_tag}/${split}"
                    if [ "$append" = false ]; then
                        >"${processed_data_subdir}/${split}/wav.scp"
                    fi
                    while read -r line; do
                        utt_id="${line%% *}"
                        resampled_file="${line#* }"
                        outfile="${wav_dump}/${dataset_tag}/${split}/$(basename "${resampled_file}")"
                        outfile=$(realpath "${outfile}")
                        mv "${resampled_file}" "${outfile}"
                        echo "${utt_id} ${outfile}" >> "${processed_data_subdir}/${split}/wav.scp"
                    done < "${processed_data_subdir}/${split}/wav.scp.tmp"
                    rm "${processed_data_subdir}/${split}/wav.scp.tmp"
                done
            fi

            # 1.4. Split train into dev and train
            train_dev_total=$(wc -l < "${processed_data_subdir}/${train_set}_with_dev/wav.scp")
            num_dev=$(echo "($train_dev_total * 0.05)/1" | bc)
            if [ "$num_dev" -lt 1 ]; then
                num_dev=1
            elif [ "$num_dev" -gt 50 ]; then
                num_dev=50 # Limit to 50 utterances for development set
            fi
            log_info "Splitting ${train_set}_with_dev set into ${train_set} and ${dev_set} sets for dataset: ${dataset_tag} with num_dev: ${num_dev}"
            ./local/data/preprocess/split_dataset.sh \
                --source_file "${processed_data_subdir}/${train_set}_with_dev/wav.scp" \
                --output_train_file "${processed_data_subdir}/${train_set}/wav.scp" \
                --output_test_file "${processed_data_subdir}/${dev_set}/wav.scp" \
                --num_test "${num_dev}" \
                --append false
            rm -rf "${processed_data_subdir}/${train_set}_with_dev"

            # 1.5. Remove empty wavs
            for split in $dev_set $eval_set $train_set; do
                ./local/data/preprocess/filter_empty_audio.sh "${processed_data_subdir}/${split}/wav.scp"
            done

            # 1.6. Clean up to save space
            if [ "$clean_up" = true ]; then
                if [ -d "${resampled_wav_dump}/${fs}/${dataset_tag}" ]; then
                    rm -rf "${resampled_wav_dump}/${fs}/${dataset_tag}"
                    rm -f "${processed_data_subdir}/wav_orig_fs${fs}.scp"
                fi
            fi
            ) &
            pids+=($!)
            job_names+=("${dataset_tag}")
        done < "${nonespnet_dataset_paths}"

        fail_count=0
        for i in "${!pids[@]}"; do
            pid="${pids[$i]}"
            job="${job_names[$i]}"
            if ! wait "$pid"; then
                log_error "Data preprocessing Stage 1 for dataset ${job} failed."
                fail_count=$((fail_count + 1))
            fi
        done

        if [ "$fail_count" -gt 0 ]; then
            log_error "Data preprocessing Stage 1 failed for ${fail_count}/${#pids[@]} dataset(s). Please check the logs."
            exit 1
        fi
        log_info "Finished preprocessing all datasets in ${nonespnet_dataset_paths}"
    else
        log_info "Non-ESPnet dataset paths file ${nonespnet_dataset_paths} does not exist. Skipped preprocessing stage 1."
    fi
fi


if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    if [[ -n "$combined_dataset_paths" ]]; then
        log_info "Data preprocessing Stage 2: Combining metadata for all datasets and saving to ${combined_dataset_paths}"
        # Overwrite existing combined_dataset_paths if append is false
        if [ "$append" = false ]; then
            >"${combined_dataset_paths}"
        fi

        if [ -f "${nonespnet_dataset_paths}" ]; then
            while IFS="," read -r dataset_tag dataset_folder; do
                processed_data_subdir="${processed_datasets_dir}/${dataset_tag}"
                # stage 2.1. Validate data files
                for split in $dev_set $eval_set $train_set; do
                    if [ ! -f "${processed_data_subdir}/${split}/wav.scp" ]; then
                        log_error "File ${processed_data_subdir}/${split}/wav.scp does not exist. Please check if paths are correct in ${nonespnet_dataset_paths}."
                        exit 1
                    fi
                    ./local/data/checks/check_duplicate_lines.sh "${processed_data_subdir}/${split}/wav.scp"
                done
                # Stage 2.2. Add nonespnet dataset to combined dataset paths
                echo "${dataset_tag},${processed_data_subdir},${train_set},${dev_set},${eval_set}" >>"${combined_dataset_paths}"
            done <"${nonespnet_dataset_paths}"
            log_info "Added nonespnet dataset paths ${nonespnet_dataset_paths} to combined dataset paths ${combined_dataset_paths}"
        fi

        if [ -f "${espnet_dataset_paths}" ]; then
            cat "${espnet_dataset_paths}" >>"${combined_dataset_paths}"
            log_info "Added espnet dataset paths ${espnet_dataset_paths} to combined dataset paths ${combined_dataset_paths}"
        fi
    else
        log_info "Combined dataset paths file ${combined_dataset_paths} is not set. Skipped preprocessing stage 2."
    fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    if [ -f "${datasets_to_extract_feats}" ]; then
        log_info "Data preprocessing Stage 3: Generating labels for datasets in ${datasets_to_extract_feats}"
        if [ "$use_gpu" = true ]; then
            _nj=1 # You could set this to a higher value if you have submit _nj * len(kmeans_features) jobs at once
        else
            _nj=8
        fi

        # process the combined dataset paths
        pids=()
        job_names=()
        while IFS="," read -r dataset_tag dataset_folder _train_set _dev_set _eval_set; do
            (
                if [ ! -d "${dataset_folder}" ]; then
                    log_error "Dataset folder ${dataset_folder} does not exist. Please check if paths are correct in ${datasets_to_extract_feats}."
                    exit 1
                fi

                # for split in $_dev_set $_eval_set $_train_set; do
                for split in $_dev_set; do
                    data_split_dir="${dataset_folder}/${split}"
                    if [ ! -d "${data_split_dir}" ]; then
                        log_error "Dataset split ${data_split_dir} does not exist. Please check if paths are correct in ${datasets_to_extract_feats}."
                        exit 1
                    fi

                    # create token files
                    ./local/data/extract_feats/dump_tokens.sh --data_split_dir "${data_split_dir}" --km_folder "${km_folder}" --kmeans_features "${kmeans_features}" --RVQ_layers ${RVQ_layers} --audio_sample_rate ${fs} --use_gpu "${use_gpu}" --cmd "${cmd}" --nj "${_nj}" --espnet_path "${espnet_path}" --audio_ext "${audio_ext}" || exit 1

                    # create espnet speaker files
                    if [ "$append" = false ]; then
                        rm -rf "${spemb_dump}/${dataset_tag}/${split}"
                    else
                        log_error "Appending to existing spemb_dump directory is not supported. Please set append to false or adjust the script."
                        exit 1
                    fi
                    ./local/data/extract_feats/dump_spembs.sh --data_split_dir "${data_split_dir}" --pretrained_model "${spemb_pretrained_model}" --toolkit "${spemb_toolkit}" --spk_embed_tag "${spemb_tag}" --resample_package "${spemb_resample_package}" --use_gpu "${use_gpu}" --cmd "${cmd}" --nj "${_nj}" --espnet_path "${espnet_path}" --spemb_dump_dir "${spemb_dump}/${dataset_tag}/${split}" || exit 1
                done

                wait # Wait for all splits within this dataset to finish
            ) &
            pids+=($!)
            job_names+=("${dataset_tag}")
        done <"${datasets_to_extract_feats}"

        fail_count=0
        for i in "${!pids[@]}"; do
            pid="${pids[$i]}"
            job="${job_names[$i]}"
            if ! wait "$pid"; then
                log_error "Data preprocessing Stage 3 for dataset ${job} failed."
                fail_count=$((fail_count + 1))
            fi
        done

        if [ "$fail_count" -gt 0 ]; then
            log_error "Data preprocessing Stage 3 failed for ${fail_count}/${#pids[@]} dataset(s). Please check the logs."
            exit 1
        fi
    else
        log_info "Dataset paths file ${datasets_to_extract_feats} does not exist. Skipped preprocessing stage 3."
    fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    if [ -f "${combined_dataset_paths}" ]; then
        log_info "Data preprocessing Stage 4: Combining datasets in ${combined_dataset_paths} into ${combined_datadir}"
        if [ "$append" = false ]; then
            rm -rf "${combined_datadir}"
        fi
        splits=("${dev_set}" "${eval_set}" "${train_set}")
        mkdir -p "${combined_datadir}/${train_set}" "${combined_datadir}/${dev_set}" "${combined_datadir}/${eval_set}"
        while IFS="," read -r dataset_tag dataset_folder _train_set _dev_set _eval_set; do
            if [ ! -d "${dataset_folder}" ]; then
                continue
            fi
            orig_splits=("${_dev_set}" "${_eval_set}" "${_train_set}")
            files=("wav.scp" "${spemb_tag}.scp")
            key_file="wav.scp"
            token_files=()
            for kmeans_feature in ${kmeans_features}; do
                kmeans_feature_type=$(echo "${kmeans_feature}" | cut -d'/' -f1)
                nclusters=$(echo "${kmeans_feature}" | cut -d'/' -f3)
                if [ ${kmeans_feature} = "mfcc" ]; then # MFCC has no layer
                    layer=
                else
                    layer=$(echo "${kmeans_feature}" | cut -d'/' -f2)
                fi
                token_file="pseudo_labels_${kmeans_feature_type}_${layer}_km${nclusters}.txt"
                files+=("${token_file}")
                token_files+=("${token_file}")
            done

            for i in "${!orig_splits[@]}"; do
                orig_split="${orig_splits[$i]}"
                out_split="${splits[$i]}"

                if [ ! -d "${dataset_folder}/${orig_split}" ]; then
                    log_error "Dataset split ${dataset_folder}/${orig_split} does not exist. Please check if paths are correct in ${combined_dataset_paths}."
                    exit 1
                fi
                # if token files are not empty, check if they are aligned
                if [ -n "${token_files}" ]; then
                    python local/data/checks/check_token_alignment.py --data_folder "${dataset_folder}/${orig_split}" --token_files "${token_files[@]}"
                    log_info "Checked alignment for token files in ${dataset_folder}/${orig_split}"
                fi

                for file in "${files[@]}"; do
                    if [ ! -f "${dataset_folder}/${orig_split}/${file}" ]; then
                        log_error "File ${dataset_folder}/${orig_split}/${file} does not exist. Please check if paths are correct in ${combined_dataset_paths}."
                        exit 1
                    fi
                    ./local/data/checks/check_duplicate_lines.sh "${dataset_folder}/${orig_split}/${file}"
                    if [ "${file}" != "${key_file}" ]; then
                        ./local/data/checks/check_utt_alignment.sh "${dataset_folder}/${orig_split}/${file}" "${dataset_folder}/${orig_split}/${key_file}"
                        log_info "Checked alignment for ${file} in ${dataset_folder}/${orig_split}"
                    fi
                    cat "${dataset_folder}/${orig_split}/${file}" >>"${combined_datadir}/${out_split}/${file}"
                done
            done
        done <"${combined_dataset_paths}"
    else
        log_info "Combined dataset paths file ${combined_dataset_paths} does not exist. Skipped preprocessing stage 4."
    fi
fi
