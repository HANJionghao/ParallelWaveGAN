set -e
set -u
set -o pipefail

datasets_to_extract_feats=local/data/processed_raw_dataset_paths.csv # set it to a processed raw dataset csv if tokens are already generated for espnet datasets
raw_dataset_paths=local/data/raw_dataset_paths.csv
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
audio_ext=flac
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

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    if [ -f "${raw_dataset_paths}" ]; then
        echo "Data preprocessing Stage 0: Creating wav.scp files for raw datasets"
        while IFS="," read -r dataset_tag dataset_folder audio_segment_mode; do
            echo "Creating wav.scp for dataset: ${dataset_tag}"
            processed_data_subdir="${processed_datasets_dir}/${dataset_tag}"
            mkdir -p "${processed_data_subdir}"
            # 1. create wav.scp file
            if [ "$append" = false ] && [ -f "${processed_data_subdir}/wav_orig.scp" ]; then
                > "${processed_data_subdir}/wav_orig.scp"
            fi
            ./local/data/dataset_specifics/${dataset_tag}/create_wav_csp.sh --dataset_folder "${dataset_folder}" --output_wav_scp "${processed_data_subdir}/wav_orig.scp"
        done <"${raw_dataset_paths}"
    else
        echo "Raw dataset paths file ${raw_dataset_paths} does not exist. Skipped preprocessing stage 0."
    fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    if [ -f "${raw_dataset_paths}" ]; then
        echo "Data preprocessing Stage 1: Preprocessing raw datasets from ${raw_dataset_paths}"
        while IFS="," read -r dataset_tag dataset_folder audio_segment_mode; do
            echo "Preprocessing dataset: ${dataset_tag}"
            processed_data_subdir="${processed_datasets_dir}/${dataset_tag}"
            # Stage 1.1. Resample and combine to mono with sox, rename file by ${utt_id}.wav
            ./local/data/preprocess/resample_to_mono.sh --source_wav_scp "${processed_data_subdir}/wav_orig.scp" --output_wav_scp "${processed_data_subdir}/wav_orig_fs${fs}.scp" --wav_dump "${resampled_wav_dump}/${fs}/${dataset_tag}" --fs "${fs}" --append "${append}" --audio_ext "${audio_ext}"
            # Stage 1.2. Split the wav files into train vs test (i.e., the last song); dev will be split after segmenting
            mkdir -p "${processed_data_subdir}/${train_set}" "${processed_data_subdir}/${dev_set}" "${processed_data_subdir}/${eval_set}"
            ./local/data/preprocess/split_dataset.sh --source_file "${processed_data_subdir}/wav_orig_fs${fs}.scp" --output_train_file "${processed_data_subdir}/${train_set}/wav.scp.tmp" --output_test_file "${processed_data_subdir}/${eval_set}/wav.scp.tmp" --num_test 1 --append false
            # Stage 1.3. Segment wav files, name file by ${utt_id}_${segment_id}
            if [ "$audio_segment_mode" != "none" ]; then
                for split in $eval_set $train_set; do
                    ./local/data/preprocess/segment.sh --source_wav_scp "${processed_data_subdir}/${split}/wav.scp.tmp" --output_wav_scp "${processed_data_subdir}/${split}/wav.scp" --wav_dump "${wav_dump}/${dataset_tag}/${split}" --append "${append}" --segment_mode "${audio_segment_mode}" --remove_short true
                    rm "${processed_data_subdir}/${split}/wav.scp.tmp"
                done
            fi
            # Stage 1.4. Split the train set into tr_no_dev and dev sets
            ./local/data/preprocess/split_dataset.sh --source_file "${processed_data_subdir}/${train_set}/wav.scp" --output_train_file "${processed_data_subdir}/${train_set}/wav.scp" --output_test_file "${processed_data_subdir}/${dev_set}/wav.scp" --num_test 50 --append "${append}"
            # Stage 1.5. Remove empty wav files to avoid errors in audio loading
            for split in $dev_set $eval_set $train_set; do
                ./local/data/preprocess/filter_empty_audio.sh "${processed_data_subdir}/${split}/wav.scp"
            done
            # Stage 1.6. Remove resampled unsegmented wav files if clean_up is true
            if [ "$clean_up" = true ]; then
                if [ -d "${resampled_wav_dump}" ]; then
                    rm -rf "${resampled_wav_dump}"
                    rm ${processed_data_subdir}/wav_orig_fs${fs}.scp
                fi
            fi
        done <"${raw_dataset_paths}"
    else
        echo "Raw dataset paths file ${raw_dataset_paths} does not exist. Skipped preprocessing stage 1."
    fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    if [[ -n "$combined_dataset_paths" ]]; then
        echo "Data preprocessing Stage 2: Combining metadata for all datasets and saving to ${combined_dataset_paths}"
        # Overwrite existing combined_dataset_paths if append is false
        if [ "$append" = false ]; then
            > "${combined_dataset_paths}"
        fi

        if [ -f "${raw_dataset_paths}" ]; then
            while IFS="," read -r dataset_tag dataset_folder; do
                processed_data_subdir="${processed_datasets_dir}/${dataset_tag}"
                # stage 2.1. Validate data files
                for split in $dev_set $eval_set $train_set; do
                    ./local/data/checks/check_duplicate_lines.sh "${processed_data_subdir}/${split}/wav.scp"
                done
                # 顺便加上check是否文件间有lines重复的
                # Stage 2.2. Add raw dataset to combined dataset paths
                echo "${dataset_tag},${processed_data_subdir},${train_set},${dev_set},${eval_set}" >> "${combined_dataset_paths}"
            done <"${raw_dataset_paths}"
            echo "Added raw dataset paths ${raw_dataset_paths} to combined dataset paths ${combined_dataset_paths}"
        fi

        if [ -f "${espnet_dataset_paths}" ]; then
            cat "${espnet_dataset_paths}" >> "${combined_dataset_paths}"
            echo "Added espnet dataset paths ${espnet_dataset_paths} to combined dataset paths ${combined_dataset_paths}"
        fi
    else
        echo "Combined dataset paths file ${combined_dataset_paths} is not set. Skipped preprocessing stage 2."
    fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    if [ -f "${datasets_to_extract_feats}" ]; then
        echo "Data preprocessing Stage 3: Generating labels for datasets in ${datasets_to_extract_feats}"
        if [ "$use_gpu" = true ]; then
            _nj=1
        else
            _nj=8
        fi

        # process the combined dataset paths
        while IFS="," read -r dataset_tag dataset_folder _train_set _dev_set _eval_set; do
            if [ ! -d "${dataset_folder}" ]; then
                echo "Dataset folder ${dataset_folder} does not exist. Please check if paths are correct in ${datasets_to_extract_feats}."
                exit 1
            fi

            for split in $_dev_set $_eval_set $_train_set; do
                if [ ! -d "${dataset_folder}/${split}" ]; then
                    echo "Dataset split ${dataset_folder}/${split} does not exist. Please check if paths are correct in ${datasets_to_extract_feats}."
                    exit 1
                fi
                # create token files
                ./local/data/extract_feats/dump_tokens.sh --data_split_dir "${dataset_folder}/${split}" --km_folder "${km_folder}" --kmeans_features "${kmeans_features}" --RVQ_layers ${RVQ_layers} --audio_sample_rate ${fs} --use_gpu "${use_gpu}" --cmd "${cmd}" --nj "${_nj}" --espnet_path "${espnet_path}" --audio_ext "${audio_ext}"

                # create espnet speaker files
                if [ "$append" = false ]; then
                    rm -rf "${spemb_dump}/${dataset_tag}/${split}"
                else
                    echo "Appending to existing spemb_dump directory is not supported. Please set append to false or adjust the script."
                    exit 1
                fi
                ./local/data/extract_feats/dump_spembs.sh --data_split_dir "${dataset_folder}/${split}" --pretrained_model "${spemb_pretrained_model}" --toolkit "${spemb_toolkit}" --spk_embed_tag "${spemb_tag}" --resample_package "${spemb_resample_package}" --use_gpu "${use_gpu}" --cmd "${cmd}" --nj "${_nj}" --espnet_path "${espnet_path}" --spemb_dump_dir "${spemb_dump}/${dataset_tag}/${split}"
            done
        done <"${datasets_to_extract_feats}"
    else
        echo "Dataset paths file ${datasets_to_extract_feats} does not exist. Skipped preprocessing stage 3."
    fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    if [ -f "${combined_dataset_paths}" ]; then
        echo "Data preprocessing Stage 4: Combining datasets in ${combined_dataset_paths} into ${combined_datadir}"
        mkdir -p "${combined_datadir}/${train_set}" "${combined_datadir}/${dev_set}" "${combined_datadir}/${eval_set}"
        while IFS="," read -r dataset_tag dataset_folder _train_set _dev_set _eval_set; do
            if [ ! -d "${dataset_folder}" ]; then
                continue
            fi

            for file in wav.scp "${spemb_tag}.scp"; do
                for split in $_dev_set $_eval_set $_train_set; do
                    if [ ! -f "${dataset_folder}/${split}/${file}" ]; then
                        echo "File ${dataset_folder}/${split}/${file} does not exist. Please check if paths are correct in ${datasets_to_extract_feats}."
                        exit 1
                    fi
                done
                cat "${dataset_folder}/${_train_set}/${file}" >> "${combined_datadir}/${train_set}/${file}"
                cat "${dataset_folder}/${_dev_set}/${file}" >> "${combined_datadir}/${dev_set}/${file}"
                cat "${dataset_folder}/${_eval_set}/${file}" >> "${combined_datadir}/${eval_set}/${file}"

                # Validate combined dataset files
                for split in $dev_set $eval_set $train_set; do
                    ./local/data/checks/check_duplicate_lines.sh "${combined_datadir}/${split}/${file}"
                done
            done

            for kmeans_feature in ${kmeans_features}; do
                kmeans_feature_type=$(echo "${kmeans_feature}" | cut -d'/' -f1)
                nclusters=$(echo "${kmeans_feature}" | cut -d'/' -f3)
                if [ ${kmeans_feature} = "mfcc" ]; then # MFCC has no layer
                    layer=
                else
                    layer=$(echo "${kmeans_feature}" | cut -d'/' -f2)
                fi
                token_file="pseudo_labels_${kmeans_feature_type}_${layer}_km${nclusters}.txt"
                for split in $_dev_set $_eval_set $_train_set; do
                    if [ ! -f "${dataset_folder}/${split}/${token_file}" ]; then
                        echo "File ${dataset_folder}/${split}/${token_file} does not exist. Please check if paths are correct in ${datasets_to_extract_feats}."
                        exit 1
                    fi
                    cat "${dataset_folder}/${split}/${token_file}" >> "${combined_datadir}/tokens/${token_file}"
                done

                # Validate combined dataset files
                ./local/data/checks/check_duplicate_lines.sh "${combined_datadir}/tokens/${token_file}"
            done
        done <"${combined_dataset_paths}"
    else
        echo "Combined dataset paths file ${combined_dataset_paths} does not exist. Skipped preprocessing stage 4."
    fi
fi
