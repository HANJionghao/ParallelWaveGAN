set -e
set -u
set -o pipefail

espnet_path=/ocean/projects/cis210027p/jhan7/cartoon_voice/model/espnet_dev/
data_split_dir= # where wav.scp files are located
km_folder=/ocean/projects/cis210027p/jhan7/cartoon_voice/model/espnet_dev/egs2/mixed/svs2/exp/kmeans
kmeans_features= # e.g., "hubert_large_ll60k/6/1024 wavlm_large/23/1024 wavlm_large/6/1024"
RVQ_layers=1
audio_sample_rate=
audio_ext=wav # Currently, only wav is supported for pwg
cmd=
use_gpu=true
nj=1
verbose=false

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1

mkdir -p "${data_split_dir}"
data_split_dir=$(readlink -f "${data_split_dir}")

if ${use_gpu}; then
    _cmd="${cmd} --gpu 1"
else
    _cmd="${cmd} --gpu 0"
fi

(
    # go to a espnet folder with pyscripts and scripts
    cd "${espnet_path}"/egs2/ami/asr1 || exit 1
    . ./path.sh || exit 1

    mkdir -p "${data_split_dir}/logs"

    if [ ! -f "${data_split_dir}/utt2num_samples" ]; then
        echo "Creating utt2num_samples..."
        # create utt2num_samples
        scripts/audio/format_wav_scp.sh --nj 1 --cmd "${cmd}" \
            --audio-format "${audio_ext}" \
            "${data_split_dir}/wav.scp" "${data_split_dir}/logs" || exit 1
        cp "${data_split_dir}/logs/utt2num_samples" "${data_split_dir}/utt2num_samples"
    else
        echo "utt2num_samples already exists."
    fi

    if [ "${nj}" -gt 1 ]; then
        split_scps=""
        for n in $(seq ${nj}); do
            split_scps+=" ${data_split_dir}/logs/inference_kmeans.${n}.scp"
        done
        utils/split_scp.pl "${data_split_dir}/wav.scp" ${split_scps}
    else
        cp "${data_split_dir}/wav.scp" "${data_split_dir}/logs/inference_kmeans.1.scp"
    fi

    for n in $(seq ${nj}); do
        awk '(FILENAME==ARGV[1]){utt2num[$1]=$2} (FILENAME==ARGV[2]){print($1, utt2num[$1])}' \
            "${data_split_dir}/utt2num_samples" ${data_split_dir}/logs/inference_kmeans.${n}.scp \
            >${data_split_dir}/logs/utt2num_samples.${n}
    done

    # reach each feature
    pids=()

    for kmeans_feature in $kmeans_features; do
    (
        nclusters=$(echo $kmeans_feature | cut -d'/' -f3)

        if [ ${kmeans_feature} = "mfcc" ]; then # MFCC has no layer
            kmeans_feature_type=$(echo "${kmeans_feature}" | cut -d/ -f1)
            layer=
            kmeans_feature_conf="{type=mfcc}"
        else
            kmeans_feature_type=$(echo "${kmeans_feature}" | cut -d/ -f1)
            layer=$(echo "${kmeans_feature}" | cut -d/ -f2)
            if [ ${kmeans_feature_type} = "mert" ]; then
                kmeans_feature_conf="{type=mert,conf={fs=24000,multilayer_feature=False,layer=${layer},download_path=${mert_url}}}"
            elif [ ${kmeans_feature_type} = "encodec" ]; then
                kmeans_feature_conf="{type=encodec,conf={fs=48000,bandwidth=12,multilayer_feature=False,layer=${layer},download_path=${encodec_url}}}"
            elif [ ${kmeans_feature_type} = "contentvec" ]; then
                kmeans_feature_conf="{type=contentvec,conf={layer=${layer}}}"
            # contentvec2
            elif [ ${kmeans_feature_type} = "contentvec2" ]; then
                kmeans_feature_conf="{type=contentvec2,conf={layer=${layer}}}"
            elif [ ${kmeans_feature_type} != "multi" ]; then
                s3prl_conf="{upstream=${kmeans_feature_type}}"
                kmeans_feature_conf="{type=s3prl,conf={s3prl_conf=${s3prl_conf},download_dir=ckpt,multilayer_feature=False,layer=${layer}}}"
            fi
        fi

        km_path="${km_folder}/${kmeans_feature_type}_${layer}_${nclusters}clusters/km_${nclusters}.mdl"
        if [ ${kmeans_feature_type} = "contentvec2" ]; then
            km_path="${km_folder}/contentvec_${layer}_${nclusters}clusters/km_${nclusters}.mdl"
        fi

        mkdir -p "${data_split_dir}/logs/pseudo_labels_${kmeans_feature_type}_${layer}"

        echo "Processing ${kmeans_feature} for ${data_split_dir}"
        ${_cmd} JOB=1:${nj} "${data_split_dir}/logs/pseudo_labels_${kmeans_feature_type}_${layer}/inference_km${nclusters}.JOB.log" \
            PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
            python pyscripts/feats/dump_km_label.py \
                --in_filetype sound \
                --online_feature_extract true \
                --feature_conf "'${kmeans_feature_conf}'" \
                --audio_sample_rate "${audio_sample_rate}" \
                --km_path "${km_path}" \
                --RVQ_layers "${RVQ_layers}" \
                --out_filetype "mat" \
                --use_gpu ${use_gpu} \
                --utt2num_samples "${data_split_dir}/logs/utt2num_samples.JOB" \
                --batch_bins 1 \
                "scp:${data_split_dir}/logs/inference_kmeans.JOB.scp" \
                "ark,t:${data_split_dir}/logs/pseudo_labels_${kmeans_feature_type}_${layer}/km${nclusters}.JOB.txt" || exit 1

        for n in $(seq ${nj}); do
            if [ ! -f "${data_split_dir}/logs/pseudo_labels_${kmeans_feature_type}_${layer}/km${nclusters}.${n}.txt" ]; then
                echo "File ${data_split_dir}/logs/pseudo_labels_${kmeans_feature_type}_${layer}/km${nclusters}.${n}.txt not generated."
                exit 1
            fi
            cat "${data_split_dir}/logs/pseudo_labels_${kmeans_feature_type}_${layer}/km${nclusters}.${n}.txt" || exit 1
        done | sed 's/ \[ \| \]//g' | LC_ALL=C sort -k1,1 -u >"${data_split_dir}/pseudo_labels_${kmeans_feature_type}_${layer}_km${nclusters}.txt" || exit 1
    ) &
    pids+=($!)
    done
)
