#!/bin/bash

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;

# basic settings
stage=-1       # stage to start
stop_stage=100 # stage to stop
verbose=1      # verbosity level (lower is less info)
n_gpus=1       # number of gpus in training
n_gpus_eval=0       # number of gpus in training
n_jobs=8       # number of parallel jobs in feature extraction

# NOTE(kan-bayashi): renamed to conf to avoid conflict in parse_options.sh
conf=conf/hifigan_token_16k_nodp_f0.v1.yaml

# directory path setting
raw_dataset_paths=local/data/raw_dataset_paths.csv
espnet_dataset_paths=local/data/espnet_dataset_paths.csv
combined_dataset_paths=local/data/combined_dataset_paths.csv
datasets_to_extract_feats=local/data/combined_dataset_paths.csv # set it to local/data/raw_dataset_paths.csv if espnet files already generated labels

dumpdir=dump           # directory to dump features
datadir=data           # directory to save data
wav_dump=wav_dump # directory to save wav files

# preprocessing and feats extraction setting
audio_ext=flac # audio file extension to be used after resampling
use_gpu_in_feats_extract=true
km_folder=
kmeans_features=
RVQ_layers=1
espnet_path=
local_data_opts=

# training related setting
tag=""     # tag for directory to save model
resume=""  # checkpoint path to resume training
           # (e.g. <path>/<to>/checkpoint-10000steps.pkl)

# decoding related setting
checkpoint="" # checkpoint path to be used for decoding
              # if not provided, the latest one will be used
              # (e.g. <path>/<to>/checkpoint-400000steps.pkl)

train_set="train"       # name of training data directory
dev_set="dev"           # name of development data direcotry
eval_set="test"         # name of evaluation data direcotry

token_text=""
multi_token_files=""    # list of multi token (only used in multi token pattern)
# multi_token_mix_type="sequence" # ["sequence", "frame"], mix type of multi token

use_f0=true                                   # whether to add f0
use_vuv=false                                 # whether to add vuv
use_embedding_feats=false                     # whether to use pretrain feature as input
use_spk_embed=false                           # whether to use speaker embedding
use_class_condition=false                     # whether to use class condition
vc_datadir="" # directory to save vc features TODO(jhan): personal use only, remove this in PR
vc_dumpdir="" # directory to save vc features TODO(jhan): personal use only, remove this in PR
spk_embed_scp_tag="espnet_spk"                # scp file for pre-extracted speaker embeddings
class_scp_tag="class"
pretrained_model="facebook/hubert-base-ls960" # pre-trained model (confirm it on Huggingface)
use_multi_layer=false          # Whether to use multi layer
feat_layer=3                    # Number of total layers for multi layer, specific layer for single layer.
use_gpu_in_preprocess=false # Whether to use GPU in preprocess. Prefer to use GPU for crepe f0 extraction.

fs=16000
subexp="exp"

versa_dir=    # directory of versa repo
versa_python= # python path with versa installed

summary_ref_conf= # reference config file for summary; used to compare with the current config

use_multi_resolution_token=false # Whether to use multi resolution

train_batch_sampler_conf="{}"
dev_batch_sampler_conf="{}"

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

set -euo pipefail

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo "Stage 0: Data preparation"

    if [ "${use_gpu_in_feats_extract}" = true ]; then
        _cmd="${cuda_cmd}"
    else
        _cmd="${decode_cmd}"
    fi
    ./local/data/data.sh \
        --raw_dataset_paths "${raw_dataset_paths}" \
        --espnet_dataset_paths "${espnet_dataset_paths}" \
        --combined_dataset_paths "${combined_dataset_paths}" \
        --datasets_to_extract_feats "${datasets_to_extract_feats}" \
        --processed_datasets_dir "${datadir}_processed" \
        --combined_datadir "${datadir}" \
        --wav_dump "${wav_dump}" \
        --resampled_wav_dump "wav_dump_resampled${fs}" \
        --fs "${fs}" \
        --train_set "${train_set}" \
        --dev_set "${dev_set}" \
        --eval_set "${eval_set}" \
        --audio_ext "${audio_ext}" \
        --km_folder "${km_folder}" \
        --kmeans_features "${kmeans_features}" \
        --RVQ_layers "${RVQ_layers}" \
        --use_gpu "${use_gpu_in_feats_extract}" \
        --stage 1 \
        --stop_stage 100 \
        --cmd "${_cmd}" \
        --espnet_path "${espnet_path}" \
        --append false \
        --clean_up true \
        ${local_data_opts}

    # sort -o ${datadir}/train/wav.scp ${datadir}/train/wav.scp
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Stage 1: Feature extraction"
    # if use multi token, multi_token_files should be provided
    if [ "${use_multi_layer}" = true ]; then
        if [ -z "${multi_token_files}" ]; then
            echo "Valid --multi_token_files is not provided. Please prepare it by yourself."
            exit 1
        fi
    else
        if [ -z "${token_text}" ]; then
            echo "Valid --token_text is not provided. Please prepare it by yourself."
            echo "--token_text have is the path of a kaldi-style text file. Below is an example."
            cat << EOF
----------------------------------
utt_id_1 0 0 0 0 1 1 1 1 2 2 2 2
utt_id_2 0 0 0 0 0 0 3 3 3 3 3 3 5 5 5 5
...
EOF
            exit 1
        fi
    fi
    # extract raw features
    pids=()
    for name in "${train_set}" "${dev_set}" "${eval_set}"; do
    (
        [ ! -e "${dumpdir}/${name}/raw" ] && mkdir -p "${dumpdir}/${name}/raw"
        echo "Feature extraction start. See the progress via ${dumpdir}/${name}/raw/preprocessing.*.log."
        extra_files=
        if [ "${use_spk_embed}" = true ]; then
            extra_files+="${datadir}/${name}/${spk_embed_scp_tag}.scp "
        fi

        if [ "${use_class_condition}" = true ]; then
            extra_files+="${datadir}/${name}/${class_scp_tag}.scp "
        fi
        for file in ${token_files}; do
            if [ ! -f "${datadir}/${name}/${file}" ]; then
                echo "ERROR: ${datadir}/${name}/${file} does not exist."
                exit 1
            fi
            extra_files+="${datadir}/${name}/${file} "
        done

        utils/make_subset_data.sh "${datadir}/${name}" "${n_jobs}" "${dumpdir}/${name}/raw" "${extra_files}"

        _opts=
        if [ "${use_f0}" = true ]; then
            _opts+="--use-f0 "
        fi
        if [ "${use_multi_layer}" = true ]; then
            _opts+="--use-multi-layer "
            _opts+="--feat-layer ${feat_layer} "
            _opts+="--multi-token-files \"${multi_token_files}\" "
            # _opts+="--multi-token-mix-type ${multi_token_mix_type} "
        else
            _opts+="--text ${token_text} "
        fi
        if [ "${use_embedding_feats}" = true ]; then
            _opts+="--use-embedding-feats "
            _opts+="--pretrained-model ${pretrained_model} "
            _opts+="--feat-layer ${feat_layer} "
        fi
        if [ "${use_multi_resolution_token}" = true ]; then
            _opts+="--use-multi-resolution-token "
        fi
        if [ "${use_spk_embed}" = true ]; then
            _opts+="--spk-embed-scp ${dumpdir}/${name}/raw/${spk_embed_scp_tag}.JOB.scp "
        fi
        if [ "${use_class_condition}" = true ]; then
            _opts+="--class-scp ${dumpdir}/${name}/raw/${class_scp_tag}.JOB.scp "
        fi

        # preprocess embedding feature instead of token
        if [ "${use_gpu_in_preprocess}" = true ]; then
            ${cuda_cmd} JOB=1:${n_jobs} --gpu "${n_gpus}" "${dumpdir}/${name}/raw/preprocessing.JOB.log" \
                local/preprocess_token.py \
                    --config "${conf}" \
                    --scp "${dumpdir}/${name}/raw/wav.JOB.scp" \
                    --dumpdir "${dumpdir}/${name}/raw/dump.JOB" \
                    --verbose "${verbose}" ${_opts}
        else
            ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/raw/preprocessing.JOB.log" \
                local/preprocess_token.py \
                    --config "${conf}" \
                    --scp "${dumpdir}/${name}/raw/wav.JOB.scp" \
                    --dumpdir "${dumpdir}/${name}/raw/dump.JOB" \
                    --verbose "${verbose}" ${_opts}
        fi
        echo "Successfully finished feature extraction of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished feature extraction."
fi

if [ -z "${tag}" ]; then
    expdir="${subexp}/${train_set}_$(basename "${conf}" .yaml)"
else
    expdir="${subexp}/${train_set}_${tag}"
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: Network training"
    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    if [ "${n_gpus}" -gt 1 ]; then
        train="python -m parallel_wavegan.distributed.launch --nproc_per_node ${n_gpus} -c parallel-wavegan-train"
    else
        train="parallel-wavegan-train"
    fi
    _opts=
    if [ "${use_f0}" = true ]; then
        _opts+="--use-f0 "
    fi
    if [ "${use_multi_resolution_token}" = true ]; then
        _opts+="--use-multi-resolution-token "
    fi
    if [ "${use_spk_embed}" = true ]; then
        _opts+="--additional-feature-keys spemb "
    fi
    if [ "${use_class_condition}" = true ]; then
        _opts+="--additional-feature-keys class_idx "
    fi
    if [ "${use_vuv}" = true ]; then
        _opts+="--additional-feature-keys vuv "
    fi
    # shellcheck disable=SC2012
    resume="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    echo "Training start. See the progress via ${expdir}/train.log."
    ${cuda_cmd} --gpu "${n_gpus}" "${expdir}/train.log" \
        ${train} \
            --config "${conf}" \
            --train-dumpdir "${dumpdir}/${train_set}/raw" \
            --dev-dumpdir "${dumpdir}/${dev_set}/raw" \
            --train-batch-sampler-conf "${train_batch_sampler_conf}" \
            --dev-batch-sampler-conf "${dev_batch_sampler_conf}" \
            --outdir "${expdir}" \
            --resume "${resume}" \
            --verbose "${verbose}" ${_opts}
    echo "Successfully finished training."
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Network decoding"
    # shellcheck disable=SC2012
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/wav/$(basename "${checkpoint}" .pkl)"
    pids=()
    for name in "${dev_set}" "${eval_set}"; do
    (
        [ ! -e "${outdir}/${name}" ] && mkdir -p "${outdir}/${name}"
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        echo "Decoding start. See the progress via ${outdir}/${name}/decode.log."
        _opts=
        if [ "${use_f0}" = true ]; then
            _opts+="--use-f0 "
        fi
        if [ "${use_multi_resolution_token}" = true ]; then
            _opts+="--use-multi-resolution-token "
        fi
        if [ "${use_spk_embed}" = true ]; then
            _opts+="--additional-feature-keys spemb "
        fi
        if [ "${use_class_condition}" = true ]; then
            _opts+="--additional-feature-keys class_idx "
        fi

        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/decode.log" \
            parallel-wavegan-decode \
                --dumpdir "${dumpdir}/${name}/raw" \
                --checkpoint "${checkpoint}" \
                --outdir "${outdir}/${name}" \
                --verbose "${verbose}" ${_opts} 
        if [ "${use_spk_embed}" = true ] && [ -n "${vc_dumpdir}" ]; then
            [ ! -e "${outdir}_vc/${name}" ] && mkdir -p "${outdir}_vc/${name}"
            ${cuda_cmd} --gpu "${n_gpus}" "${outdir}_vc/${name}/decode.log" \
                parallel-wavegan-decode \
                    --dumpdir "${vc_dumpdir}/${name}/raw" \
                    --checkpoint "${checkpoint}" \
                    --outdir "${outdir}_vc/${name}" \
                    --verbose "${verbose}" ${_opts} 
        fi
        echo "Successfully finished decoding of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished decoding."
fi

if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    echo "Stage 4: Scoring"
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    for dset in ${eval_set}; do
        _data="${datadir}/${dset}"
        _gt_wavscp="${_data}/wav.scp"
        _dir="${expdir}/wav/$(basename "${checkpoint}" .pkl)"
        _gen_wavdir="${_dir}/${dset}"

        # Objective Evaluation - MCD
        if [ -s "${_dir}/MCD_res/mcd_avg_result.txt" ]; then # skip if already exists
            echo "Skip MCD scoring since ${_dir}/MCD_res/mcd_avg_result.txt already exists"
        else
            echo "Begin Scoring for MCD metrics on ${dset}, results are written under ${_dir}/MCD_res"

            mkdir -p "${_dir}/MCD_res"
            python utils/py_utils/evaluate_mcd.py \
                "${_gen_wavdir}" \
                "${_gt_wavscp}" \
                --outdir "${_dir}/MCD_res"
        fi

        # Objective Evaluation - log-F0 RMSE
        if [ -s "${_dir}/F0_res/log_f0_rmse_avg_result.txt" ]; then # skip if already exists
            echo "Skip F0 scoring since ${_dir}/F0_res/log_f0_rmse_avg_result.txt already exists"
        else
            echo "Begin Scoring for F0 related metrics on ${dset}, results are written under ${_dir}/F0_res"

            mkdir -p "${_dir}/F0_res"
            python utils/py_utils/evaluate_f0.py \
                "${_gen_wavdir}" \
                "${_gt_wavscp}" \
                --outdir "${_dir}/F0_res"
        fi

        # Objective Evaluation - semitone ACC
        if [ -s "${_dir}/SEMITONE_res/semitone_acc_avg_result.txt" ]; then # skip if already exists
            echo "Skip SEMITONE scoring since ${_dir}/SEMITONE_res/semitone_acc_avg_result.txt already exists"
        else
            echo "Begin Scoring for SEMITONE related metrics on ${dset}, results are written under ${_dir}/SEMITONE_res"

            mkdir -p "${_dir}/SEMITONE_res"
            python utils/py_utils/evaluate_semitone.py \
                "${_gen_wavdir}" \
                "${_gt_wavscp}" \
                --outdir "${_dir}/SEMITONE_res"
        fi

        # Objective Evaluation - VUV error
        if [ -s "${_dir}/VUV_res/vuv_error_avg_result.txt" ]; then # skip if already exists
            echo "Skip VUV scoring since ${_dir}/VUV_res/vuv_error_avg_result.txt already exists"
        else
            echo "Begin Scoring for VUV related metrics on ${dset}, results are written under ${_dir}/VUV_res"

            mkdir -p "${_dir}/VUV_res"
            python utils/py_utils/evaluate_vuv.py \
                "${_gen_wavdir}" \
                "${_gt_wavscp}" \
                --outdir "${_dir}/VUV_res"
        fi

        #  Objective Evaluation - speaker similarity
        _gen_wavscp="${_dir}/${dset}_wav.scp"
        # create if file does not exist or is empty
        if [ ! -s "${_gen_wavscp}" ]; then
            find ${_gen_wavdir} -name "*.wav" | sort | while read -r line; do
                uttid=$(basename "${line}" _gen.wav)
                echo "${uttid} ${line}"
            done > "${_gen_wavscp}"
            echo "Generated ${_gen_wavscp}"
        fi

        if [ -s "${_dir}/SPK_res/spk_similarity_avg_result.txt" ]; then # skip if already exists
            echo "Skip speaker similarity scoring since ${_dir}/SPK_res/spk_similarity_avg_result.txt already exists"
        else
            echo "Begin Scoring for speaker similarity metrics on ${dset}, results are written under ${_dir}/SPK_res"
            _opts=
            if [ "${n_gpus_eval}" -gt 1 ]; then
                _opts+="--use_gpu true "
                _cmd="${cuda_cmd} --gpu ${n_gpus_eval}"
            else
                _cmd="${decode_cmd}"
            fi
            mkdir -p "${_dir}/SPK_res"
            ${_cmd} "${_dir}/SPK_res/score.log" \
                ${versa_python} ${versa_dir}/versa/bin/scorer.py \
                    --score_config ${versa_dir}/egs/separate_metrics/spk_similarity.yaml \
                    --pred ${_gen_wavscp} \
                    --gt ${_gt_wavscp} \
                    --output_file ${_dir}/SPK_res/versa_utt2speaker_similarity \
                    --io kaldi \
                    ${_opts}

            python local/format_versa_result.py \
                ${_dir}/SPK_res/versa_utt2speaker_similarity \
                'spk_similarity' \
                ${_dir}/SPK_res
        fi

        # Objective Evaluation - SingMOS
        if [ -s "${_dir}/SingMOS_res/singmos_avg_result.txt" ]; then # skip if already exists
            echo "Skip SingMOS scoring since ${_dir}/SingMOS_res/singmos_avg_result.txt already exists"
        else
            echo "Begin Scoring for SingMOS metrics on ${dset}, results are written under ${_dir}/SingMOS_res"

            mkdir -p "${_dir}/SingMOS_res"
            ${cuda_cmd} --gpu "${n_gpus}" "${_dir}/SingMOS_res/score.log" \
                ${versa_python} ${versa_dir}/versa/bin/scorer.py \
                    --score_config ${versa_dir}/egs/separate_metrics/pseudo_mos.yaml \
                    --pred ${_gen_wavscp} \
                    --gt ${_gt_wavscp} \
                    --output_file ${_dir}/SingMOS_res/versa_utt2singmos \
                    --io kaldi \
                    --use_gpu true # force to use gpu for SingMOS eval
            python local/format_versa_result.py \
                ${_dir}/SingMOS_res/versa_utt2singmos \
                'singmos' \
                ${_dir}/SingMOS_res
        fi

        if [ -d "${_dir}_vc" ] && [ -n "${vc_datadir}" ]; then
            _dir_vc="${_dir}_vc"
            _gen_wavdir_vc="${_dir_vc}/${dset}"
            _gt_wavscp_vc="${vc_datadir}/${dset}/vc_wav.scp"

            # Objective Evaluation - log-F0 RMSE
            if [ -s "${_dir}/VC_F0_res/log_f0_rmse_avg_result.txt" ]; then # skip if already exists
                echo "Skip F0 scoring since ${_dir}/VC_F0_res/log_f0_rmse_avg_result.txt already exists"
            else
                echo "Begin Scoring for F0 related metrics on ${dset}, results are written under ${_dir}/VC_F0_res"
                mkdir -p "${_dir}/VC_F0_res"
                python utils/py_utils/evaluate_f0.py \
                    "${_gen_wavdir_vc}" \
                    "${_gt_wavscp}" \
                    --outdir "${_dir}/VC_F0_res"
            fi

            # Objective Evaluation - semitone ACC
            if [ -s "${_dir}/VC_SEMITONE_res/semitone_acc_avg_result.txt" ]; then # skip if already exists
                echo "Skip SEMITONE scoring since ${_dir}/VC_SEMITONE_res/semitone_acc_avg_result.txt already exists"
            else
                echo "Begin Scoring for SEMITONE related metrics on ${dset}, results are written under ${_dir}/VC_SEMITONE_res"
                mkdir -p "${_dir}/VC_SEMITONE_res"
                python utils/py_utils/evaluate_semitone.py \
                    "${_gen_wavdir_vc}" \
                    "${_gt_wavscp}" \
                    --outdir "${_dir}/VC_SEMITONE_res"
            fi

            # Objective Evaluation - VUV error
            if [ -s "${_dir}/VC_VUV_res/vuv_error_avg_result.txt" ]; then # skip if already exists
                echo "Skip VUV scoring since ${_dir}/VC_VUV_res/vuv_error_avg_result.txt already exists"
            else
                echo "Begin Scoring for VUV related metrics on ${dset}, results are written under ${_dir}/VC_VUV_res"
                mkdir -p "${_dir}/VC_VUV_res"
                python utils/py_utils/evaluate_vuv.py \
                    "${_gen_wavdir_vc}" \
                    "${_gt_wavscp}" \
                    --outdir "${_dir}/VC_VUV_res"
            fi

            # Objective Evaluation - speaker similarity
            _gen_wavscp_vc="${_dir_vc}/${dset}_wav.scp"
            # create if file does not exist or is empty
            if [ ! -s "${_gen_wavscp_vc}" ]; then
                find ${_gen_wavdir_vc} -name "*.wav" | sort | while read -r line; do
                    uttid=$(basename "${line}" _gen.wav)
                    echo "${uttid} ${line}"
                done > "${_gen_wavscp_vc}"
                echo "Generated ${_gen_wavscp_vc}"
            fi


            if [ -s "${_dir}/VC_SPK_res/spk_similarity_avg_result.txt" ]; then # skip if already exists
                echo "Skip speaker similarity scoring since ${_dir}/VC_SPK_res/spk_similarity_avg_result.txt already exists"
            else
                echo "Begin Scoring for speaker similarity metrics on ${dset}, results are written under ${_dir}/VC_SPK_res"
                _opts=
                if [ "${n_gpus_eval}" -gt 1 ]; then
                    _opts+="--use_gpu true "
                    _cmd="${cuda_cmd} --gpu ${n_gpus_eval}"
                else
                    _cmd="${decode_cmd}"
                fi
                mkdir -p "${_dir}/VC_SPK_res"
                ${_cmd} "${_dir}/VC_SPK_res/score.log" \
                    ${versa_python} ${versa_dir}/versa/bin/scorer.py \
                        --score_config ${versa_dir}/egs/separate_metrics/spk_similarity.yaml \
                        --pred ${_gen_wavscp_vc} \
                        --gt ${_gt_wavscp_vc} \
                        --output_file ${_dir}/VC_SPK_res/versa_utt2speaker_similarity \
                        --io kaldi \
                        ${_opts}

                python local/format_versa_result.py \
                    ${_dir}/VC_SPK_res/versa_utt2speaker_similarity \
                    'spk_similarity' \
                    ${_dir}/VC_SPK_res
            fi

            # Objective Evaluation - SingMOS
            if [ -s "${_dir}/VC_SingMOS_res/singmos_avg_result.txt" ]; then # skip if already exists
                echo "Skip SingMOS scoring since ${_dir}/VC_SingMOS_res/singmos_avg_result.txt already exists"
            else
                echo "Begin Scoring for SingMOS metrics on ${dset}, results are written under ${_dir}/VC_SingMOS_res"
                mkdir -p "${_dir}/VC_SingMOS_res"
                ${cuda_cmd} --gpu "${n_gpus}" "${_dir}/VC_SingMOS_res/score.log" \
                    ${versa_python} ${versa_dir}/versa/bin/scorer.py \
                        --score_config ${versa_dir}/egs/separate_metrics/pseudo_mos.yaml \
                        --pred ${_gen_wavscp_vc} \
                        --gt ${_gt_wavscp_vc} \
                        --output_file ${_dir}/VC_SingMOS_res/versa_utt2singmos \
                        --io kaldi \
                        --use_gpu true # force to use gpu for SingMOS eval
                python local/format_versa_result.py \
                    ${_dir}/VC_SingMOS_res/versa_utt2singmos \
                    'singmos' \
                    ${_dir}/VC_SingMOS_res
            fi
        fi
    done
else
    echo "Skip the evaluation stages"
fi

if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
    echo "Stage 5: Results summary"
    # summarize results in csv
    summary_csv="summary.csv"
    echo "Summarize results in ${summary_csv}"
    python ./local/update_scoring_summary.py ${expdir} ${summary_ref_conf} ${summary_csv}
fi
echo "Finished."
