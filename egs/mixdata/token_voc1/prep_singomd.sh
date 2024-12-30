#!/bin/bash

# Copyright 2024 Yuxun Tang
#  MIT License (https://opensource.org/licenses/MIT)

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;

# basic settings
stage=-1       # stage to start
stop_stage=100 # stage to stop
verbose=1      # verbosity level (lower is less info)
n_gpus=1       # number of gpus in training
n_jobs=8      # number of parallel jobs in feature extraction

# NOTE(kan-bayashi): renamed to conf to avoid conflict in parse_options.sh
conf=conf/hifigan_token_16k_nodp_f0.v1.yaml

# directory path setting
db_root=db_root # direcotry including wavfiles (MODIFY BY YOURSELF)
                          # each wav filename in the directory should be unique
                          # ├──  mixdata
                          #    ├── path/to/database1
                          #         ├── utt_1.wav
                          #         ├── utt_2.wav
                          #         │   ...
                          #         └── utt_N.wav
                          #    ├── path/to/database
                          #    ...
dumpdir=dump           # directory to dump features
dumpdir_token=dump_token

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
use_f0=false                    # whether to add f0 
use_embedding_feats=true      # whether to use pretrain feature as input
pretrained_model="facebook/hubert-base-ls960"      # pre-trained model (confirm it on Huggingface)
use_multi_layer=true
emb_layer=6

fs=16000
subexp="exp"

store_feature=false             # store model intermediate representation
storedir=feat_store             # directory

use_cluster_token=true
nclusters=1024
portion=1.0
feature_type=emb

use_multi_resolution=false
rs_list=

skip_score=false

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

set -euo pipefail

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo "Stage 0: Data preparation"
    if [ ! -e "${db_root}" ]; then
        echo "ERROR: data source does not exist."
    	exit 1
    fi
    mkdir -p data
    mkdir -p data/${dev_set}
    mkdir -p data/${eval_set}
    mkdir -p data/${train_set}
    > data/${dev_set}/wav.scp
    > data/${eval_set}/wav.scp
    > data/${train_set}/wav.scp
    > data/${dev_set}/utt2num_samples
    > data/${eval_set}/utt2num_samples
    > data/${train_set}/utt2num_samples
    mkdir -p wav_dump
    echo -e "Please make sure fs=${fs} is right sample rate for model.\n" 

    for db_dir in "$db_root"/*/; do
        dbname=$(basename "$db_dir")
        # if [ $dbname == "opencpop" ]; then
        echo "database path: $db_dir"
        echo "database name: $dbname"
        
        python local/data_prep.py ${db_dir} \
            --dbname ${dbname} \
            --wav_dumpdir wav_dump \
            --sr ${fs} \

        dev_num=50
        eval_num=200
        train_num=$(($(wc -l < data/database/$dbname/wav.scp) - dev_num - eval_num))

        aggregate_database() {
            local filename=$1

            sort -o data/database/$dbname/$filename data/database/$dbname/$filename

            head -n $dev_num data/database/$dbname/$filename >> data/${dev_set}/$filename
            head -n $((dev_num + eval_num)) data/database/$dbname/$filename > data/${eval_set}/$filename.tmp
            tail -n $eval_num data/${eval_set}/$filename.tmp >> data/${eval_set}/$filename
            rm data/${eval_set}/$filename.tmp
            tail -n $train_num data/database/$dbname/$filename >> data/${train_set}/$filename
        }

        aggregate_database "wav.scp"
        aggregate_database "utt2num_samples"

        echo "$dbname finshed!"
        # else
        #     echo "skip $dbname"
        # fi
    done
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Stage 1: Embedding Feature extraction"
    # extract raw features
    pids=()
    # for name in "${train_set}" "${dev_set}" "${eval_set}"; do
    for name in "${train_set}"; do
    (
        [ ! -e "${dumpdir}/${name}/raw" ] && mkdir -p "${dumpdir}/${name}/raw"
        echo "Feature extraction start. See the progress via ${dumpdir}/${name}/raw/preprocessing.*.log."
        utils/make_subset_data.sh "data/${name}" "${n_jobs}" "${dumpdir}/${name}/raw"

        _opts=
        if [ ${use_f0} == true ]; then
            _opts+="--use-f0 "
        fi
        if [ ${use_multi_layer} == "true" ]; then
            _opts+="--use-multi-layer "
        fi
        if [ ${use_embedding_feats} == "true" ]; then
            _opts+="--use-embedding-feats "
            _opts+="--pretrained-model ${pretrained_model} "
            _opts+="--emb-layer ${emb_layer} "
        fi

        # preprocess embedding feature instead of token
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/raw/preprocessing.JOB.log" \
            local/preprocess_token.py \
                --config "${conf}" \
                --scp "${dumpdir}/${name}/raw/wav.JOB.scp" \
                --dumpdir "${dumpdir}/${name}/raw/dump.JOB" \
                --verbose "${verbose}" ${_opts}
        echo "Successfully finished feature extraction of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished feature extraction."
fi

if [ -z "${tag}" ]; then
    expdir="${subexp}/${train_set}_opencpop_$(basename "${conf}" .yaml)"
else
    expdir="${subexp}/${train_set}_opencpop_$(basename "${conf}" .yaml)_${tag}"
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: Network training / Adatper training"
    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    if [ "${n_gpus}" -gt 1 ]; then
        train="python -m parallel_wavegan.distributed.launch --nproc_per_node ${n_gpus} -c parallel-wavegan-train"
    else
        train="parallel-wavegan-train"
    fi
    _opts=
    if [ ${use_f0} == true ]; then
        _opts+="--use-f0 "
    fi
    # shellcheck disable=SC2012
    resume="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    echo "Training start. See the progress via ${expdir}/train.log."
    ${cuda_cmd} --gpu "${n_gpus}" "${expdir}/train.log" \
        ${train} \
            --config "${conf}" \
            --train-dumpdir "${dumpdir}/${train_set}/raw" \
            --dev-dumpdir "${dumpdir}/${dev_set}/raw" \
            --outdir "${expdir}" \
            --resume "${resume}" \
            --verbose "${verbose}" ${_opts}
    echo "Successfully finished training."
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Network decoding / Embedding feature extracting"
    # shellcheck disable=SC2012
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/wav/$(basename "${checkpoint}" .pkl)"
    pids=()
    for name in "${train_set}"; do
    # for name in "${dev_set}" "${eval_set}" "${train_set}"; do
    (
        [ ! -e "${outdir}/${name}" ] && mkdir -p "${outdir}/${name}"
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        echo "Decoding start. See the progress via ${outdir}/${name}/decode.log."
        _opts=
        if [ ${use_f0} == true ]; then
            _opts+="--use-f0 "
        fi
        if [ ${store_feature} == true ]; then
            _opts+="--store-feature "
            _opts+="--storedir ${storedir}/${name} "
        fi
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/decode.log" \
            parallel-wavegan-decode \
                --dumpdir "${dumpdir}/${name}/raw" \
                --checkpoint "${checkpoint}" \
                --outdir "${outdir}/${name}" \
                --verbose "${verbose}" ${_opts}      
        echo "Successfully finished decoding of ${name} set."
    ) & 
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished decoding."
fi

if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ] && [ ${use_cluster_token} = true ]; then
    echo "Stage 4: Cluster embedding feature to obtain token feature (K-means)"
    
    _opts=

    if [ ${store_feature} == true ]; then
        _opts+="--featdir $storedir "
    else
        _opts+="--featdir $dumpdir "
    fi

    utils/perform_kmeans.sh \
        --sub_stage 3 --sub_stop_stage 3 \
        --feature_type ${feature_type} \
        --conf ${conf} \
        --nclusters ${nclusters} \
        --portion ${portion} \
        --use_multi_resolution ${use_multi_resolution} ${_opts}
    
else
    echo "Skip stage for clustering token"
fi

if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ] && [ $skip_score == false ]; then
    echo "Stage 5: Scoring Model / Adapter"
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    for dset in ${eval_set}; do
        _data="data/${dset}"
        _gt_wavscp="${_data}/wav.scp"
        _dir="${expdir}/wav/$(basename "${checkpoint}" .pkl)"
        _gen_wavdir="${_dir}/${dset}"

        # Objective Evaluation - MCD
        echo "Begin Scoring for MCD metrics on ${dset}, results are written under ${_dir}/MCD_res"

        mkdir -p "${_dir}/MCD_res"
        python utils/py_utils/evaluate_mcd.py \
            ${_gen_wavdir} \
            ${_gt_wavscp} \
            --outdir "${_dir}/MCD_res"

        # Objective Evaluation - log-F0 RMSE
        echo "Begin Scoring for F0 related metrics on ${dset}, results are written under ${_dir}/F0_res"

        mkdir -p "${_dir}/F0_res"
        python utils/py_utils/evaluate_f0.py \
            ${_gen_wavdir} \
            ${_gt_wavscp} \
            --outdir "${_dir}/F0_res"

        # # Objective Evaluation - semitone ACC
        # echo "Begin Scoring for SEMITONE related metrics on ${dset}, results are written under ${_dir}/SEMITONE_res"

        # mkdir -p "${_dir}/SEMITONE_res"
        # python utils/py_utils/evaluate_semitone.py \
        #     ${_gen_wavdir} \
        #     ${_gt_wavscp} \
        #     --outdir "${_dir}/SEMITONE_res"

        #     # Objective Evaluation - VUV error
        # echo "Begin Scoring for VUV related metrics on ${dset}, results are written under ${_dir}/VUV_res"

        # mkdir -p "${_dir}/VUV_res"
        # python utils/py_utils/evaluate_vuv.py \
        #     ${_gen_wavdir} \
        #     ${_gt_wavscp} \
        #     --outdir "${_dir}/VUV_res"

    done
else
    echo "Skip the evaluation stages"
fi

echo "Finished"