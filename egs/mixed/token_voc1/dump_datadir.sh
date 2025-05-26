#!/bin/bash
# create dump folder for datadir

# variables to be set
datadir=data/human_tok-audio_corvid_spemb_16k_0203
dumpdir=dump16k/human_contentvec-audio_corvid_spemb_16k_0203
token_files="pseudo_labels_contentvec_12_km1024.txt"
token_folder_name="tokens"
train_set=tr_no_dev
dev_set=dev
eval_set=eval

# values that are normally unchanged
use_spk_embed=true
conf=conf/hifigan_token_16k_nodp_f0_spemb.v1.yaml
use_gpu_in_preprocess=false

. utils/parse_options.sh || exit 1;

echo "train_set: ${train_set}"

echo "Dump datadir: ${datadir} to dumpdir: ${dumpdir}, with token_files: ${token_files}"

# if token_files is a list of files, set use_multi_layer to true, else false

if [[ "$token_files" =~ \  ]]; then
    use_multi_layer=true
else
    use_multi_layer=false
fi

token_folder="${datadir}/${token_folder_name}"
if [ "${use_multi_layer}" = true ]; then
    multi_token_files="" # after yuxun's token file ordering fixes
    # feat_layer= is the count of files in token_files, separated by space
    feat_layer=$(echo ${token_files} | wc -w)
    # if ${datadir}/${split}/token_file exists, then use it
    token_files_all_under_split=true
    for split in ${train_set} ${dev_set} ${eval_set}; do
        for file in ${token_files}; do
            if [ ! -f ${datadir}/${split}/${file} ]; then
                token_files_all_under_split=false
                break
            fi
        done
    done
    if [ "${token_files_all_under_split}" = false ]; then
        for file in ${token_files}; do
            multi_token_files="${multi_token_files} ${token_folder}/${file}"
        done
    else
        multi_token_files="$token_files"
    fi
    ./run.sh --stage 1 --stop_stage 1 --datadir ${datadir} --conf "${conf}" --dumpdir ${dumpdir} --use_spk_embed ${use_spk_embed} --train_set "${train_set}" --eval_set eval --multi_token_files "${multi_token_files}" --use_multi_layer ${use_multi_layer} --feat_layer ${feat_layer} --use_gpu_in_preprocess ${use_gpu_in_preprocess}
else
    token_text="${token_folder}/${token_files}"
    ./run.sh --stage 1 --stop_stage 1 --datadir ${datadir} --conf "${conf}" --dumpdir ${dumpdir} --use_spk_embed ${use_spk_embed} --train_set "${train_set}" --eval_set eval --token_text ${token_text} --use_multi_layer ${use_multi_layer} --use_gpu_in_preprocess ${use_gpu_in_preprocess}
fi

