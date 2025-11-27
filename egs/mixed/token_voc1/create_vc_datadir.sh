#!/bin/bash
# create data and dump for corvid spemb + human token/audio

# create data dir with all files from source dump, except for espnet_spk.scp which will be replaced with the new spemb
source_dump=data/human16k
new_spemb_dump=data/corvid1208_dur_filtered_concat6in_order_fs16000
outdir=data/human_tok-audio_corvid_spemb_16k_0203
random_select=false
# spk_tags=espnet_spk
# dset="dev eval tr_no_dev"
spk_tags="espnet_spk clap_large audiomae"
dset="dev eval"

. utils/parse_options.sh || exit 1;

echo "Creating VC data directory: ${outdir} from source_dump=${source_dump} and new_spemb_dump=${new_spemb_dump}"
mkdir -p ${outdir}

cp -r ${source_dump}/* ${outdir}

# if not shuffle
if [ "${random_select}" = true ]; then
    for dset in ${dset}; do
        > ${outdir}/${dset}/espnet_spk.scp.tmp
        > ${outdir}/${dset}/vc_wav.scp.tmp
        while IFS=' ' read -r utt old_spk_path; do
            # paste ${new_spemb_dump}/${dset}/espnet_spk.scp ${new_spemb_dump}/${dset}/wav.scp | shuf -n 1 
            read utt_spk new_spk_path utt_spk_wav target_spk_wav <<< $(paste ${new_spemb_dump}/${dset}/espnet_spk.scp ${new_spemb_dump}/${dset}/wav.scp | shuf -n 1)
            echo "${utt} ${new_spk_path}" >> ${outdir}/${dset}/espnet_spk.scp.tmp
            echo "${utt} ${target_spk_wav}" >> ${outdir}/${dset}/vc_wav.scp.tmp
        done < ${source_dump}/${dset}/espnet_spk.scp
        rm ${outdir}/${dset}/espnet_spk.scp
        mv ${outdir}/${dset}/espnet_spk.scp.tmp ${outdir}/${dset}/espnet_spk.scp
        mv ${outdir}/${dset}/vc_wav.scp.tmp ${outdir}/${dset}/vc_wav.scp
    done
else
    for dset in ${dset}; do
        exec 3< ${new_spemb_dump}/${dset}/espnet_spk.scp
        while IFS=' ' read -r utt old_spk_path; do
            if ! IFS=' ' read -r _ new_spk_path <&3; then
                # If the second file reaches the end, rewind it to the beginning
                exec 3< ${new_spemb_dump}/${dset}/espnet_spk.scp
                IFS=' ' read -r _ new_spk_path <&3
            fi
            echo "${utt} ${new_spk_path}" >> ${outdir}/${dset}/espnet_spk.scp.tmp
        done < ${source_dump}/${dset}/espnet_spk.scp

        rm ${outdir}/${dset}/espnet_spk.scp
        mv ${outdir}/${dset}/espnet_spk.scp.tmp ${outdir}/${dset}/espnet_spk.scp
    done

    # add vc_wav.scp for voice conversion evaluation (eval stage)
    for dset in ${dset}; do
        exec 3< ${new_spemb_dump}/${dset}/wav.scp
        while IFS=' ' read -r utt old_spk_path; do
            if ! IFS=' ' read -r _ new_spk_path <&3; then
                # If the second file reaches the end, rewind it to the beginning
                exec 3< ${new_spemb_dump}/${dset}/wav.scp
                IFS=' ' read -r _ new_spk_path <&3
            fi
            echo "${utt} ${new_spk_path}" >> ${outdir}/${dset}/vc_wav.scp.tmp
        done < ${source_dump}/${dset}/wav.scp

        mv ${outdir}/${dset}/vc_wav.scp.tmp ${outdir}/${dset}/vc_wav.scp
    done
fi