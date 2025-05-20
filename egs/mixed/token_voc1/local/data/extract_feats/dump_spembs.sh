pretrained_model=espnet/voxcelebs12_rawnet3
toolkit=espnet
spk_embed_tag=espnet_spk
resample_package=torchaudio
data_split_dir=
spemb_dump_dir=
cmd=
nj=1
use_gpu=true
espnet_path=/ocean/projects/cis210027p/jhan7/cartoon_voice/model/espnet_dev/

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1

data_split_dir=$(readlink -f "${data_split_dir}")
mkdir -p "${spemb_dump_dir}"
spemb_dump_dir=$(readlink -f "${spemb_dump_dir}")

if ${use_gpu}; then
    ngpu=1
else
    ngpu=0
fi

(
    cd ${espnet_path}/egs2/ami/asr1
    . ./path.sh

    scripts/utils/extract_spk_embed_utt.sh --nj 1 \
        --gpu "${ngpu}" \
        --cmd "${cmd}" \
        --data "${data_split_dir}" \
        --output "${spemb_dump_dir}" \
        --spk_embed_tag "${spk_embed_tag}" \
        --pretrained_model "${pretrained_model}" \
        --resample_package "${resample_package}" \
        --toolkit "${toolkit}"
)

ln -sf "${spemb_dump_dir}/${spk_embed_tag}.scp" "${data_split_dir}/${spk_embed_tag}.scp"
