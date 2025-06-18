# This script iterates through all the audio files in ${dataset_folder} and creates a wav.scp file.

dataset_folder=
verbose=false
output_wav_scp=
dataset_tag=gdc

# Expected format of the dataset folder:
# dataset_folder/
# ├── 2019/
# │   └── ...
# └── ...

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1

if [ -z "${dataset_folder}" ]; then
    echo "[ERROR] Please set the dataset_folder variable."
    exit 1
fi
if [ -z "${output_wav_scp}" ]; then
    echo "[ERROR] Please set the output_wav_scp variable."
    exit 1
fi

> "${output_wav_scp}"

dataset_folder=$(realpath "${dataset_folder}")

mkdir -p "$(dirname "${output_wav_scp}")"
for year_folder in "${dataset_folder}"/*; do
    year=$(basename "${year_folder}")
    find "${year_folder}" -type f -name "*.wav" | sort | while read -r file; do
        category=$(basename "$(dirname "$file")") # e.g. "XXX/2018/Sonniss.com - GDC 2018 - Game Audio Bundle Part 2of8/Bluezone Corporation -  Subspace Distortion - Sci Fi Cinematic Samples/Bluezone_BC0233_transition_016.wav" -> "Bluezone Corporation -  Subspace Distortion - Sci Fi Cinematic Samples"  
        category=$(echo "$category" | perl -CS -pe 's/[\p{Zs}\t]*([\p{P}])[\p{Zs}\t]*/$1/g' | perl -CS -pe 'chomp; s/\p{Space}/_/g') # "Bluezone_Corporation-Subspace_Distortion-Sci_Fi_Cinematic_Samples"
        category="${category//[\']/}" # "Bluezone_Corporation-Subspace_Distortion-Sci_Fi_Cinematic_Samples"
        category="${category//[.,]/_}" # "Bluezone_Corporation-Subspace_Distortion-Sci_Fi_Cinematic_Samples"
        hash_code=$(basename "${file%.*}" | md5sum | cut -c1-8)
        echo "${dataset_tag}_${year}_${category}_${hash_code} ${file}" >> "${output_wav_scp}"
    done
done

sort -k1,1 -u "${output_wav_scp}" -o "${output_wav_scp}"

if [ "$verbose" = true ]; then
    echo "[INFO] Finished creating ${output_wav_scp} from ${dataset_folder} for dataset: ${dataset_tag}"
fi
