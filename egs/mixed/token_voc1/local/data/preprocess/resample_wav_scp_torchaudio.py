# resample_wav_scp_torchaudio.py ${fs} ${args.source_wav_scp} ${args.wav_dump} ${args.output_wav_scp}

from pathlib import Path
import argparse
import numpy as np
import torch
import torchaudio
from espnet2.fileio.sound_scp import SoundScpReader
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(
        description="Resample and convert audio files to mono"
    )
    parser.add_argument("--fs", type=int, required=True, help="Sampling rate")
    parser.add_argument(
        "--source_wav_scp", type=str, required=True, help="Source audio scp file"
    )
    parser.add_argument(
        "--output_wav_scp", type=str, required=True, help="Output audio scp file"
    )
    parser.add_argument(
        "--wav_dump",
        type=str,
        required=True,
        help="Directory to dump resampled audio files",
    )
    parser.add_argument(
        "--audio_ext",
        type=str,
        default="wav",
        help="Audio file extension (e.g., wav, flac)",
    )
    return parser


def main(args):
    wav_dump = Path(args.wav_dump)
    wav_dump.mkdir(parents=True, exist_ok=True)

    source_wav_scp = SoundScpReader(args.source_wav_scp, np.float32)

    with open(args.output_wav_scp, "w") as f:
        for utt, (in_sr, audio) in tqdm(source_wav_scp.items()):
            # convert to mono if stereo
            if audio.ndim == 2:
                audio = audio.mean(axis=0)
            if in_sr != args.fs:
                resample = torchaudio.transforms.Resample(
                    orig_freq=in_sr, new_freq=args.fs
                )
                audio = resample(torch.tensor(audio))

            resampled_audio_path = wav_dump / f"{utt}.{args.audio_ext}"
            try:
                torchaudio.save(
                    resampled_audio_path,
                    audio.unsqueeze(0),
                    args.fs,
                    encoding="PCM_S",
                    bits_per_sample=16,
                )
            except:
                torchaudio.save(
                    resampled_audio_path,
                    audio,
                    args.fs,
                    encoding="PCM_S",
                    bits_per_sample=16,
                )
            f.write(f"{utt} {resampled_audio_path}\n")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)