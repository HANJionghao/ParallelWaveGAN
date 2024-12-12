from __future__ import annotations

import torch
import torch.nn.functional as F
import logging
from typing import Literal


class SpkEmbedExtractor(torch.nn.Module):
    def __init__(
        self,
        toolkit,
        pretrained_model,
        in_sr,
        freeze=True,
        device: str | torch.device = None,
    ):
        from pathlib import Path
        import torchaudio.transforms as T

        super().__init__()

        self.toolkit = toolkit
        self.tgt_sr = 16000  # NOTE(jiatong): following 16khz convertion
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.in_sr = in_sr
        if self.in_sr != self.tgt_sr:
            self.resample = T.Resample(orig_freq=self.in_sr, new_freq=self.tgt_sr)
        else:
            self.resample = None
        self.chunk_size = 5 * self.tgt_sr  # 5 seconds
        self.freeze = freeze

        if self.toolkit == "speechbrain":
            # TODO(jhan): verify this
            from speechbrain.dataio.preprocess import AudioNormalizer
            from speechbrain.pretrained import EncoderClassifier

            self.audio_norm = AudioNormalizer()
            self.model = EncoderClassifier.from_hparams(
                source=pretrained_model, run_opts={"device": device}
            )
        elif self.toolkit == "rawnet":
            from RawNet3 import RawNet3
            from RawNetBasicBlock import Bottle2neck

            self.model = RawNet3(
                Bottle2neck,
                model_scale=8,
                context=True,
                summed=True,
                encoder_type="ECA",
                nOut=256,
                out_bn=False,
                sinc_stride=10,
                log_sinc=True,
                norm_sinc="mean",
                grad_mult=1,
            )
            tools_dir = Path().resolve().parent.parent.parent / "tools"
            self.model.load_state_dict(
                torch.load(
                    tools_dir / "RawNet/python/RawNet3/models/weights/model.pt",
                    map_location=lambda storage, loc: storage,
                )["model"]
            )
            self.model.to(device)
        elif self.toolkit == "espnet":
            from espnet2.tasks.spk import SpeakerTask

            if pretrained_model.endswith("pth"):
                logging.info(
                    "the provided model path is end with pth,"
                    "assume it not a huggingface model"
                )
                model_file = pretrained_model
                # NOTE(jiatong): set default config file as None
                # assume config is the same path as the model file
                train_config = None
                # TODO(jhan): test functionality
            else:
                logging.info(
                    "the provided model path is not end with pth,"
                    "assume use huggingface model"
                )
                from espnet_model_zoo.downloader import ModelDownloader

                d = ModelDownloader()
                downloaded = d.download_and_unpack(pretrained_model)
                model_file = downloaded["model_file"]
                train_config = downloaded["train_config"]

            self.model, _ = SpeakerTask.build_model_from_file(
                train_config, model_file, str(self.device)
            )
            self.model.to(device)

        if freeze:
            self.freeze_model()

    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def _rawnet_extract_embd(self, audio, n_samples=48000, n_segments=10):
        # TODO(jhan): verify this torch implementation
        if len(audio) < n_samples:  # RawNet3 was trained using utterances of 3 seconds
            shortage = n_samples - len(audio) + 1
            audio = torch.cat([audio, audio[:shortage]], dim=0)
        audios = []
        startframe = torch.linspace(0, len(audio) - n_samples, steps=n_segments)
        for asf in startframe:
            audios.append(audio[int(asf) : int(asf) + n_samples])
        audios = torch.stack(audios, dim=0).to(audio.device)
        with torch.no_grad():
            output = self.model(audios)
        return output.mean(0)

    def _espnet_extract_embd(self, audio):
        output = self.model(speech=audio, extract_embd=True)
        return output

    def forward(self, wav: torch.Tensor):
        # wav: (B, T)
        if self.freeze:
            self.model.eval()
        if self.resample:
            wav = self.resample(wav)
        if self.toolkit == "speechbrain":
            # TODO(jhan): verify this
            wav = self.audio_norm(wav, self.in_sr).to(wav.device)
            embeds = self.model.encode_batch(wav)[0]
        elif self.toolkit == "rawnet":
            embeds = self._rawnet_extract_embd(wav)
        elif self.toolkit == "espnet":
            embeds = self._espnet_extract_embd(wav)
        return embeds


class SpeakerContrastiveLoss(torch.nn.Module):
    """Speaker contrastive loss module."""

    def __init__(
        self,
        device: str | torch.device,
        speaker_model_conf: dict,
        temperature: float | Literal["learnable"] = None,
    ):
        """Initialize SpeakerContrastiveLoss module.

        Args:
            device (str or torch.device): Device type.
            in_sr (int): Input sampling rate.
            speaker_model_conf (dict): Speaker embedding configuration.
            temperature (float or str): Temperature parameter for contrastive loss. If "learnable", it will be a learnable parameter.

        """
        super().__init__()
        self.temperature_inverse = (
            1 / temperature
            if temperature != "learnable"
            else torch.nn.Parameter(torch.tensor(1 / 0.07))
        )
        speaker_model_conf = speaker_model_conf or {}
        if (spk_embed_tool := speaker_model_conf.get("tool", None)) and (
            pretrained_model := speaker_model_conf.get("pretrained_model", None)
        ):
            self.speaker_embedding_model = SpkEmbedExtractor(
                toolkit=spk_embed_tool,
                pretrained_model=pretrained_model,
                in_sr=speaker_model_conf["in_sr"],
                device=device,
                freeze=speaker_model_conf.get("freeze", True),
            )
        else:
            raise ValueError(
                "Tool or pretrained_model is not provided."
                "Please check the configuration file."
            )

    def _get_speaker_embedding(self, wav: torch.Tensor) -> torch.Tensor:
        """Get speaker embedding.

        Args:
            wav (Tensor): Input waveform (B, T).

        Returns:
            Tensor: Speaker embedding (B, spk_embed_dim).

        """
        return self.speaker_embedding_model(wav)

    def forward(
        self,
        singing_hat: torch.Tensor,
        spk_embs_positive: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Calculate speaker contrastive loss.

        Args:
            singing_hat (Tensor): Predicted singing waveform tensor (B, 1, T).
            spk_embs_positive (Tensor): Speaker embeddings tensor from groundtruth (B, D). Paired with spk_embeds_hat.

        Returns:
            Tensor: Speaker contrastive loss value.

        """
        # get speaker embeddings
        assert (
            singing_hat.size(1) == 1
        ), f"singing_hat should be (B, 1, T), but got {singing_hat.size()}"
        spk_embs_hat = self._get_speaker_embedding(singing_hat.squeeze(1))

        # normalize embeddings
        spk_embs_hat = F.normalize(spk_embs_hat, dim=1)  # (B, D)
        spk_embs_positive = F.normalize(spk_embs_positive, dim=1)  # (B, D)

        # calculate cosine similarity
        sim_matrix = torch.matmul(spk_embs_hat, spk_embs_positive.t())  # (B, B)
        sim_matrix = sim_matrix * self.temperature_inverse
        target = torch.arange(sim_matrix.size(0), device=sim_matrix.device)  # (B,)
        loss = F.cross_entropy(sim_matrix, target) + F.cross_entropy(
            sim_matrix.t(), target
        )
        return loss
