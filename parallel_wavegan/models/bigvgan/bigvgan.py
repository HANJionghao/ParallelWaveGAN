# Code adapted from https://github.com/NVIDIA/BigVGAN

# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

import typing
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
from torchaudio.transforms import Resample

import activations
from utils import init_weights, get_padding
from alias_free_activation.torch.act import Activation1d as TorchActivation1d


class AMPBlock1(torch.nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    AMPBlock1 has additional self.convs2 that contains additional Conv1d layers with a fixed dilation=1 followed by each layer in self.convs1

    Args:
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.
        snake_logscale (bool)
        use_cuda_kernel (bool): If set to True, loads optimized CUDA kernels for AMP. This should be used for inference only, as training is not supported with CUDA kernels.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
        snake_logscale: bool = True,
        use_cuda_kernel: bool = False,
    ):
        super().__init__()

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for _ in range(len(dilation))
            ]
        )
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(
            self.convs2
        )  # Total number of conv layers

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if use_cuda_kernel:
            from alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        # Activation functions
        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.Snake(
                            channels, alpha_logscale=snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.SnakeBeta(
                            channels, alpha_logscale=snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class AMPBlock2(torch.nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    Unlike AMPBlock1, AMPBlock2 does not contain extra Conv1d layers with fixed dilation=1

    Args:
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.
        snake_logscale (bool)
        use_cuda_kernel (bool): If set to True, loads optimized CUDA kernels for AMP. This should be used for inference only, as training is not supported with CUDA kernels.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
        snake_logscale: bool = True,
        use_cuda_kernel: bool = False,
    ):
        super().__init__()

        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs.apply(init_weights)

        self.num_layers = len(self.convs)  # Total number of conv layers

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if use_cuda_kernel:
            from alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        # Activation functions
        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.Snake(
                            channels, alpha_logscale=snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.SnakeBeta(
                            channels, alpha_logscale=snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        for c, a in zip(self.convs, self.activations):
            xt = a(x)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class DiscreteSymbolBigVGAN(torch.nn.Module):
    """
    Discrete Symbol BigVGAN generator module that applies anti-aliased periodic activation for residual blocks (resblocks),
    with optional f0 handling.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        channels (int): Number of hidden representation channels.
        kernel_size (int): Kernel size of initial and final conv layer.
        upsample_scales (list): List of upsampling scales.
        upsample_kernel_sizes (list): List of kernel sizes for upsampling layers.
        resblock (str): Name of AMPBlock to use. BigVGAN uses AMPBlock1 as default.
        resblock_kernel_sizes (list): List of kernel sizes for residual blocks.
        resblock_dilations (list): List of dilation list for residual blocks.
        nonlinear_activation (str): Activation function module name.
        nonlinear_activation_params (dict): Hyperparameters for activation function.
        use_tanh_at_final (bool)
        use_bias_at_final (bool)
        use_weight_norm (bool): Whether to use weight norm.
            If set to true, it will be applied to all of the conv layers.
        num_tokens (int): Number of discrete symbols (e.g., vocab size).
        num_token_layers (int)
        use_weight_sum (bool)
        use_learned_weights (bool)
        use_soft_tokens (bool)
        num_speakers (int): Number of speaker embeddings.
        use_speaker_embedding (int)
        speaker_embedding_dim (int): Speaker embedding dimension.
        speaker_feature_fusion (str)
        use_f0 (bool): Whether to use f0 (fundamental frequency) as input.
        f0_embedding_dim (int)
        use_cuda_kernel (bool): If set to True, loads optimized CUDA kernels for AMP. This should be used for inference only, as training is not supported with CUDA kernels.
    """
    def __init__(
        self,
        in_channels=512,
        out_channels=1,
        channels=512,
        kernel_size=7,
        upsample_scales=(8, 8, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4),
        resblock="AMPBlock1",
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        nonlinear_activation="snakebeta",
        nonlinear_activation_params=None,
        use_tanh_at_final=False,
        use_bias_at_final=False,
        use_weight_norm=True,
        num_tokens=100,
        num_token_layers=3,
        use_weight_sum=False,
        use_learned_weights=False,
        use_soft_tokens=False,
        num_speakers=0,
        use_speaker_embedding=False,
        speaker_embedding_dim=128,
        speaker_feature_fusion="fc_add",
        use_f0=False,
        f0_embedding_dim=256,
        use_cuda_kernel: bool = False,
    ):
        super().__init__()

        self.num_speakers = num_speakers
        self.use_speaker_embedding = use_speaker_embedding
        self.speaker_feature_fusion = speaker_feature_fusion
        self.use_f0 = use_f0
        self.use_cuda_kernel = use_cuda_kernel
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_scales)
        self.num_token_layers = num_token_layers
        self.use_soft_tokens = use_soft_tokens
        self.use_weight_sum = use_weight_sum
        self.use_learned_weights = use_learned_weights
        
        if nonlinear_activation_params is None:
            nonlinear_activation_params = {}
        # check hyperparameters are valid
        if not use_weight_norm:
            raise NotImplementedError(
                "Please use weight normalization for BigVGAN. Set use_weight_norm=True in the config file."
            )
        assert kernel_size % 2 == 1, "Kernal size must be odd number."
        assert len(upsample_scales) == len(upsample_kernel_sizes)
        assert len(resblock_dilations) == len(resblock_kernel_sizes)
        assert len(upsample_scales) != 0, "upsample_scales must not be empty"
        assert len(upsample_kernel_sizes) != 0, "upsample_kernel_sizes must not be empty"

        
        if self.use_cuda_kernel:
            from alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        conv_pre_channels = in_channels

        # Embedding for discrete symbols
        if not self.use_soft_tokens:
            self.token_embedding = torch.nn.ModuleList(
                [
                    torch.nn.Embedding(
                        num_embeddings=num_tokens, embedding_dim=in_channels
                    )
                    for _ in range(self.num_token_layers)
                ]
            )
        if self.use_weight_sum:
            if self.use_learned_weights:
                self.token_weights = torch.nn.Parameter(
                    torch.ones(self.num_token_layers)
                )

        # F0 embedding (if use_f0 is True)
        if self.use_f0:
            self.f0_embedding = torch.nn.Linear(in_features=1, out_features=f0_embedding_dim)
            conv_pre_channels += f0_embedding_dim

        # Speaker embedding
        if self.num_speakers > 0:
            self.speaker_embedding = torch.nn.Embedding(num_embeddings=num_speakers, embedding_dim=speaker_embedding_dim)
        if self.num_speakers > 0 or self.use_speaker_embedding:
            if self.speaker_feature_fusion == "concat":
                conv_pre_channels += speaker_embedding_dim
            elif self.speaker_feature_fusion == "add":
                assert speaker_embedding_dim == conv_pre_channels, \
                    f"Speaker embedding dimension {speaker_embedding_dim} must match conv_pre_channels {conv_pre_channels} for 'add' fusion."
            elif self.speaker_feature_fusion == "fc_add":
                self.speaker_embedding_proj = torch.nn.Linear(speaker_embedding_dim, conv_pre_channels)
            else:
                raise NotImplementedError(
                    f"Speaker feature fusion method '{self.speaker_feature_fusion}' is not implemented."
                )


        # Pre-conv
        self.conv_pre = weight_norm(
            Conv1d(
                conv_pre_channels,
                channels,
                kernel_size,
                1,
                padding=(kernel_size - 1) // 2,
            )
        )

        # Define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        if resblock == "AMPBlock1":
            resblock_class = AMPBlock1
        elif resblock == "AMPBlock2":
            resblock_class = AMPBlock2
        else:
            raise NotImplementedError(
                f"Incorrect resblock class specified in the config 'generator_params'. Got {resblock}"
            )

        # Transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_scales, upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                channels // (2 ** i),
                                channels // (2 ** (i + 1)),
                                k,
                                u,
                                padding=(k - u) // 2,
                            )
                        )
                    ]
                )
            )

        # Residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = channels // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilations)
            ):
                self.resblocks.append(
                    resblock_class(
                        channels=ch,
                        kernel_size=k,
                        dilation=d,
                        activation=nonlinear_activation,
                        snake_logscale=nonlinear_activation_params.get("snake_logscale", True),
                        use_cuda_kernel=use_cuda_kernel,
                    )
                )

        # Post-conv
        if nonlinear_activation == "snake":
            activation_post = activations.Snake(
                ch, alpha_logscale=nonlinear_activation_params.get("snake_logscale", True)
            )
        elif nonlinear_activation == "snakebeta":
            activation_post = activations.SnakeBeta(
                ch, alpha_logscale=nonlinear_activation_params.get("snake_logscale", True)
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'nonlinear_activation'."
            )
        self.activation_post = Activation1d(activation=activation_post)
        # Whether to use bias for the final conv_post.
        self.use_bias_at_final = use_bias_at_final
        self.conv_post = weight_norm(
            Conv1d(
                ch,
                out_channels,
                kernel_size,
                1,
                padding=(kernel_size - 1) // 2,
                bias=use_bias_at_final,
            )
        )

        # Weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

        # Final tanh activation.
        self.use_tanh_at_final = use_tanh_at_final

    def forward(self, x, f0=None, additional_feats=None):
        """
        Calculate forward propagation.
        
        Args:
            x (Tensor): Input token tensor: (B, L, T) for discrete tokens, (B, L, T, in_channels) for continuous tokens
            f0 (Tensor): Input f0 tensor (B, 1, T)
            additional_feats (dict): Additional features
                - sid (Tensor): Speaker ID tensor (B, 1)
                - spemb (Tensor): Speaker embedding tensor (B, speaker_embedding_dim)
        Returns:
            Tensor: Output tensor (B, out_channels, T').
        """
        # Embedding for discrete symbols
        if not self.use_soft_tokens:
            token_embeddings = [
                self.token_embedding[i](x[:, i, :].long()) # (B, T, in_channels)
                for i in range(self.num_token_layers)
            ]
            x = torch.stack(token_embeddings, dim=1)  # (B, L, T, in_channels)
            x = x.transpose(1, -1) # (B, in_channels, T, L)
        else:
            x = x.transpose(1, -1) # (B, in_channels, T, L)
        
        if x.size(1) == 1:
            x = x.squeeze(1)
        elif self.use_weight_sum:
            if self.use_learned_weights:
                norm_weights = F.softmax(self.token_weights, dim=0) # (L,)
                x = torch.matmul(x, norm_weights)
            else:
                x = torch.mean(x, dim=-1) # (B, in_channels, T)
        else:
            raise NotImplementedError(
                "Not implemented for use_weight_sum=False and num_token_layers>1"
            )

        # If f0 is used, process f0 and concatenate
        if self.use_f0 and f0 is not None:
            f0 = self.f0_embedding(f0.transpose(1, 2)).transpose(1, 2)
            x = torch.cat([x, f0], dim=1)

        # Speaker embedding
        if self.num_speakers > 0:
            assert "sid" in additional_feats, f"Speaker ID (sid) must be provided in additional_feats. Current keys: {additional_feats.keys()}"
            spemb = self.speaker_embedding(additional_feats["sid"].long()) # (B, speaker_embedding_dim)
        if self.use_speaker_embedding:
            assert "spemb" in additional_feats, f"Speaker embedding must be provided in additional_feats. Current keys: {additional_feats.keys()}. Please rerun run.sh with --use_spk_embed true"
            spemb = F.normalize(additional_feats["spemb"])
        if self.num_speakers > 0 or self.use_speaker_embedding:
            if self.speaker_feature_fusion == "concat":
                spemb = spemb.unsqueeze(2).expand(-1, -1, x.size(2)) # (B, speaker_embedding_dim, T)
                x = torch.cat([x, spemb], dim=1) # (B, C + speaker_embedding_dim, T)
            elif self.speaker_feature_fusion == "add":
                spemb = spemb.unsqueeze(2) # (B, C, 1)
                x = x + spemb # (B, C, T)
            elif self.speaker_feature_fusion == "fc_add":
                spemb = self.speaker_embedding_proj(spemb).unsqueeze(2)
                x = x + spemb # (B, C, T)

        # Pre-conv
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # Upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # Post-conv
        x = self.activation_post(x)
        x = self.conv_post(x)

        # Final tanh activation
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)  # Bound the output to [-1, 1]

        return x

    # def remove_weight_norm(self):
    #     """Remove weight normalization across all layers."""
    #     self.apply(lambda m: remove_weight_norm(m) if isinstance(m, (Conv1d, ConvTranspose1d)) else None)


# Adapted from https://github.com/open-mmlab/Amphion/blob/main/models/vocoders/gan/discriminator/mssbcqtd.py under the MIT license.
#   LICENSE is in incl_licenses directory.
class DiscriminatorCQT(nn.Module):
    def __init__(
        self,
        sampling_rate: int,
        in_channels: int,
        out_channels: int,
        filters: int,
        max_filters: int,
        filters_scale: int,
        dilations: List[int],
        hop_length: int,
        n_octaves: int,
        bins_per_octave: int,
    ):
        super().__init__()

        self.filters = filters
        self.max_filters = max_filters
        self.filters_scale = filters_scale
        self.kernel_size = (3, 9)
        self.dilations = dilations
        self.stride = (1, 2)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fs = sampling_rate
        self.hop_length = hop_length
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave

        # Lazy-load
        from nnAudio import features

        self.cqt_transform = features.cqt.CQT2010v2(
            sr=self.fs * 2,
            hop_length=self.hop_length,
            n_bins=self.bins_per_octave * self.n_octaves,
            bins_per_octave=self.bins_per_octave,
            output_format="Complex",
            pad_mode="constant",
        )

        self.conv_pres = nn.ModuleList()
        for _ in range(self.n_octaves):
            self.conv_pres.append(
                nn.Conv2d(
                    self.in_channels * 2,
                    self.in_channels * 2,
                    kernel_size=self.kernel_size,
                    padding=self.get_2d_padding(self.kernel_size),
                )
            )

        self.convs = nn.ModuleList()

        self.convs.append(
            nn.Conv2d(
                self.in_channels * 2,
                self.filters,
                kernel_size=self.kernel_size,
                padding=self.get_2d_padding(self.kernel_size),
            )
        )

        in_chs = min(self.filters_scale * self.filters, self.max_filters)
        for i, dilation in enumerate(self.dilations):
            out_chs = min(
                (self.filters_scale ** (i + 1)) * self.filters, self.max_filters
            )
            self.convs.append(
                weight_norm(
                    nn.Conv2d(
                        in_chs,
                        out_chs,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        dilation=(dilation, 1),
                        padding=self.get_2d_padding(self.kernel_size, (dilation, 1)),
                    )
                )
            )
            in_chs = out_chs
        out_chs = min(
            (self.filters_scale ** (len(self.dilations) + 1)) * self.filters,
            self.max_filters,
        )
        self.convs.append(
            weight_norm(
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=(self.kernel_size[0], self.kernel_size[0]),
                    padding=self.get_2d_padding(
                        (self.kernel_size[0], self.kernel_size[0])
                    ),
                )
            )
        )

        self.conv_post = weight_norm(
            nn.Conv2d(
                out_chs,
                self.out_channels,
                kernel_size=(self.kernel_size[0], self.kernel_size[0]),
                padding=self.get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
            )
        )

        self.activation = torch.nn.LeakyReLU(negative_slope=0.1)
        self.resample = Resample(orig_freq=self.fs, new_freq=self.fs * 2)

        self.cqtd_normalize_volume = self.cfg.get("cqtd_normalize_volume", False)
        if self.cqtd_normalize_volume:
            print(
                f"[INFO] cqtd_normalize_volume set to True. Will apply DC offset removal & peak volume normalization in CQTD!"
            )

    def get_2d_padding(
        self,
        kernel_size: typing.Tuple[int, int],
        dilation: typing.Tuple[int, int] = (1, 1),
    ):
        return (
            ((kernel_size[0] - 1) * dilation[0]) // 2,
            ((kernel_size[1] - 1) * dilation[1]) // 2,
        )

    def forward(self, x: torch.tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        outs = []

        if self.cqtd_normalize_volume:
            # Remove DC offset
            x = x - x.mean(dim=-1, keepdims=True)
            # Peak normalize the volume of input audio
            x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)

        x = self.resample(x)

        z = self.cqt_transform(x)

        z_amplitude = z[:, :, :, 0].unsqueeze(1)
        z_phase = z[:, :, :, 1].unsqueeze(1)

        z = torch.cat([z_amplitude, z_phase], dim=1)
        z = torch.permute(z, (0, 1, 3, 2))  # [B, C, W, T] -> [B, C, T, W]

        latent_z = []
        for i in range(self.n_octaves):
            latent_z.append(
                self.conv_pres[i](
                    z[
                        :,
                        :,
                        :,
                        i * self.bins_per_octave : (i + 1) * self.bins_per_octave,
                    ]
                )
            )
        latent_z = torch.cat(latent_z, dim=-1)

        for i, l in enumerate(self.convs):
            latent_z = l(latent_z)

            latent_z = self.activation(latent_z)
            outs.append(latent_z)

        latent_z = self.conv_post(latent_z)
        outs.append(latent_z)

        return outs


class MultiScaleSubbandCQTDiscriminator(nn.Module):
    def __init__(
        self,
        sampling_rate: int,
        in_channels: int = 1,
        out_channels: int = 1,
        filters: int = 32,
        max_filters: int = 1024,
        filters_scale: int = 1,
        dilations: List[int] = [1, 2, 4],
        hop_lengths: int = [512, 256, 256],
        n_octaves: int = [9, 9, 9],
        bins_per_octaves: int = [24, 36, 48],
    ):
        super().__init__()

        self.discriminators = nn.ModuleList(
            [
                DiscriminatorCQT(
                    sampling_rate,
                    in_channels,
                    out_channels,
                    filters,
                    max_filters,
                    filters_scale,
                    dilations,
                    hop_length=hop_lengths[i],
                    n_octaves=n_octaves[i],
                    bins_per_octave=bins_per_octaves[i],
                )
                for i in range(len(hop_lengths))
            ]
        )

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:

        outs = []

        for disc in self.discriminators:
            outs += [disc(x)]

        return outs


class BigVGANMultiResolutionMultiPeriodDiscriminator(torch.nn.Module):
    """HiFi-GAN multi-scale + multi-period discriminator module."""

    def __init__(
        self,
        mpd_type: str,
        mpd_params: Dict,
        mrd_type: str,
        mrd_params: Dict,
    ):
        """Initilize HiFiGAN multi-scale + multi-period discriminator module.

        Args:
            scales (int): Number of multi-scales.
            scale_downsample_pooling (str): Pooling module name for downsampling of the inputs.
            scale_downsample_pooling_params (dict): Parameters for the above pooling module.
            scale_discriminator_params (dict): Parameters for hifi-gan scale discriminator module.
            follow_official_norm (bool): Whether to follow the norm setting of the official
                implementaion. The first discriminator uses spectral norm and the other
                discriminators use weight norm.
            periods (list): List of periods.
            period_discriminator_params (dict): Parameters for hifi-gan period discriminator module.
                The period parameter will be overwritten.

        """
        super().__init__()
        if mpd_type != "HiFiGANMultiPeriodDiscriminator":
            raise NotImplementedError(
                f"mpd_type {mpd_type} is not supported."
            )
        import parallel_wavegan.models.hifigan
        mpd_class = parallel_wavegan.models.hifigan.HiFiGANMultiPeriodDiscriminator
        self.mpd = mpd_class(**mpd_params)
        ALLOWED_MRD_TYPES = {
            "MultiScaleSubbandCQTDiscriminator": MultiScaleSubbandCQTDiscriminator,
        }
        mrd_class = ALLOWED_MRD_TYPES.get(mrd_type)
        if mrd_class is None:
            raise NotImplementedError(
                f"mrd_type {mrd_type} is not supported. "
                f"Allowed types are: {list(ALLOWED_MRD_TYPES.keys())}"
            )
        self.mrd = mrd_class(**mrd_params)

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List: List of list of each discriminator outputs,
                which consists of each layer output tensors.
                Multi scale and multi period ones are concatenated.

        """
        mrd_outs = self.mrd(x)
        mpd_outs = self.mpd(x)
        return mrd_outs + mpd_outs
