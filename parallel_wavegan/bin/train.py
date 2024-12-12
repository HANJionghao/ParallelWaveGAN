#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Train Parallel WaveGAN."""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict

import matplotlib
import numpy as np
import parallel_wavegan.batch_sampler
import soundfile as sf
import torch
import yaml
import math
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

import parallel_wavegan
import parallel_wavegan.models
import parallel_wavegan.optimizers
from parallel_wavegan.datasets import (
    AudioDataset,
    AudioMelDataset,
    AudioMelF0Dataset,
    AudioMelF0ExcitationDataset,
    AudioMelSCPDataset,
    AudioSCPDataset,
    AudioMRTokenlDataset,
)
from parallel_wavegan.layers import PQMF
from parallel_wavegan.losses import (
    DiscriminatorAdversarialLoss,
    DurationPredictorLoss,
    FeatureMatchLoss,
    GeneratorAdversarialLoss,
    MelSpectrogramLoss,
    MultiResolutionSTFTLoss,
    SpeakerContrastiveLoss,
)
from parallel_wavegan.utils import read_hdf5

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


class Trainer(object):
    """Customized trainer module for Parallel WaveGAN training."""

    def __init__(
        self,
        steps,
        epochs,
        data_loader,
        sampler,
        model,
        criterion,
        optimizer,
        scheduler,
        config,
        device=torch.device("cpu"),
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders. It must contrain "train" and "dev" loaders.
            model (dict): Dict of models. It must contrain "generator" and "discriminator" models.
            criterion (dict): Dict of criterions. It must contrain "stft" and "mse" criterions.
            optimizer (dict): Dict of optimizers. It must contrain "generator" and "discriminator" optimizers.
            scheduler (dict): Dict of schedulers. It must contrain "generator" and "discriminator" schedulers.
            config (dict): Config dict loaded from yaml format configuration file.
            device (torch.deive): Pytorch device instance.

        """
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.sampler = sampler
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.writer = SummaryWriter(config["outdir"])
        self.finish_train = False
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)
        self.is_vq = "VQVAE" in config.get("generator_type", "ParallelWaveGANGenerator")
        self.use_duration_prediction = "Duration" in config.get(
            "generator_type", "ParallelWaveGANGenerator"
        )
        # self.use_f0 = "F0" in config.get(
        #     "generator_type", "ParallelWaveGANGenerator"
        # )

    def run(self):
        """Run training."""
        self.tqdm = tqdm(
            initial=self.steps, total=self.config["train_max_steps"], desc="[train]"
        )
        while True:
            # train one epoch
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logging.info("Finished training.")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.

        """
        state_dict = {
            "optimizer": {
                "generator": self.optimizer["generator"].state_dict(),
                "discriminator": self.optimizer["discriminator"].state_dict(),
            },
            "scheduler": {
                "generator": self.scheduler["generator"].state_dict(),
                "discriminator": self.scheduler["discriminator"].state_dict(),
            },
            "steps": self.steps,
            "epochs": self.epochs,
        }
        if self.config["distributed"]:
            state_dict["model"] = {
                "generator": self.model["generator"].module.state_dict(),
                "discriminator": self.model["discriminator"].module.state_dict(),
            }
        else:
            state_dict["model"] = {
                "generator": self.model["generator"].state_dict(),
                "discriminator": self.model["discriminator"].state_dict(),
            }

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if self.config["distributed"]:
            self.model["generator"].module.load_state_dict(
                state_dict["model"]["generator"],
            )
            self.model["discriminator"].module.load_state_dict(
                state_dict["model"]["discriminator"],
                strict=False,
            )
        else:
            self.model["generator"].load_state_dict(
                state_dict["model"]["generator"],
            )
            self.model["discriminator"].load_state_dict(
                state_dict["model"]["discriminator"],
                strict=False,
            )
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer["generator"].load_state_dict(
                state_dict["optimizer"]["generator"]
            )
            self.optimizer["discriminator"].load_state_dict(
                state_dict["optimizer"]["discriminator"]
            )
            self.scheduler["generator"].load_state_dict(
                state_dict["scheduler"]["generator"]
            )
            self.scheduler["discriminator"].load_state_dict(
                state_dict["scheduler"]["discriminator"]
            )

    def _train_step(self, batch):
        """Train model one step."""
        # parse batch and send to device
        is_batch_pair = (
            self.data_loader["train"].batch_sampler.__class__.__name__
            == "CategoryPairSampler"
        )
        if self.use_duration_prediction:
            x, y, ds = self._parse_batch(batch, is_batch_list=is_batch_pair)
        else:
            x, y = self._parse_batch(batch, is_batch_list=is_batch_pair)

        if is_batch_pair:
            x, x_ref = x
            y, y_ref = y
            if self.use_duration_prediction:
                ds, ds_ref = ds

        #######################
        #      Generator      #
        #######################
        if self.steps > self.config.get("generator_train_start_steps", 0):
            # initialize
            gen_loss = 0.0

            if self.is_vq:
                # vq case
                if self.config["generator_params"]["in_channels"] == 1:
                    y_, z_e, z_q = self.model["generator"](y, *x)
                else:
                    y_mb = self.criterion["pqmf"].analysis(y)
                    y_, z_e, z_q = self.model["generator"](y_mb, *x)
                quantize_loss = self.criterion["mse"](z_q, z_e.detach())
                commit_loss = self.criterion["mse"](z_e, z_q.detach())
                self.total_train_loss["train/quantization_loss"] += quantize_loss.item()
                self.total_train_loss["train/commitment_loss"] += commit_loss.item()
                gen_loss += quantize_loss + self.config["lambda_commit"] * commit_loss
            elif self.use_duration_prediction:
                assert ds is not None
                y_, ds_ = self.model["generator"](x, ds)
                duration_loss = self.criterion["duration"](ds_, ds)
                self.total_train_loss["train/duration_loss"] += duration_loss.item()
                gen_loss += duration_loss
            else:
                y_ = self.model["generator"](*x)

            # reconstruct the signal from multi-band signal
            if self.config["generator_params"]["out_channels"] > 1:
                y_mb_ = y_
                y_ = self.criterion["pqmf"].synthesis(y_mb_)

            # multi-resolution sfft loss
            if self.config["use_stft_loss"]:
                sc_loss, mag_loss = self.criterion["stft"](y_, y)
                gen_loss += sc_loss + mag_loss
                self.total_train_loss[
                    "train/spectral_convergence_loss"
                ] += sc_loss.item()
                self.total_train_loss[
                    "train/log_stft_magnitude_loss"
                ] += mag_loss.item()

            # subband multi-resolution stft loss
            if self.config["use_subband_stft_loss"]:
                gen_loss *= 0.5  # for balancing with subband stft loss
                if not self.is_vq:
                    y_mb = self.criterion["pqmf"].analysis(y)
                sub_sc_loss, sub_mag_loss = self.criterion["sub_stft"](y_mb_, y_mb)
                gen_loss += 0.5 * (sub_sc_loss + sub_mag_loss)
                self.total_train_loss[
                    "train/sub_spectral_convergence_loss"
                ] += sub_sc_loss.item()
                self.total_train_loss[
                    "train/sub_log_stft_magnitude_loss"
                ] += sub_mag_loss.item()

            # mel spectrogram loss
            if self.config["use_mel_loss"]:
                mel_loss = self.criterion["mel"](y_, y)
                gen_loss += mel_loss
                self.total_train_loss["train/mel_loss"] += mel_loss.item()

            # weighting aux loss
            gen_loss *= self.config.get("lambda_aux", 1.0)

            # adversarial loss
            if self.steps > self.config["discriminator_train_start_steps"]:
                p_ = self.model["discriminator"](y_)
                adv_loss = self.criterion["gen_adv"](p_)
                self.total_train_loss["train/adversarial_loss"] += adv_loss.item()

                # feature matching loss
                if self.config["use_feat_match_loss"]:
                    # no need to track gradients
                    with torch.no_grad():
                        p = self.model["discriminator"](y)
                    fm_loss = self.criterion["feat_match"](p_, p)
                    self.total_train_loss[
                        "train/feature_matching_loss"
                    ] += fm_loss.item()
                    adv_loss += self.config["lambda_feat_match"] * fm_loss

                # add adversarial loss to generator loss
                gen_loss += self.config["lambda_adv"] * adv_loss

            if self.config["use_speaker_contra_loss"]:
                # speaker contrastive loss
                spemb_ref = x_ref[2]["spemb"]
                del x_ref, y_ref # TODO(jhan): May remove this line, and delete ds_ref if exists
                spk_loss = self.criterion["speaker_contra"](y_, spemb_ref)
                gen_loss += self.config["lambda_speaker_contra"] * spk_loss
                self.total_train_loss["train/speaker_contra_loss"] += spk_loss.item()

            self.total_train_loss["train/generator_loss"] += gen_loss.item()

            # update generator
            self.optimizer["generator"].zero_grad()
            gen_loss.backward()
            if self.config["generator_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model["generator"].parameters(),
                    self.config["generator_grad_norm"],
                )
            self.optimizer["generator"].step()
            self.scheduler["generator"].step()

        #######################
        #    Discriminator    #
        #######################
        if self.steps > self.config["discriminator_train_start_steps"]:
            if self.config.get("update_prediction_after_generator_update", True):
                # re-compute y_ which leads better quality
                with torch.no_grad():
                    if self.is_vq:
                        if self.config["generator_params"]["in_channels"] == 1:
                            y_, _, _ = self.model["generator"](y, *x)
                        else:
                            y_, _, _ = self.model["generator"](y_mb, *x)
                    elif self.use_duration_prediction:
                        assert ds is not None
                        y_, _ = self.model["generator"](x, ds)
                    else:
                        y_ = self.model["generator"](*x)
                if self.config["generator_params"]["out_channels"] > 1:
                    y_ = self.criterion["pqmf"].synthesis(y_)

            # discriminator loss
            p = self.model["discriminator"](y)
            p_ = self.model["discriminator"](y_.detach())
            real_loss, fake_loss = self.criterion["dis_adv"](p_, p)
            dis_loss = real_loss + fake_loss
            self.total_train_loss["train/real_loss"] += real_loss.item()
            self.total_train_loss["train/fake_loss"] += fake_loss.item()
            self.total_train_loss["train/discriminator_loss"] += dis_loss.item()

            # update discriminator
            self.optimizer["discriminator"].zero_grad()
            dis_loss.backward()
            if self.config["discriminator_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model["discriminator"].parameters(),
                    self.config["discriminator_grad_norm"],
                )
            self.optimizer["discriminator"].step()
            self.scheduler["discriminator"].step()

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _train_epoch(self):
        """Train model one epoch."""
        try:
            self.data_loader["train"].batch_sampler.update_epoch(self.epochs)
        except AttributeError:
            pass
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            if self.config["rank"] == 0:
                self._check_log_interval()
                self._check_eval_interval()
                self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logging.info(
            f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
            f"({self.train_steps_per_epoch} steps per epoch)."
        )

        # needed for shuffle in distributed training
        if self.config["distributed"]:
            self.sampler["train"].set_epoch(self.epochs)

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        is_batch_pair = (
            self.data_loader["dev"].batch_sampler.__class__.__name__
            == "CategoryPairSampler"
        )
        # parse batch and send to device
        if self.use_duration_prediction:
            x, y, ds = self._parse_batch(batch, is_batch_list=is_batch_pair)
        else:
            x, y = self._parse_batch(batch, is_batch_list=is_batch_pair)

        if is_batch_pair:
            x, x_ref = x
            y, y_ref = y
            if self.use_duration_prediction:
                ds, ds_ref = ds

        #######################
        #      Generator      #
        #######################
        if self.is_vq:
            if self.config["generator_params"]["in_channels"] == 1:
                y_, z_e, z_q = self.model["generator"](y, *x)
            else:
                y_mb = self.criterion["pqmf"].analysis(y)
                y_, z_e, z_q = self.model["generator"](y_mb, *x)
            quantize_loss = self.criterion["mse"](z_q, z_e.detach())
            commit_loss = self.criterion["mse"](z_e, z_q.detach())
        elif self.use_duration_prediction:
            assert ds is not None
            y_, ds_ = self.model["generator"](x, ds)
            duration_loss = self.criterion["duration"](ds_, torch.log(ds))
        else:
            y_ = self.model["generator"](*x)
        if self.config["generator_params"]["out_channels"] > 1:
            y_mb_ = y_
            y_ = self.criterion["pqmf"].synthesis(y_mb_)

        # initialize
        aux_loss = 0.0

        # multi-resolution stft loss
        if self.config["use_stft_loss"]:
            sc_loss, mag_loss = self.criterion["stft"](y_, y)
            aux_loss += sc_loss + mag_loss
            self.total_eval_loss["eval/spectral_convergence_loss"] += sc_loss.item()
            self.total_eval_loss["eval/log_stft_magnitude_loss"] += mag_loss.item()

        # subband multi-resolution stft loss
        if self.config.get("use_subband_stft_loss", False):
            aux_loss *= 0.5  # for balancing with subband stft loss
            if not self.is_vq:
                y_mb = self.criterion["pqmf"].analysis(y)
            sub_sc_loss, sub_mag_loss = self.criterion["sub_stft"](y_mb_, y_mb)
            self.total_eval_loss[
                "eval/sub_spectral_convergence_loss"
            ] += sub_sc_loss.item()
            self.total_eval_loss[
                "eval/sub_log_stft_magnitude_loss"
            ] += sub_mag_loss.item()
            aux_loss += 0.5 * (sub_sc_loss + sub_mag_loss)

        # mel spectrogram loss
        if self.config["use_mel_loss"]:
            mel_loss = self.criterion["mel"](y_, y)
            aux_loss += mel_loss
            self.total_eval_loss["eval/mel_loss"] += mel_loss.item()

        # weighting stft loss
        aux_loss *= self.config.get("lambda_aux", 1.0)

        # adversarial loss
        p_ = self.model["discriminator"](y_)
        adv_loss = self.criterion["gen_adv"](p_)
        gen_loss = aux_loss + self.config["lambda_adv"] * adv_loss

        # feature matching loss
        if self.config["use_feat_match_loss"]:
            p = self.model["discriminator"](y)
            fm_loss = self.criterion["feat_match"](p_, p)
            self.total_eval_loss["eval/feature_matching_loss"] += fm_loss.item()
            gen_loss += (
                self.config["lambda_adv"] * self.config["lambda_feat_match"] * fm_loss
            )

        if self.config["use_speaker_contra_loss"]:
            # speaker contrastive loss
            spemb_ref = x_ref[2]["spemb"]
            del x_ref, y_ref # TODO(jhan): May remove this line, and delete ds_ref if exists
            spk_loss = self.criterion["speaker_contra"](y_, spemb_ref)
            gen_loss += self.config["lambda_speaker_contra"] * spk_loss
            self.total_eval_loss["eval/speaker_contra_loss"] += spk_loss.item()

        #######################
        #    Discriminator    #
        #######################
        p = self.model["discriminator"](y)
        p_ = self.model["discriminator"](y_)

        # discriminator loss
        real_loss, fake_loss = self.criterion["dis_adv"](p_, p)
        dis_loss = real_loss + fake_loss

        # add to total eval loss
        self.total_eval_loss["eval/adversarial_loss"] += adv_loss.item()
        self.total_eval_loss["eval/generator_loss"] += gen_loss.item()
        self.total_eval_loss["eval/real_loss"] += real_loss.item()
        self.total_eval_loss["eval/fake_loss"] += fake_loss.item()
        self.total_eval_loss["eval/discriminator_loss"] += dis_loss.item()
        if self.is_vq:
            self.total_eval_loss["eval/quantization_loss"] += quantize_loss.item()
            self.total_eval_loss["eval/commitment_loss"] += commit_loss.item()
        if self.use_duration_prediction:
            self.total_eval_loss["eval/duration_loss"] += duration_loss.item()

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        for key in self.model.keys():
            self.model[key].eval()

        # do not change batch for evaluation
        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(
            tqdm(self.data_loader["dev"], desc="[eval]"), 1
        ):
            # eval one step
            self._eval_step(batch)

            # save intermediate result
            if eval_steps_per_epoch == 1:
                self._genearete_and_save_intermediate_result(batch)

        logging.info(
            f"(Steps: {self.steps}) Finished evaluation "
            f"({eval_steps_per_epoch} steps per epoch)."
        )

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logging.info(
                f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}."
            )

        # record
        self._write_to_tensorboard(self.total_eval_loss)

        # reset
        self.total_eval_loss = defaultdict(float)

        # restore mode
        for key in self.model.keys():
            self.model[key].train()

    @torch.no_grad()
    def _genearete_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        # delayed import to avoid error related backend error
        import matplotlib.pyplot as plt

        is_batch_pair = (
            self.data_loader["dev"].batch_sampler.__class__.__name__
            == "CategoryPairSampler"
        )
        # parse batch and send to device
        if self.use_duration_prediction:
            x_batch, y_batch, _ = self._parse_batch(batch, is_batch_list=is_batch_pair)
        else:
            x_batch, y_batch = self._parse_batch(batch, is_batch_list=is_batch_pair)
        if is_batch_pair:
            x_batch, _ = x_batch
            y_batch, _ = y_batch

        # generate
        if self.is_vq:
            if self.config["generator_params"]["in_channels"] == 1:
                y_batch_, _, _ = self.model["generator"](y_batch, *x_batch)
            else:
                y_batch_, _, _ = self.model["generator"](
                    self.criterion["pqmf"].analysis(y_batch), *x_batch
                )
        elif self.use_duration_prediction:
            y_batch_, _ = self.model["generator"].synthesis(x_batch)
        else:
            y_batch_ = self.model["generator"](*x_batch)
        if self.config["generator_params"]["out_channels"] > 1:
            y_batch_ = self.criterion["pqmf"].synthesis(y_batch_)

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (y, y_) in enumerate(zip(y_batch, y_batch_), 1):
            # convert to ndarray
            y, y_ = y.view(-1).cpu().numpy(), y_.view(-1).cpu().numpy()

            # plot figure and save it
            figname = os.path.join(dirname, f"{idx}.png")
            plt.subplot(2, 1, 1)
            plt.plot(y)
            plt.title("groundtruth speech")
            plt.subplot(2, 1, 2)
            plt.plot(y_)
            plt.title(f"generated speech @ {self.steps} steps")
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

            # save as wavfile
            y = np.clip(y, -1, 1)
            y_ = np.clip(y_, -1, 1)
            sf.write(
                figname.replace(".png", "_ref.wav"),
                y,
                self.config["sampling_rate"],
                "PCM_16",
            )
            sf.write(
                figname.replace(".png", "_gen.wav"),
                y_,
                self.config["sampling_rate"],
                "PCM_16",
            )

            if idx >= self.config["num_save_intermediate_results"]:
                break

    def _parse_batch(self, batch, is_batch_list=False):
        """Parse batch and send to the device."""
        # parse batch
        if is_batch_list:
            return tuple(
                zip(*(self._parse_batch(b, is_batch_list=False) for b in batch))
            )
        if self.use_duration_prediction:
            inputs, targets, durations = batch
        else:
            inputs, targets = batch

        # send inputs to device
        if isinstance(inputs, torch.Tensor):
            x = inputs.to(self.device)
        elif isinstance(inputs, (tuple, list)):
            x = []
            for x_item in inputs:
                if x_item is None:
                    x.append(None)
                elif isinstance(x_item, dict):
                    if x_item != {}: # remove empty additional features for compatibility
                        for key, value in x_item.items():
                            x_item[key] = value.to(self.device)
                        x.append(x_item)
                else:
                    x.append(x_item.to(self.device))
        else:
            raise ValueError(f"Not supported type ({type(inputs)}).")

        # send targets to device
        if isinstance(targets, torch.Tensor):
            y = targets.to(self.device)
        elif isinstance(targets, (tuple, list)):
            y = [None if y is None else y.to(self.device) for y in targets]
        else:
            raise ValueError(f"Not supported type ({type(targets)}).")

        if self.use_duration_prediction:
            # send durations to device (for model with duration prediction only)
            if isinstance(durations, torch.Tensor):
                ds = durations.to(self.device)
            elif isinstance(durations, (tuple, list)):
                ds = [None if d is None else d.to(self.device) for d in durations]
            else:
                raise ValueError(f"Not supported type ({type(durations)}).")

            return x, y, ds

        return x, y

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config["save_interval_steps"] == 0:
            self.save_checkpoint(
                os.path.join(self.config["outdir"], f"checkpoint-{self.steps}steps.pkl")
            )
            logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config["log_interval_steps"]
                logging.info(
                    f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}."
                )
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True


class Collater(object):
    """Customized collater for Pytorch DataLoader in training."""

    def __init__(
        self,
        batch_max_steps=20480,
        hop_size=256,
        aux_context_window=2,
        use_noise_input=False,
        use_f0=False,
        use_f0_and_excitation=False,
        use_aux_input=True,
        use_duration=False,
        use_global_condition=False,
        use_local_condition=False,
        pad_value=0,
        batch_partition_sizes=None,
    ):
        """Initialize customized collater for PyTorch DataLoader.

        Args:
            batch_max_steps (int): The maximum length of input signal in batch.
            hop_size (int): Hop size of auxiliary features.
            aux_context_window (int): Context window size for auxiliary feature conv.
            use_noise_input (bool): Whether to use noise input.
            use_f0_and_excitation (bool): Whether to use f0 and ext. input.
            use_aux_input (bool): Whether to use auxiliary input.
            use_duration (bool): Whether to use duration for duration prediction.
            use_global_condition (bool): Whether to use global conditioning.
            use_local_condition (bool): Whether to use local conditioning.
            batch_partition_sizes (list): List of batch size splits.

        """
        if hop_size is not None:
            if batch_max_steps % hop_size != 0:
                batch_max_steps += -(batch_max_steps % hop_size)
            assert batch_max_steps % hop_size == 0
            self.hop_size = hop_size
            self.batch_max_frames = batch_max_steps // hop_size
        self.batch_max_steps = batch_max_steps
        self.aux_context_window = aux_context_window
        self.use_noise_input = use_noise_input
        self.use_f0 = use_f0
        self.use_f0_and_excitation = use_f0_and_excitation
        self.use_aux_input = use_aux_input
        self.use_duration = use_duration
        self.use_global_condition = use_global_condition
        self.use_local_condition = use_local_condition
        self.pad_value = pad_value
        self.batch_partition_sizes = batch_partition_sizes
        if not self.use_aux_input:
            assert not self.use_noise_input, "Not supported."
            assert not self.use_duration, "Not supported."
        if self.use_noise_input:
            assert not self.use_duration, "Not supported."
        if self.use_local_condition:
            assert not self.use_aux_input and not self.use_duration, "Not supported."
        if self.use_global_condition:
            assert not self.use_aux_input and not self.use_duration, "Not supported."

        # set useful values in random cutting
        if self.use_aux_input or self.use_local_condition:
            self.start_offset = aux_context_window
            self.end_offset = -(self.batch_max_frames + aux_context_window)
            self.mel_threshold = self.batch_max_frames + 2 * aux_context_window
        else:
            self.start_offset = 0
            self.end_offset = -self.batch_max_steps
            self.audio_threshold = self.batch_max_steps

    def __call__(self, batch):
        """Convert into batch tensors.

        Args:
            batch (list): list of tuple of the pair of audio and features, with optional additional features appended.

        Returns:
            Tuple: Tuple of Gaussian noise batch (B, 1, T) and auxiliary feature
                batch (B, C, T'), where T = (T' - 2 * aux_context_window) * hop_size.
                If use_noise_input = False, Gaussian noise batch is not included.
                If use_aux_input = False, auxiliary feature batch is not included.
                If both use_noise_input and use_aux_input to False, this tuple is
                not returned.
            Tensor: Target signal batch (B, 1, T).

        """
        if self.batch_partition_sizes is None:
            return self._collate_batch(batch)

        start = 0
        batches = []
        for length in self.batch_partition_sizes:
            end = start + length
            batch_ = batch[start:end]
            batches.append(self._collate_batch(batch_))
            start = end
        return batches

    def _collate_batch(self, batch):
        if self.use_aux_input:
            #################################
            #          MEL2WAV CASE         #
            #################################
            # check length
            batch = [
                self._adjust_length(*b) for b in batch if len(b[1]) > self.mel_threshold
            ]
            xs, cs = [b[0] for b in batch], [b[1] for b in batch]
            if self.use_f0:
                fs = [b[2] for b in batch]
                additional_features = {
                    feature: [b[3][feature] for b in batch]
                    for feature in batch[0][3].keys()
                }
            if self.use_f0_and_excitation:
                fs, es = [b[2] for b in batch], [b[3] for b in batch]

            # make batch with random cut
            c_lengths = [len(c) for c in cs]
            start_frames = np.array(
                [
                    np.random.randint(self.start_offset, cl + self.end_offset)
                    for cl in c_lengths
                ]
            )
            x_starts = start_frames * self.hop_size
            x_ends = x_starts + self.batch_max_steps
            c_starts = start_frames - self.aux_context_window
            c_ends = start_frames + self.batch_max_frames + self.aux_context_window
            y_batch = [x[start:end] for x, start, end in zip(xs, x_starts, x_ends)]
            c_batch = [c[start:end] for c, start, end in zip(cs, c_starts, c_ends)]

            # convert each batch to tensor, asuume that each item in batch has the same length
            y_batch, c_batch = np.array(y_batch), np.array(c_batch)
            y_batch = torch.tensor(y_batch, dtype=torch.float).unsqueeze(1)  # (B, 1, T)

            if self.use_f0_and_excitation:
                f_batch = [f[start:end] for f, start, end in zip(fs, c_starts, c_ends)]
                e_batch = [e[start:end] for e, start, end in zip(es, c_starts, c_ends)]
                f_batch, e_batch = np.array(f_batch), np.array(e_batch)
                f_batch = torch.tensor(f_batch, dtype=torch.float).unsqueeze(
                    1
                )  # (B, 1, T')
                e_batch = torch.tensor(e_batch, dtype=torch.float)  # (B, 1, T', C')
                e_batch = e_batch.reshape(e_batch.shape[0], 1, -1)  # (B, 1, T' * C')

            if self.use_f0:
                f_batch = [f[start:end] for f, start, end in zip(fs, c_starts, c_ends)]
                f_batch = np.array(f_batch)
                f_batch = torch.tensor(f_batch, dtype=torch.float).unsqueeze(
                    1
                )  # (B, 1, T')
                additional_batch = {
                    key: torch.tensor(
                        np.array(
                            [
                                feat[start:end]
                                for feat, start, end in zip(features, c_starts, c_ends)
                            ]
                        )
                        if key.endswith(
                            "_t0"
                        )  # for sequential features in time, where the first dimension is time
                        else np.array(features)  # for non-sequential features
                    )
                    for key, features in additional_features.items()
                }

            # duration calculation and return with duration information
            if self.use_duration:
                updated_c_batch, d_batch = [], []
                for c in c_batch:
                    # NOTE(jiatong): assume 0 is the discrete symbol
                    # (refer to cvss_c/local/preprocess_hubert.py)
                    code, d = torch.unique_consecutive(
                        torch.tensor(c, dtype=torch.long), return_counts=True, dim=0
                    )
                    updated_c_batch.append(code)
                    d_batch.append(d)
                c_batch = self._pad_list(updated_c_batch, self.pad_value).transpose(
                    2, 1
                )  # (B, C, T')
                d_batch = self._pad_list(d_batch, 0)
                return c_batch, y_batch, d_batch

            # process data without duration prediction
            c_batch = torch.tensor(c_batch, dtype=torch.float).transpose(
                2, 1
            )  # (B, C, T')

            input_items = (c_batch,)
            if self.use_noise_input:
                # make input noise signal batch tensor
                z_batch = torch.randn(y_batch.size())  # (B, 1, T)
                input_items = (z_batch,) + input_items
            if self.use_f0_and_excitation:
                input_items = input_items + (f_batch, e_batch)
            if self.use_f0:
                input_items = input_items + (f_batch, additional_batch)

            return input_items, y_batch
        else:
            #################################
            #        VQ-WAV2WAV CASE        #
            #################################
            if self.use_local_condition:
                # check length
                batch_idx = [
                    idx
                    for idx, b in enumerate(batch)
                    if len(b[1]) >= self.mel_threshold
                ]

                # fix length
                batch_ = [
                    self._adjust_length(batch[idx][0], batch[idx][1])
                    for idx in batch_idx
                ]

                # decide random index
                l_lengths = [len(b[1]) for b in batch_]
                l_starts = np.array(
                    [
                        np.random.randint(self.start_offset, ll + self.end_offset)
                        for ll in l_lengths
                    ]
                )
                l_ends = l_starts + self.batch_max_frames
                y_starts = l_starts * self.hop_size
                y_ends = y_starts + self.batch_max_steps

                # make random batch
                y_batch = [
                    b[0][start:end] for b, start, end in zip(batch_, y_starts, y_ends)
                ]
                l_batch = [
                    b[1][start:end] for b, start, end in zip(batch_, l_starts, l_ends)
                ]
                if self.use_global_condition:
                    g_batch = [batch[idx][2].reshape(1) for idx in batch_idx]
            else:
                # check length
                if self.use_global_condition:
                    batch = [b for b in batch if len(b[0]) >= self.audio_threshold]
                else:
                    batch = [(b,) for b in batch if len(b) >= self.audio_threshold]

                # decide random index
                y_lengths = [len(b[0]) for b in batch]
                y_starts = np.array(
                    [
                        np.random.randint(self.start_offset, yl + self.end_offset)
                        for yl in y_lengths
                    ]
                )
                y_ends = y_starts + self.batch_max_steps

                # make random batch
                y_batch = [
                    b[0][start:end] for b, start, end in zip(batch, y_starts, y_ends)
                ]
                if self.use_global_condition:
                    g_batch = [b[1].reshape(1) for b in batch]

            # convert each batch to tensor, asuume that each item in batch has the same length
            y_batch = torch.tensor(y_batch, dtype=torch.float).unsqueeze(1)  # (B, 1, T)
            if self.use_local_condition:
                l_batch = torch.tensor(l_batch, dtype=torch.float).transpose(
                    2, 1
                )  # (B, C' T')
            else:
                l_batch = None
            if self.use_global_condition:
                g_batch = torch.tensor(g_batch, dtype=torch.long).view(-1)  # (B,)
            else:
                g_batch = None

            # NOTE(kan-bayashi): Always return "l" and "g" since VQ-VAE can accept None
            return (l_batch, g_batch), y_batch

    def _adjust_length(self, x, c, f0=None, excitation=None):
        """Adjust the audio and feature lengths.

        Note:
            Basically we assume that the length of x and c are adjusted
            through preprocessing stage, but if we use other library processed
            features, this process will be needed.

        """
        if len(x) < len(c) * self.hop_size:
            x = np.pad(x, (0, len(c) * self.hop_size - len(x)), mode="edge")

        # check the legnth is valid
        assert len(x) == len(c) * self.hop_size

        if f0 is not None and excitation is not None:
            return x, c, f0, excitation
        elif f0 is not None and excitation is None:
            return x, c, f0
        else:
            return x, c

    def _pad_list(self, xs, pad_value):
        """Perform padding for the list of tensors.

        Args:
            xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
            pad_value (float): Value for padding.

        Returns:
            Tensor: Padded tensor (B, Tmax, `*`).

        Examples:
            >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
            >>> x
            [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
            >>> pad_list(x, 0)
            tensor([[1., 1., 1., 1.],
                    [1., 1., 0., 0.],
                    [1., 0., 0., 0.]])

        """
        n_batch = len(xs)
        max_len = max(x.size(0) for x in xs)
        pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

        for i in range(n_batch):
            pad[i, : xs[i].size(0)] = xs[i]

        return pad


class Collater_MR(Collater):
    """Customized collater for Pytorch DataLoader in training.
        For multi-resolution specified.
    """

    def __init__(
        self,
        batch_max_steps=20480,
        hop_size=256,
        aux_context_window=2,
        use_noise_input=False,
        use_f0=False,
        use_f0_and_excitation=False,
        use_multi_resolution_token=False,
        use_aux_input=True,
        use_duration=False,
        use_global_condition=False,
        use_local_condition=False,
        pad_value=0,
    ):
        """Initialize customized collater for PyTorch DataLoader.

        Args:
            batch_max_steps (int): The maximum length of input signal in batch.
            hop_size (int): Hop size of auxiliary features.
            aux_context_window (int): Context window size for auxiliary feature conv.
            use_noise_input (bool): Whether to use noise input.
            use_f0_and_excitation (bool): Whether to use f0 and ext. input.
            use_aux_input (bool): Whether to use auxiliary input.
            use_duration (bool): Whether to use duration for duration prediction.
            use_global_condition (bool): Whether to use global conditioning.
            use_local_condition (bool): Whether to use local conditioning.

        """
        super().__init__(
            batch_max_steps=batch_max_steps,
            hop_size=hop_size,
            aux_context_window=aux_context_window,
            use_noise_input=use_noise_input,
            use_f0=use_f0,
            use_f0_and_excitation=use_f0_and_excitation,
            use_aux_input=use_aux_input,
            use_duration=use_duration,
            use_global_condition=use_global_condition,
            use_local_condition=use_local_condition,
            pad_value=pad_value,
        )
        self.use_multi_resolution_token = use_multi_resolution_token

    def __call__(self, batch):
        """Convert into batch tensors.

        Args:
            batch (list): list of tuple of the pair of audio and features.

        Returns:
            Tuple: Tuple of Gaussian noise batch (B, 1, T) and auxiliary feature
                batch (B, C, T'), where T = (T' - 2 * aux_context_window) * hop_size.
                If use_noise_input = False, Gaussian noise batch is not included.
                If use_aux_input = False, auxiliary feature batch is not included.
                If both use_noise_input and use_aux_input to False, this tuple is
                not returned.
            Tensor: Target signal batch (B, 1, T).

        """
        if self.use_aux_input:
            #################################
            #          MEL2WAV CASE         #
            #################################
            # check length
            resolution = batch[0][1][0]
            src_rs = resolution[0]
            batch = [
                self._adjust_length(*b)
                for b in batch
                if len(b[1][src_rs]) > self.mel_threshold
            ]
            xs = [b[0] for b in batch]
            cs_rs = {}
            for rs in resolution:
                cs = [b[1][rs] for b in batch]
                cs_rs[rs] = cs
            if self.use_f0:
                fs = [b[2] for b in batch]
            if self.use_f0_and_excitation:
                fs, es = [b[2] for b in batch], [b[3] for b in batch]

            # make batch with random cut
            cs = cs_rs[src_rs]
            c_lengths = [len(c) for c in cs]
            start_frames = np.array(
                [
                    np.random.randint(self.start_offset, cl + self.end_offset)
                    for cl in c_lengths
                ]
            )
            c_starts = start_frames - self.aux_context_window
            c_ends = start_frames + self.batch_max_frames + self.aux_context_window
            select_len = self.batch_max_frames + self.aux_context_window
            c_batch = {}
            for rs in resolution:
                scl = rs / src_rs
                c_starts_rs = (c_starts // scl).astype(np.int32)
                min_len = np.ceil((select_len) / scl).astype(np.int32)
                c_ends_rs = c_starts_rs + min_len
                cs = cs_rs[rs]
                c_batch[rs] = np.array(
                    [c[start:end] for c, start, end in zip(cs, c_starts_rs, c_ends_rs)]
                )

            x_starts = start_frames * self.hop_size
            x_ends = x_starts + self.batch_max_steps
            y_batch = [x[start:end] for x, start, end in zip(xs, x_starts, x_ends)]
            y_batch = np.array(y_batch)
            y_batch = torch.tensor(y_batch, dtype=torch.float).unsqueeze(1)  # (B, 1, T)

            # convert each batch to tensor, asuume that each item in batch has the same length
            if self.use_f0_and_excitation:
                f_batch = [f[start:end] for f, start, end in zip(fs, c_starts, c_ends)]
                e_batch = [e[start:end] for e, start, end in zip(es, c_starts, c_ends)]
                f_batch, e_batch = np.array(f_batch), np.array(e_batch)
                f_batch = torch.tensor(f_batch, dtype=torch.float).unsqueeze(
                    1
                )  # (B, 1, T')
                e_batch = torch.tensor(e_batch, dtype=torch.float)  # (B, 1, T', C')
                e_batch = e_batch.reshape(e_batch.shape[0], 1, -1)  # (B, 1, T' * C')

            if self.use_f0:
                f_batch = [f[start:end] for f, start, end in zip(fs, c_starts, c_ends)]
                f_batch = np.array(f_batch)
                f_batch = torch.tensor(f_batch, dtype=torch.float).unsqueeze(
                    1
                )  # (B, 1, T')

            # process data without duration prediction
            for rs in resolution:
                c_batch[rs] = torch.tensor(c_batch[rs], dtype=torch.float).transpose(1, -1)  # (B, C, T')
                # logging.info(f'c-{rs}: {c_batch[rs].shape}')

            input_items = (c_batch,)
            if self.use_f0_and_excitation:
                input_items = input_items + (f_batch, e_batch)
            if self.use_f0:
                input_items = input_items + (f_batch,)

            return input_items, y_batch
        else:
            raise NotImplementedError(
                "Only mel2wav case is accepted for multi-resolution feature."
            )

    def _adjust_length(self, x, c, f0=None, excitation=None):
        """Adjust the audio and feature lengths.

        Note:
            Basically we assume that the length of x and c are adjusted
            through preprocessing stage, but if we use other library processed
            features, this process will be needed.

        """
        resolution = c[0]
        src_rs = resolution[0]
        if len(x) < len(c[src_rs]) * self.hop_size:
            x = np.pad(x, (0, len(c[src_rs]) * self.hop_size - len(x)), mode="edge")

        # check the legnth is valid
        assert len(x) == len(c[src_rs]) * self.hop_size

        if f0 is not None and excitation is not None:
            return x, c, f0, excitation
        elif f0 is not None and excitation is None:
            return x, c, f0
        else:
            return x, c


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description=(
            "Train Parallel WaveGAN (See detail in parallel_wavegan/bin/train.py)."
        )
    )
    parser.add_argument(
        "--train-wav-scp",
        default=None,
        type=str,
        help=(
            "kaldi-style wav.scp file for training. "
            "you need to specify either train-*-scp or train-dumpdir."
        ),
    )
    parser.add_argument(
        "--train-feats-scp",
        default=None,
        type=str,
        help=(
            "kaldi-style feats.scp file for training. "
            "you need to specify either train-*-scp or train-dumpdir."
        ),
    )
    parser.add_argument(
        "--train-segments",
        default=None,
        type=str,
        help="kaldi-style segments file for training.",
    )
    parser.add_argument(
        "--train-dumpdir",
        default=None,
        type=str,
        help=(
            "directory including training data. "
            "you need to specify either train-*-scp or train-dumpdir."
        ),
    )
    parser.add_argument(
        "--dev-wav-scp",
        default=None,
        type=str,
        help=(
            "kaldi-style wav.scp file for validation. "
            "you need to specify either dev-*-scp or dev-dumpdir."
        ),
    )
    parser.add_argument(
        "--dev-feats-scp",
        default=None,
        type=str,
        help=(
            "kaldi-style feats.scp file for vaidation. "
            "you need to specify either dev-*-scp or dev-dumpdir."
        ),
    )
    parser.add_argument(
        "--dev-segments",
        default=None,
        type=str,
        help="kaldi-style segments file for validation.",
    )
    parser.add_argument(
        "--dev-dumpdir",
        default=None,
        type=str,
        help=(
            "directory including development data. "
            "you need to specify either dev-*-scp or dev-dumpdir."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="directory to save checkpoints.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml format configuration file.",
    )
    parser.add_argument(
        "--pretrain",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to load pretrained params. (default="")',
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to resume training. (default="")',
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    parser.add_argument(
        "--use-f0",
        default=False,
        action="store_true",
        help="whether to use f0 sequence.",
    )
    parser.add_argument(
        "--use-multi-resolution-token",
        default=False,
        action="store_true",
        help="whether to use multi resolution_token.",
    )
    parser.add_argument(
        "--rank",
        "--local_rank",
        default=0,
        type=int,
        help="rank for distributed training. no need to explictly specify.",
    )
    parser.add_argument(
        "--additional-feature-keys",
        default=[],  # TODO(jhan): verify if works when --additional-feature-keys is not specified
        type=str,
        nargs="*",
        help="additional feature keys to use.",
    )
    load_dict_arg = lambda x: json.loads(x.replace("'", '"'))
    parser.add_argument(
        "--train-batch-sampler-conf",
        default={},  # TODO(jhan): verify if default={} works
        type=load_dict_arg,
    )
    parser.add_argument(
        "--dev-batch-sampler-conf",
        default={},  # TODO(jhan): verify if default={} works
        type=load_dict_arg,
    )
    args = parser.parse_args()

    args.distributed = False
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(args.rank)
        # setup for distributed training
        # see example: https://github.com/NVIDIA/apex/tree/master/examples/simple/distributed
        if "WORLD_SIZE" in os.environ:
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.distributed = args.world_size > 1
        if args.distributed:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # suppress logging for distributed training
    if args.rank != 0:
        sys.stdout = open(os.devnull, "w")

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # check arguments
    if (args.train_feats_scp is not None and args.train_dumpdir is not None) or (
        args.train_feats_scp is None and args.train_dumpdir is None
    ):
        raise ValueError("Please specify either --train-dumpdir or --train-*-scp.")
    if (args.dev_feats_scp is not None and args.dev_dumpdir is not None) or (
        args.dev_feats_scp is None and args.dev_dumpdir is None
    ):
        raise ValueError("Please specify either --dev-dumpdir or --dev-*-scp.")

    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    config["version"] = parallel_wavegan.__version__  # add version info
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # get configuration
    generator_type = config.get("generator_type", "ParallelWaveGANGenerator")
    use_aux_input = "VQVAE" not in generator_type
    use_noise_input = (
        "ParallelWaveGAN" in generator_type and "VQVAE" not in generator_type
    )
    use_duration = "Duration" in generator_type
    use_local_condition = config.get("use_local_condition", False)
    use_global_condition = config.get("use_global_condition", False)
    use_f0_and_excitation = generator_type == "UHiFiGANGenerator"

    # setup query and load function
    if args.train_wav_scp is None or args.dev_wav_scp is None:
        local_query = None
        local_load_fn = None
        global_query = None
        global_load_fn = None
        if config["format"] == "hdf5":
            audio_query, mel_query = "*.h5", "*.h5"
            audio_load_fn = lambda x: read_hdf5(x, "wave")  # NOQA
            mel_load_fn = lambda x: read_hdf5(x, "feats")  # NOQA
            if use_f0_and_excitation:
                f0_query, excitation_query = "*.h5", "*.h5"
                f0_load_fn = lambda x: read_hdf5(x, "f0")  # NOQA
                excitation_load_fn = lambda x: read_hdf5(x, "excitation")  # NOQA
            if args.use_f0:
                f0_query = "*.h5"
                f0_load_fn = lambda x: read_hdf5(x, "f0")  # NOQA
            if args.use_multi_resolution_token:
                resolution_query = "*.h5"
                resolution_load_fn = lambda x: read_hdf5(x, "resolution")
            additional_feature_query_load_fn = {
                key: ("*.h5", lambda x: read_hdf5(x, key))
                for key in args.additional_feature_keys
            }
            if use_local_condition:
                local_query = "*.h5"
                local_load_fn = lambda x: read_hdf5(x, "local")  # NOQA
            if use_global_condition:
                global_query = "*.h5"
                global_load_fn = lambda x: read_hdf5(x, "global")  # NOQA
        elif config["format"] == "npy":
            audio_query, mel_query = "*-wave.npy", "*-feats.npy"
            audio_load_fn = np.load
            mel_load_fn = np.load
            if use_f0_and_excitation:
                f0_query, excitation_query = "*-f0.npy", "*-excitation.npy"
                f0_load_fn = np.load
                excitation_load_fn = np.load
            if args.use_f0:
                f0_query = "*-f0.npy"
                f0_load_fn = np.load
            if use_local_condition:
                local_query = "*-local.npy"
                local_load_fn = np.load
            if use_global_condition:
                global_query = "*-global.npy"
                global_load_fn = np.load
            additional_feature_query_load_fn = {
                key: (f"*-{key}.npy", np.load) for key in args.additional_feature_keys
            }
        else:
            raise ValueError("support only hdf5 or npy format.")

    # setup length threshold
    if config["remove_short_samples"]:
        audio_length_threshold = config["batch_max_steps"]
        mel_length_threshold = config["batch_max_steps"] // config[
            "hop_size"
        ] + 2 * config["generator_params"].get("aux_context_window", 0)
    else:
        mel_length_threshold = None
        audio_length_threshold = None

    # define dataset for training data
    if args.train_dumpdir is not None:
        if args.use_f0:
            train_dataset = AudioMelF0Dataset(
                root_dir=args.train_dumpdir,
                audio_query=audio_query,
                mel_query=mel_query,
                f0_query=f0_query,
                audio_load_fn=audio_load_fn,
                mel_load_fn=mel_load_fn,
                f0_load_fn=f0_load_fn,
                mel_length_threshold=mel_length_threshold,
                additional_feature_query_load_fn=additional_feature_query_load_fn,
                allow_cache=config.get("allow_cache", False),  # keep compatibility
            )
        elif not use_f0_and_excitation:
            if use_aux_input:
                if args.use_multi_resolution_token:
                    train_dataset = AudioMRTokenlDataset(
                        root_dir=args.train_dumpdir,
                        mel_query=mel_query,
                        resolution_query=resolution_query,
                        resolution_load_fn=resolution_load_fn,
                        local_query=local_query,
                        local_load_fn=local_load_fn,
                        global_query=global_query,
                        global_load_fn=global_load_fn,
                        mel_length_threshold=mel_length_threshold,
                        allow_cache=config.get("allow_cache", False),  # keep compatibility
                    )
                else:
                    train_dataset = AudioMelDataset(
                        root_dir=args.train_dumpdir,
                        audio_query=audio_query,
                        audio_load_fn=audio_load_fn,
                        mel_query=mel_query,
                        mel_load_fn=mel_load_fn,
                        local_query=local_query,
                        local_load_fn=local_load_fn,
                        global_query=global_query,
                        global_load_fn=global_load_fn,
                        mel_length_threshold=mel_length_threshold,
                        allow_cache=config.get("allow_cache", False),  # keep compatibility
                    )
            else:
                train_dataset = AudioDataset(
                    root_dir=args.train_dumpdir,
                    audio_query=audio_query,
                    audio_load_fn=audio_load_fn,
                    local_query=local_query,
                    local_load_fn=local_load_fn,
                    global_query=global_query,
                    global_load_fn=global_load_fn,
                    audio_length_threshold=audio_length_threshold,
                    allow_cache=config.get("allow_cache", False),  # keep compatibility
                )
        else:
            train_dataset = AudioMelF0ExcitationDataset(
                root_dir=args.train_dumpdir,
                audio_query=audio_query,
                mel_query=mel_query,
                f0_query=f0_query,
                excitation_query=excitation_query,
                audio_load_fn=audio_load_fn,
                mel_load_fn=mel_load_fn,
                f0_load_fn=f0_load_fn,
                excitation_load_fn=excitation_load_fn,
                mel_length_threshold=mel_length_threshold,
                allow_cache=config.get("allow_cache", False),  # keep compatibility
            )
    else:
        if use_f0_and_excitation:
            raise NotImplementedError(
                "SCP format is not supported for f0 and excitation."
            )
        if args.use_f0:
            raise NotImplementedError("SCP format is not supported for f0.")
        if use_local_condition:
            raise NotImplementedError("Not supported.")
        if use_global_condition:
            raise NotImplementedError("Not supported.")
        if use_aux_input:
            train_dataset = AudioMelSCPDataset(
                wav_scp=args.train_wav_scp,
                feats_scp=args.train_feats_scp,
                segments=args.train_segments,
                mel_length_threshold=mel_length_threshold,
                allow_cache=config.get("allow_cache", False),  # keep compatibility
            )
        else:
            train_dataset = AudioSCPDataset(
                wav_scp=args.train_wav_scp,
                segments=args.train_segments,
                audio_length_threshold=audio_length_threshold,
                allow_cache=config.get("allow_cache", False),  # keep compatibility
            )

    # define dataset for validation
    if args.dev_dumpdir is not None:
        if args.use_f0:
            dev_dataset = AudioMelF0Dataset(
                root_dir=args.dev_dumpdir,
                audio_query=audio_query,
                mel_query=mel_query,
                f0_query=f0_query,
                audio_load_fn=audio_load_fn,
                mel_load_fn=mel_load_fn,
                f0_load_fn=f0_load_fn,
                mel_length_threshold=mel_length_threshold,
                additional_feature_query_load_fn=additional_feature_query_load_fn,
                allow_cache=config.get("allow_cache", False),  # keep compatibility
            )
        elif not use_f0_and_excitation:
            if use_aux_input:
                if args.use_multi_resolution_token:
                    dev_dataset = AudioMRTokenlDataset(
                        root_dir=args.train_dumpdir,
                        mel_query=mel_query,
                        resolution_query=resolution_query,
                        resolution_load_fn=resolution_load_fn,
                        local_query=local_query,
                        local_load_fn=local_load_fn,
                        global_query=global_query,
                        global_load_fn=global_load_fn,
                        mel_length_threshold=mel_length_threshold,
                        allow_cache=config.get("allow_cache", False),  # keep compatibility
                    )
                else:
                    dev_dataset = AudioMelDataset(
                        root_dir=args.dev_dumpdir,
                        audio_query=audio_query,
                        audio_load_fn=audio_load_fn,
                        mel_query=mel_query,
                        mel_load_fn=mel_load_fn,
                        local_query=local_query,
                        local_load_fn=local_load_fn,
                        global_query=global_query,
                        global_load_fn=global_load_fn,
                        mel_length_threshold=mel_length_threshold,
                        allow_cache=config.get("allow_cache", False),  # keep compatibility
                    )
            else:
                dev_dataset = AudioDataset(
                    root_dir=args.dev_dumpdir,
                    audio_query=audio_query,
                    audio_load_fn=audio_load_fn,
                    local_query=local_query,
                    local_load_fn=local_load_fn,
                    global_query=global_query,
                    global_load_fn=global_load_fn,
                    audio_length_threshold=audio_length_threshold,
                    allow_cache=config.get("allow_cache", False),  # keep compatibility
                )
        else:
            dev_dataset = AudioMelF0ExcitationDataset(
                root_dir=args.dev_dumpdir,
                audio_query=audio_query,
                mel_query=mel_query,
                f0_query=f0_query,
                excitation_query=excitation_query,
                audio_load_fn=audio_load_fn,
                mel_load_fn=mel_load_fn,
                f0_load_fn=f0_load_fn,
                excitation_load_fn=excitation_load_fn,
                mel_length_threshold=mel_length_threshold,
                allow_cache=config.get("allow_cache", False),  # keep compatibility
            )
    else:
        if use_f0_and_excitation:
            raise NotImplementedError(
                "SCP format is not supported for f0 and excitation."
            )
        if args.use_f0:
            raise NotImplementedError("SCP format is not supported for f0.")
        if use_local_condition:
            raise NotImplementedError("Not supported.")
        if use_global_condition:
            raise NotImplementedError("Not supported.")
        if use_aux_input:
            dev_dataset = AudioMelSCPDataset(
                wav_scp=args.dev_wav_scp,
                feats_scp=args.dev_feats_scp,
                segments=args.dev_segments,
                mel_length_threshold=mel_length_threshold,
                allow_cache=config.get("allow_cache", False),  # keep compatibility
            )
        else:
            dev_dataset = AudioSCPDataset(
                wav_scp=args.dev_wav_scp,
                segments=args.dev_segments,
                audio_length_threshold=audio_length_threshold,
                allow_cache=config.get("allow_cache", False),  # keep compatibility
            )

    # store into dataset dict
    dataset = {
        "train": train_dataset,
        "dev": dev_dataset,
    }
    logging.info(f"The number of training files = {len(train_dataset)}.")
    logging.info(f"The number of development files = {len(dev_dataset)}.")

    # get data loader
    for x in ["train", "dev"]:
        if not args.use_multi_resolution_token:
            collater = {
                x: Collater(
                    batch_max_steps=config["batch_max_steps"],
                    hop_size=config.get("hop_size", None),
                    aux_context_window=config["generator_params"].get("aux_context_window", 0),
                    use_f0=args.use_f0,
                    use_f0_and_excitation=use_f0_and_excitation,
                    use_noise_input=use_noise_input,
                    use_aux_input=use_aux_input,
                    use_duration=use_duration,
                    use_global_condition=use_global_condition,
                    use_local_condition=use_local_condition,
                    pad_value=config["generator_params"].get(
                        "num_embs", 0
                    ),  # assume 0-based discrete symbol
                    batch_partition_sizes=(
                        [config["batch_size"], config["batch_size"]]
                        if config.get(f"{x}_batch_sampler_type", None) == "CategoryPairSampler"
                        else None
                    ),
                )
                for x in ["train", "dev"]
            }
        else:
            assert (
                "train_batch_sampler_type" not in config
                and "dev_batch_sampler_type" not in config
            ), "Batch sampler is not supported for now in multi-resolution token."
            collater = {
                x: Collater_MR(
                    batch_max_steps=config["batch_max_steps"],
                    hop_size=config.get("hop_size", None),
                    aux_context_window=config["generator_params"].get("aux_context_window", 0),
                    use_f0=args.use_f0,
                    use_f0_and_excitation=use_f0_and_excitation,
                    use_multi_resolution_token=args.use_multi_resolution_token,
                    use_noise_input=use_noise_input,
                    use_aux_input=use_aux_input,
                    use_duration=use_duration,
                    use_global_condition=use_global_condition,
                    use_local_condition=use_local_condition,
                    pad_value=config["generator_params"].get(
                        "num_embs", 0
                    ),  # assume 0-based discrete symbol
                )
                for x in ["train", "dev"]
            }
        sampler = {"train": None, "dev": None}
    if args.distributed:
        # setup sampler for distributed training
        from torch.utils.data.distributed import DistributedSampler

        sampler["train"] = DistributedSampler(
            dataset=dataset["train"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )
        sampler["dev"] = DistributedSampler(
            dataset=dataset["dev"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
        )
    data_loader = {}
    for x in ["train", "dev"]:
        if config.get(f"{x}_batch_sampler_type", None):
            batch_sampler_class = getattr(
                parallel_wavegan.batch_sampler,
                config[f"{x}_batch_sampler_type"],
            )
            batch_sampler_conf = config.get(f"{x}_batch_sampler_params", {})
            batch_sampler_conf.update(getattr(args, f"{x}_batch_sampler_conf"))
            batch_sampler_conf["batch_size"] = config["batch_size"]
            batch_sampler = batch_sampler_class(
                dataset=dataset[x],
                **batch_sampler_conf,
            )
            data_loader[x] = DataLoader(
                dataset=dataset[x],
                collate_fn=collater[x],
                num_workers=config["num_workers"],
                pin_memory=config["pin_memory"],
                batch_sampler=batch_sampler,
            )
        else:
            data_loader[x] = DataLoader(
                dataset=dataset[x],
                shuffle=False if args.distributed else True,
                collate_fn=collater[x],
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                sampler=sampler[x],
                pin_memory=config["pin_memory"],
            )

    # define models
    generator_class = getattr(
        parallel_wavegan.models,
        # keep compatibility
        config.get("generator_type", "ParallelWaveGANGenerator"),
    )
    discriminator_class = getattr(
        parallel_wavegan.models,
        # keep compatibility
        config.get("discriminator_type", "ParallelWaveGANDiscriminator"),
    )
    model = {
        "generator": generator_class(
            **config["generator_params"],
        ).to(device),
        "discriminator": discriminator_class(
            **config["discriminator_params"],
        ).to(device),
    }

    # define criterions
    criterion = {
        "gen_adv": GeneratorAdversarialLoss(
            # keep compatibility
            **config.get("generator_adv_loss_params", {})
        ).to(device),
        "dis_adv": DiscriminatorAdversarialLoss(
            # keep compatibility
            **config.get("discriminator_adv_loss_params", {})
        ).to(device),
        "mse": torch.nn.MSELoss().to(device),
    }
    if config.get("use_stft_loss", True):  # keep compatibility
        config["use_stft_loss"] = True
        criterion["stft"] = MultiResolutionSTFTLoss(
            **config["stft_loss_params"],
        ).to(device)
    if config.get("use_subband_stft_loss", False):  # keep compatibility
        assert config["generator_params"]["out_channels"] > 1
        criterion["sub_stft"] = MultiResolutionSTFTLoss(
            **config["subband_stft_loss_params"],
        ).to(device)
    else:
        config["use_subband_stft_loss"] = False
    if config.get("use_feat_match_loss", False):  # keep compatibility
        criterion["feat_match"] = FeatureMatchLoss(
            # keep compatibility
            **config.get("feat_match_loss_params", {}),
        ).to(device)
    else:
        config["use_feat_match_loss"] = False
    if config.get("use_mel_loss", False):  # keep compatibility
        if config.get("mel_loss_params", None) is None:
            criterion["mel"] = MelSpectrogramLoss(
                fs=config["sampling_rate"],
                fft_size=config["fft_size"],
                hop_size=config["hop_size"],
                win_length=config["win_length"],
                window=config["window"],
                num_mels=config["num_mels"],
                fmin=config["fmin"],
                fmax=config["fmax"],
            ).to(device)
        else:
            criterion["mel"] = MelSpectrogramLoss(
                **config["mel_loss_params"],
            ).to(device)
    else:
        config["use_mel_loss"] = False
    if config.get("use_duration_loss", False):  # keep compatibility
        if config.get("duration_loss_params", None) is None:
            criterion["duration"] = DurationPredictorLoss(
                offset=config["offset"],
                reduction=config["reduction"],
            ).to(device)
        else:
            criterion["duration"] = DurationPredictorLoss(
                **config["duration_loss_params"],
            ).to(device)
    else:
        config["use_duration_loss"] = False
    config["use_speaker_contra_loss"] = config.get("use_speaker_contra_loss", False)
    if config["use_speaker_contra_loss"]:
        if "speaker_contra_loss_params" not in config:
            raise ValueError(
                "Please specify speaker_contra_loss_params in the config if use_speaker_contra_loss is True."
            )
        if "speaker_model_conf" not in config["speaker_contra_loss_params"]:
            raise ValueError(
                "Please specify speaker_model_conf in the speaker_contra_loss_params."
            )
        config["speaker_contra_loss_params"]["speaker_model_conf"]["in_sr"] = config["sampling_rate"]

        criterion["speaker_contra"] = SpeakerContrastiveLoss(
            device=device, **config["speaker_contra_loss_params"]
        )

    # define special module for subband processing
    if config["generator_params"]["out_channels"] > 1:
        criterion["pqmf"] = PQMF(
            subbands=config["generator_params"]["out_channels"],
            # keep compatibility
            **config.get("pqmf_params", {}),
        ).to(device)

    # define optimizers and schedulers
    generator_optimizer_class = getattr(
        parallel_wavegan.optimizers,
        # keep compatibility
        config.get("generator_optimizer_type", "RAdam"),
    )
    discriminator_optimizer_class = getattr(
        parallel_wavegan.optimizers,
        # keep compatibility
        config.get("discriminator_optimizer_type", "RAdam"),
    )
    optimizer = {
        "generator": generator_optimizer_class(
            model["generator"].parameters(),
            **config["generator_optimizer_params"],
        ),
        "discriminator": discriminator_optimizer_class(
            model["discriminator"].parameters(),
            **config["discriminator_optimizer_params"],
        ),
    }
    generator_scheduler_class = getattr(
        torch.optim.lr_scheduler,
        # keep compatibility
        config.get("generator_scheduler_type", "StepLR"),
    )
    discriminator_scheduler_class = getattr(
        torch.optim.lr_scheduler,
        # keep compatibility
        config.get("discriminator_scheduler_type", "StepLR"),
    )
    scheduler = {
        "generator": generator_scheduler_class(
            optimizer=optimizer["generator"],
            **config["generator_scheduler_params"],
        ),
        "discriminator": discriminator_scheduler_class(
            optimizer=optimizer["discriminator"],
            **config["discriminator_scheduler_params"],
        ),
    }
    if args.distributed:
        # wrap model for distributed training
        try:
            from apex.parallel import DistributedDataParallel
        except ImportError:
            raise ImportError(
                "apex is not installed. please check https://github.com/NVIDIA/apex."
            )
        model["generator"] = DistributedDataParallel(model["generator"])
        model["discriminator"] = DistributedDataParallel(model["discriminator"])

    # show settings
    logging.info(model["generator"])
    logging.info(model["discriminator"])
    logging.info(optimizer["generator"])
    logging.info(optimizer["discriminator"])
    logging.info(scheduler["generator"])
    logging.info(scheduler["discriminator"])
    for criterion_ in criterion.values():
        logging.info(criterion_)

    # define trainer
    trainer = Trainer(
        steps=0,
        epochs=0,
        data_loader=data_loader,
        sampler=sampler,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )

    # load pretrained parameters from checkpoint
    if len(args.pretrain) != 0:
        trainer.load_checkpoint(args.pretrain, load_only_params=True)
        logging.info(f"Successfully load parameters from {args.pretrain}.")

    # resume from checkpoint
    if len(args.resume) != 0:
        trainer.load_checkpoint(args.resume)
        logging.info(f"Successfully resumed from {args.resume}.")

    # run training loop
    try:
        trainer.run()
    finally:
        trainer.save_checkpoint(
            os.path.join(config["outdir"], f"checkpoint-{trainer.steps}steps.pkl")
        )
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
