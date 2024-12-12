# Sampler that keeps equally distributed categories (i.e., classes) within
# each minibatch. If the batch_size is smaller than the number of classes,
# all samples in the minibatch will belong to different classes.
# Cross-checked with https://github.com/clovaai/voxceleb_trainer/blob/master/
# DatasetLoader.py
# 'key_file' is just a text file which describes each sample name."
# \n\n"
#     utterance_id_a\n"
#     utterance_id_b\n"
#     utterance_id_c\n"
# \n"
# The fist column is referred, so 'shape file' can be used, too.\n\n"
#     utterance_id_a 100,80\n"
#     utterance_id_b 400,80\n"
#     utterance_id_c 512,80\n",
from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path
from typing import Iterator

from torch.utils.data import Sampler, Dataset
from typeguard import typechecked

from parallel_wavegan.utils.read_text import read_2columns_text


class CategoryBalancedSampler(Sampler):
    @typechecked
    def __init__(
        self,
        dataset: Dataset,
        category2utt_file: str | Path,
        batch_size: int,
        min_batch_size: int = 1,
        drop_last: bool = False,
        epoch: int = 1,
    ):
        assert batch_size > 0, f"Batch size must be greater than 0, but got {batch_size=}"

        self.batch_size = batch_size
        self.min_batch_size = min_batch_size
        self.drop_last = drop_last
        self.epoch = epoch
        self.utt_ids = dataset.utt_ids
        self.category2utt = {
            cat: utts.split(" ")
            for cat, utts in read_2columns_text(category2utt_file).items()
        }
        self.categories = list(self.category2utt.keys())
        if len(self.categories) >= self.batch_size:
            self.n_utt_per_category_in_batch = 1
        else:
            self.n_utt_per_category_in_batch = math.ceil(
                self.batch_size / len(self.categories)
            )
        self._generate_batches()

    def _generate_batches(self):
        random.seed(self.epoch)
        random.shuffle(self.categories)
        flattened_cats = []
        shuffled_category2utt = {}
        for cat in self.categories:
            shuffled_category2utt[cat] = self.category2utt[cat][:]
            random.shuffle(shuffled_category2utt[cat])
            flattened_cats.extend([cat] * len(shuffled_category2utt[cat]))

        rand_idx = list(range(len(flattened_cats)))
        random.shuffle(rand_idx)

        self.batch_list = []
        current_batch = []
        current_batch_stats = Counter()
        # make minibatches
        for idx in rand_idx:
            # don't allow more number of samples that belong to each category
            # than n_utt_per_category_in_batch
            if (
                current_batch_stats[flattened_cats[idx]]
                >= self.n_utt_per_category_in_batch
            ):
                continue

            current_batch.append(self.utt_ids.index(shuffled_category2utt[flattened_cats[idx]].pop()))
            current_batch_stats[flattened_cats[idx]] += 1

            # append batch to batch list
            if len(current_batch) == self.batch_size:
                self.batch_list.append(current_batch)
                current_batch = []
                current_batch_stats = Counter()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_size={self.batch_size})"
        )

    def __len__(self):
        return len(self.batch_list)

    @typechecked
    def __iter__(self) -> Iterator[list[str]]:
        return iter(self.batch_list)

    def update_epoch(self, epoch: int):
        self.epoch = epoch
        self._generate_batches()
        return self
