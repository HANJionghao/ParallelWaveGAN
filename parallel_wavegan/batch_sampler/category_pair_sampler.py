from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

from torch.utils.data import Sampler, Dataset
from typeguard import typechecked

from .category_balanced_sampler import CategoryBalancedSampler
from parallel_wavegan.utils.read_text import read_2columns_text


class CategoryPairSampler(Sampler):
    @typechecked
    def __init__(
        self,
        dataset: Dataset,
        category2utt_file: str | Path,
        utt2category_file: str | Path,
        batch_size: int,
        min_batch_size: int = 1,
        drop_last: bool = False,
        epoch: int = 1,
    ):
        assert (
            batch_size > 0
        ), f"Batch size must be greater than 0, but got {batch_size=}"
        self.batch_size = batch_size
        self.min_batch_size = min_batch_size
        self.drop_last = drop_last
        self.epoch = epoch
        self.utt_ids = dataset.utt_ids
        self.category_balanced_sampler = CategoryBalancedSampler(
            dataset=dataset,
            category2utt_file=category2utt_file,
            batch_size=self.batch_size,
            min_batch_size=self.min_batch_size,
            drop_last=self.drop_last,
            epoch=self.epoch,
        )
        self.category2utt = self.category_balanced_sampler.category2utt
        self.utt2category = read_2columns_text(utt2category_file)
        self._generate_batches()

    def _generate_batches(self):
        self.batch_list = self.category_balanced_sampler.batch_list

        random.seed(self.epoch)
        shuffled_category2utt = {}
        for category in self.category2utt:
            shuffled_category2utt[category] = self.category2utt[category][:]
            random.shuffle(shuffled_category2utt[category])

        for idx, batch in enumerate(self.batch_list):
            for utt_id in batch:
                utt = self.utt_ids[utt_id]
                if utt not in self.utt2category:
                    raise ValueError(f"utterance {utt} not found in utt2category")
                category = self.utt2category[utt]
                if category not in shuffled_category2utt:
                    raise ValueError(f"category {category} not found in category2utt")
                selected_utt = shuffled_category2utt[category].pop()
                self.batch_list[idx] = list(self.batch_list[idx])
                self.batch_list[idx].append(self.utt_ids.index(selected_utt))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(N-batch={len(self)}, "
            f"batch_size={self.batch_size})"
        )

    def __len__(self):
        return len(self.batch_list)

    @typechecked
    def __iter__(self) -> Iterator[list[str]]:
        return iter(self.batch_list)

    def update_epoch(self, epoch: int):
        self.epoch = epoch
        self.category_balanced_sampler.update_epoch(epoch)
        self._generate_batches()
        return self
