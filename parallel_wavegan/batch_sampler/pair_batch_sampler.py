from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, Optional
import itertools

from torch.utils.data import (
    Sampler,
    Dataset,
    SequentialSampler,
    RandomSampler,
    BatchSampler,
    SubsetRandomSampler,
)
from typeguard import typechecked

from parallel_wavegan.utils.read_text import read_2columns_text


class PairBatchSampler(Sampler):
    @typechecked
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        category2utt_file: Optional[str | Path] = None,
        reference_categories: Optional[list[str]] = None,
        min_batch_size: int = 1,
        drop_last: bool = False,
        epoch: int = 1,
        shuffle: bool = True,
    ):

        assert batch_size > 0, f"Batch size must be greater than 0, but got {batch_size=}"

        self.batch_size = batch_size
        self.min_batch_size = min_batch_size
        self.drop_last = drop_last
        self.epoch = epoch  # NOTE: Unused
        self.utt_ids = dataset.utt_ids
        print(f"Total utterances: {len(self.utt_ids)}")  # TODO: Remove debug log

        # Read and filter category-to-utterance mapping
        if category2utt_file is not None:
            category2utt = read_2columns_text(category2utt_file)
            self.category2utt = {
                cat: [utt for utt in utts.split() if utt in self.utt_ids]
                for cat, utts in category2utt.items()
            }
            # Collect reference utterance indices
            self.reference_utt_indices = [
                self.utt_ids.index(utt)
                for cat in reference_categories
                for utt in self.category2utt.get(cat, [])
            ]
        else:
            self.category2utt = {}
            self.reference_utt_indices = range(len(self.utt_ids))  # Use all utterances as reference

        # Define samplers
        self.source_sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        self.reference_sampler = SubsetRandomSampler(self.reference_utt_indices) if shuffle else SequentialSampler(self.reference_utt_indices) # FIXME(jhan): sequential sampler can be wrong, check this

        # Define batch samplers
        self.source_batch_sampler = BatchSampler(self.source_sampler, batch_size, drop_last)
        self.reference_batch_sampler = BatchSampler(self.reference_sampler, batch_size, drop_last=True)

        # Generate initial batches
        self._generate_batches()

    def _generate_batches(self):
        """Generate paired batches from source and reference samplers."""
        self.batch_list = []

        reference_batch_iterator = itertools.islice(
            itertools.cycle(self.reference_batch_sampler),
            len(self.source_batch_sampler),
        )

        for i, (source_batch, reference_batch) in enumerate(zip(self.source_batch_sampler, reference_batch_iterator)):
            self.batch_list.append(source_batch + reference_batch)

        # Handle the last batch when drop_last=False
        if not self.drop_last and self.batch_list and len(self.batch_list[-1]) < self.batch_size * 2:
            self.batch_list[-1] = self.batch_list[-1][:(len(self.batch_list[-1]) - self.batch_size) * 2]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_batches={len(self)}, batch_size={self.batch_size})"

    def __len__(self):
        return len(self.batch_list)

    @typechecked
    def __iter__(self) -> Iterator[list[int]]:
        return iter(self.batch_list)

    def update_epoch(self, epoch: int):
        """Update the epoch number and regenerate batches."""
        self.epoch = epoch  # NOTE: Unused
        self._generate_batches()
        return self
