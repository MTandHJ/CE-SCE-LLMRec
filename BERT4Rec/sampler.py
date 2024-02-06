

from typing import Optional, Tuple, List

import numpy as np

import torchdata.datapipes as dp

from freerec.data.postprocessing.sampler import SeqTrainUniformSampler
from freerec.data.datasets.base import RecDataSet
from freerec.data.utils import negsamp_vectorized_bsearch


__all__ = ['SeqTrainSampler']


@dp.functional_datapipe("seq_train_sampling_")
class SeqTrainSampler(SeqTrainUniformSampler):
    r"""
    A functional datapipe for yielding (user, positives, targets).

    Parameters:
    -----------
    source_dp: dp.iter.IterableWrapper 
        A datapipe that yields users.
    dataset: RecDataSet 
        The dataset object that contains field objects.
    leave_one_out: bool, default to `True`
        `True`: take the last one as a target
        `False`: take `posItems[1:]` as targets
    num_negatives: int
        The number of negatives.
    pool: str
        `unseen`: sampling from unseen items;
        `non-target`: sampling from all items except the current target;
        `all`: sampling from all items.
    """

    def __init__(
        self, source_dp: dp.iter.IterableWrapper, 
        dataset: Optional[RecDataSet] = None, 
        leave_one_out: bool = True, num_negatives: int = 1,
        pool: str = "unseen"
    ) -> None:
        super().__init__(source_dp, dataset, leave_one_out, num_negatives)
        assert pool in ("unseen", "non-target", "all"), "Invalid pool setting ..."

        if pool == 'unseen':
            self._sampling_fn = self._sample_from_unseen
        elif pool == 'non-target':
            self._sampling_fn = self._sample_from_non_target
        else:
            self._sampling_fn = self._sample_from_all

    def _sample_each(self, target):
        return negsamp_vectorized_bsearch(
            [target], self.Item.count, self.num_negatives
        )

    def _sample_from_unseen(self, user: int, positives: Tuple) -> List[int]:
        seen = self.posItems[user]
        return negsamp_vectorized_bsearch(
            seen, self.Item.count, (len(positives), self.num_negatives)
        )

    def _sample_from_non_target(self, user: int, positives: Tuple) -> List[int]:
        return np.stack(self.listmap(
            self._sample_each, positives
        ))

    def _sample_from_all(self, user: int, positives: Tuple) -> List[int]:
        return np.random.randint(
            0, self.Item.count, (len(positives), self.num_negatives)
        )

    def __iter__(self):
        for user, seq in self.source:
            if self._check(seq):
                negatives = self._sampling_fn(user, seq)
                yield [user, seq, negatives]