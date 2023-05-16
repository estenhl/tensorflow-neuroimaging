import tensorflow as tf

from tensorflow.python.framework.dtypes import DType
from typing import Any

from .sampler import Sampler


class NumericalSampler(Sampler):
    def __init__(self, *, shape: Tuple[int] = (3,), minval: Any, maxval: Any,
                 dtype: DType, seed: int = 42):
        self.shape = shape
        self.minval = minval
        self.maxval = maxval
        self.dtype = dtype

        super().__init__(seed=seed)

    def __call__(self) -> tf.Tensor:
        return self.generator.uniform(shape=self.shape,
                                      minval=self.minval,
                                      maxval=self.maxval,
                                      dtype=self.dtype)
