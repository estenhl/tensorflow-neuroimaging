import tensorflow as tf

from abc import abstractmethod, ABC

class Sampler(ABC):
    """Abstract base class for samplers. Samplers are used to sample
    random values from a distribution. The sample should be different
    every time the tensor is used in a computation.
    """

    def __init__(self, seed: int = 42):
        self.generator = tf.random.Generator.from_seed(seed)

    @abstractmethod
    def __call__(self) -> tf.Tensor:
        """Returns the tensor generating the random values."""

