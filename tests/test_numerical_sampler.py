import tensorflow as tf

from tensorflow_neuroimaging.augmentations import NumericalSampler


def test_numerical_sampler_varies():
    """Tests that the NumericalSampler returns a different value every
    time it is called.
    """
    sampler = NumericalSampler(shape=(3,), minval=0, maxval=1, dtype=tf.float32)

    assert not np.array_equal(sampler(), sampler()), \
           ('NumericalSampler should return a different value every time it '
            'is called')
