import tensorflow as tf

from typing import Tuple

from .affine_mri_augmenter import AffineMRIAugmenter


class Flip(AffineMRIAugmenter):
    """MRI augmenter flipping the image along the given axes."""

    @staticmethod
    def _validate_params(flips: tf.Tensor):
        tf.debugging.assert_shapes([(flips, (3,))],
                                   ('Flip should receive a 3-dimensional '
                                    'flips-vector'))
        tf.debugging.assert_type(flips, tf.bool,
                                 'Flip should receive a boolean flips-vector')

    @staticmethod
    def _compute_transform(image: tf.Tensor, \
                           flips: tf.Tensor) -> Tuple[tf.Tensor]:
        image_shape = tf.cast(tf.shape(image), tf.float32)
        image_center = image_shape / tf.constant(2.0)

        # Transform the boolean vector into a vector of -1s for the axes
        # to flip, and 1 otherwise
        flips = tf.cast(flips, tf.float32)
        flips = tf.constant(1.0) - flips
        flips -= tf.constant(0.5)
        flips *= tf.constant(2.0)

        matrix = tf.eye(3) * flips
        offset = image_center - (image_center * flips)

        return matrix, offset

    def __init__(self, flips: tf.Tensor):
        self.flips = flips
