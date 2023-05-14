import tensorflow as tf

from abc import ABC
from typing import Tuple

class MRIAugmenter(ABC):
    """Class for augmenting 3-dimensional structural MRIs. All methods
    that correlate to a transformation does not modify the image,
    but instead returns a transformation matrix and offset that can be
    used to transform the image. All transformations have the same
    interface although some of the arguments are redundant for some
    operations.
    """

    @staticmethod
    def _assert_3d(image: tf.Tensor):
        tf.Assert(tf.rank(image) == 3,
                  'MRIAugmenter only supports 3-dimensional images')

    @staticmethod
    def _resolve_image_shape(image: tf.Tensor,
                             image_shape: tf.Tensor) -> tf.Tensor:
        image_shape = image_shape if image_shape is not None \
                      else tf.shape(image)

        return image_shape

    @staticmethod
    def _validate_flip_parameters(image: tf.Tensor, flips: tf.Tensor):
        MRIAugmenter._assert_3d(image)
        tf.Assert(tf.shape(flips) == 3,
                  ('MRIAugmenter.flip flips argument should be a vector of '
                   'length 3'))
        tf.debugging.assert_type(flips, tf.bool,
                                 ('MRIAugmenter.flip flips argument should be '
                                  'boolean'))

    @staticmethod
    def flip(image: tf.Tensor, *, flips: tf.Tensor,
             image_shape: tf.Tensor = None) -> Tuple[tf.Tensor]:
        """Flip the image across each axis independently. Which axes
        are flipped is determined by the flips argument.

        Parameters
        ----------
        image : tf.Tensor
            A 3-dimensional tensor representing a structural MRI.
        flips : tf.Tensor
            A boolean vector indicating which axes to flip.
        image_shape : tf.Tensor
            Optional image shape. If not provided, the shape will be
            inferred.

        Returns:
            A tuple where the first element is a 3x3 transformation
            matrix and the second element is a 3-dimensional vector
            representing the offset.
        """
        MRIAugmenter._validate_flip_parameters(image, flips)
        image_shape = MRIAugmenter._resolve_image_shape(image, image_shape)
        image_shape = tf.cast(image_shape, tf.float32)
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

    @staticmethod
    def shift(image: tf.Tensor, offset: tf.Tensor) -> Tuple[tf.Tensor]:
