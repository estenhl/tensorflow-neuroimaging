import tensorflow as tf

from abc import ABC, abstractmethod

class MRIAugmenter(ABC):
    """Abstract base class for MRI Augmenters."""

    @abstractmethod
    def __call__(self, image: tf.Tensor) -> tf.Tensor:
        """Apply augmentations to image.

        Parameters
        ----------
        image : tf.Tensor
            A 3-dimensional tensor representing a structural MRI.

        Returns
        -------
        image : tf.Tensor
            The augmented image.
        """

    @staticmethod
    def validate_image(image: tf.Tensor):
        """Validate the image passed to the augmenter.

        Parameters
        ----------
        image : tf.Tensor
            A 3-dimensional tensor representing a structural MRI.

        Raises
        ------
        tf.errors.InvalidargumentError
            If the image is not 3-dimensional.
        TypeError
            If the image is not of type tf.float32."""
        tf.Assert(tf.rank(image) == 3,
                  'MRIAugmenter only supports 3-dimensional images')
        tf.debugging.assert_type(image, tf.float32,
                                 'MRIAugmenter only supports float32 images')
