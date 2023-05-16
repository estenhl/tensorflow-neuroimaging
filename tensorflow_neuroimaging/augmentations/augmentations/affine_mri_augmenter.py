"""MRI augmenter applying affine transformations."""
import tensorflow as tf

from abc import ABC, abstractmethod
from typing import Tuple

from .mri_augmenter import MRIAugmenter

class AffineMRIAugmenter(MRIAugmenter, ABC):
    """Abstract base class for MRI augmenters applying affine
    transformations.
    """

    @staticmethod
    @abstractmethod
    def _validate_params(*args, **kwargs):
        """Used internally to validate the parameters passed to
        compute_transform.

        Parameters
        ----------
        **kwargs
            The parameters passed to compute_transform

        Raises
        ------
        AssertionError
            If the parameters are invalid.
        """

    @abstractmethod
    def _compute_transform(self, image: tf.Tensor, **kwargs) -> Tuple[tf.Tensor]:
        """Internal method to compute the affine transformation
        parameters which are overwritten by subclasses.
        """

    @staticmethod
    def _validate_matrix(matrix: tf.Tensor):
        """Used internally to validate the transformation matrix passed
        to apply_transformation.

        Parameters
        ----------
        matrix : tf.Tensor
            A transformation matrix.

        Raises
        ------
        ValueError
            If the matrix is not 3x3.
        TypeError
            If the matrix is not of type tf.float32.
        """
        tf.debugging.assert_shapes([(matrix, (3, 3))],
                                   ('AffineMRIAugmenter only supports 3x3 '
                                    'transformation matrices'))
        tf.debugging.assert_type(matrix, tf.float32,
                                 ('AffineMRIAugmenter only supports float32 '
                                  'transformation matrices'))

    @staticmethod
    def _validate_offset(offset: tf.Tensor):
        """Used internally to validate the offset passed to
        apply_transformation.

        Parameters
        ----------
        offset : tf.Tensor
            A vector representing the offset.

        Raises
        ------
        ValueError
            If the offset is not 3-dimensional.
        TypeError
            If the offset is not of type tf.float32.
        """
        tf.debugging.assert_shapes([(offset, (3,))],
                                   ('AffineMRIAugmenter only supports '
                                    '3-dimensional vectors as offsets'))
        tf.debugging.assert_type(offset, tf.float32,
                                 ('AffineMRIAugmenter only supports float32 '
                                  'vectors as offsets'))

    @classmethod
    def compute_transform(cls, image: tf.Tensor, **kwargs) -> Tuple[tf.Tensor]:
        """Compute the affine transformation parameters.

        Parameters
        ----------
        image : tf.Tensor
            A 3-dimensional tensor representing a structural MRI.

        Returns
        -------
        transform : tf.Tensor
            A 3x3 transformation matrix.
        offset : tf.Tensor
            A 3-dimensional vector representing the offset.
        """
        cls.validate_image(image)
        cls._validate_params(**kwargs)

        return cls._compute_transform(image, **kwargs)

    def __init__(self, interpolation: str = 'BILINEAR'):
        """Initialize the affine augmenter.

        Parameters
        ----------
        interpolation : str
            The interpolation method to use. Can be either 'BILINEAR'
            or 'NEAREST_NEIGHBOR'.
        """
        assert interpolation in ['BILINEAR', 'NEAREST_NEIGHBOR'], \
                ('AffineMRIAugmenter only supports BILINEAR and '
                'NEAREST_NEIGHBOR interpolation')

        self.interpolation = interpolation

    def apply_transformation(self, image: tf.Tensor, matrix: tf.Tensor,
                             offset: tf.Tensor,
                             dims: tf.Tensor = None) -> tf.Tensor:
        """Applies a given affine transformation to a structural MRI.
        NOTE: Due to no support for 3-dimensional affine transforms
        in TensorFlow, this method is implemented using a projective
        transform in two randomly selected dimensions. The remaining is
        kept fixed.

        Parameters
        ----------
        image : tf.Tensor
            A 3-dimensional tensor representing a structural MRI.
        matrix : tf.Tensor
            A 3x3 transformation matrix.
        offset : tf.Tensor
            A 3-dimensional vector representing the offset.

        Returns
        -------
        image : tf.Tensor
            The transformed MRI.
        """
        self._validate_matrix(matrix)
        self._validate_offset(offset)

        generator = tf.random.get_global_generator()

        # Overly complicated way to create a list of dimensions such
        # that the two dimensions which should be modified are first
        # in their original order. Tensorflow is not able to generate
        # a random integer, so we have to generate a random float and
        # cast it.
        # See https://github.com/tensorflow/tensorflow/issues/60597
        if dims is None:
            frozen = generator.uniform(shape=(), minval=0, maxval=2.5,
                                    dtype=tf.float32)
            frozen = tf.cast(frozen, tf.int32)
            dims = tf.range(2)
            dims = tf.where(dims >= frozen, dims + 1, dims)
            dims = tf.concat(values=[dims, tf.expand_dims(frozen, 0)], axis=0)

        # Reorder the dimensions of the image such that the two
        # dimensions which should be modified are first
        image = tf.transpose(image, dims)

        # Create the projection vector
        # NOTE: The projection vector has the transforms for the
        # X-dimensions first, e.g. the second row of the matrix
        projection = tf.stack([
            matrix[dims[1]][dims[1]], matrix[dims[1]][dims[0]], offset[dims[1]],
            matrix[dims[0]][dims[1]], matrix[dims[0]][dims[0]], offset[dims[0]],
            0, 0
        ])

        # Calculate output shape of the transposed image
        output_shape = tf.gather(tf.shape(image), dims)
        output_shape = tf.slice(output_shape, [0], [2])

        # Applies the transformation to the two first dimensions
        image = tf.raw_ops.ImageProjectiveTransformV3(
            images=tf.expand_dims(image, 0),
            transforms=tf.expand_dims(projection, 0),
            output_shape=output_shape,
            fill_value=0.,
            interpolation=self.interpolation
        )[0]

        # Reorder the image dimensions back to their original order
        image = tf.transpose(image, tf.argsort(dims))

        return image

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
        super().validate_image(image)

        transform, offset = self.compute_transform(image)

        return self.apply_transformation(image, transform, offset)

