import tensorflow as tf

from abc import ABC, abstractmethod
from typing import Tuple

from .mri_augmenter import MRIAugmenter

class AffineMRIAugmenter(MRIAugmenter, ABC):
    """Abstract base class for MRI augmenters applying affine
    transformations.
    """

    @abstractmethod
    def compute_transform(self, image: tf.Tensor) -> Tuple[tf.Tensor]:
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
        pass

    @staticmethod
    def _validate_matrix(matrix: tf.Tensor):
        tf.debugging.assert_shapes([(matrix, (3, 3))],
                                   ('AffineMRIAugmenter only supports 3x3 '
                                    'transformation matrices'))
        tf.debugging.assert_type(matrix, tf.float32,
                                 ('AffineMRIAugmenter only supports float32 '
                                  'transformation matrices'))

    @staticmethod
    def _validate_offset(offset: tf.Tensor):
        tf.debugging.assert_shapes([(offset, (3,))],
                                   ('AffineMRIAugmenter only supports '
                                    '3-dimensional vectors as offsets'))
        tf.debugging.assert_type(offset, tf.float32,
                                 ('AffineMRIAugmenter only supports float32 '
                                  'vectors as offsets'))

    def __init__(self, interpolation: str = 'BILINEAR'):
        """Initialize the affine augmenter.

        Parameters
        ----------
        interpolation : str
            The interpolation method to use. Can be either 'BILINEAR'
            or 'NEAREST_NEIGHBOR'.
        """
        assert (interpolation in ['BILINEAR', 'NEAREST_NEIGHBOR'],
                ('AffineMRIAugmenter only supports BILINEAR and '
                'NEAREST_NEIGHBOR interpolation'))

        self.interpolation = interpolation

    def apply_transformation(self, image: tf.Tensor, matrix: tf.Tensor,
                             offset: tf.Tensor) -> tf.Tensor:
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
        # in their original order
        frozen = generator.uniform((), minval=0, maxval=3, dtype=tf.int32)
        dims = tf.range(2)
        dims = tf.where(dims >= frozen, dims + 1, dims)
        dims = tf.concat([dims, tf.expand_dims(frozen, 0)], axis=0)

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

        return self._apply_transformation(image, transform, offset)

