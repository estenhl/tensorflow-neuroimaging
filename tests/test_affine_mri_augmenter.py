from typing import Tuple
import numpy as np
import tensorflow as tf

from tensorflow_neuroimaging.augmentations import AffineMRIAugmenter


class TestMRIAugmenter(AffineMRIAugmenter):
    __test__ = False

    def compute_transform(self, *args, **kwargs) -> Tuple:
        raise NotImplementedError()

def test_affine_mri_augmenter_invalid_interpolation():
    """Test that the affine MRI augmenter raises an error if an
    invalid interpolation method is given.
    """
    try:
        TestMRIAugmenter(interpolation='INVALID')
        assert False, ('AffineMRIAugmenter should raise an error if an '
                       'invalid interpolation method is given')
    except AssertionError:
        pass

def test_affine_mri_augmenter_applies_transformation():
    """Test that the affine MRI augmenter applies the given
    transformation.
    """
    image = tf.random.uniform((3, 3, 3))
    augmenter = TestMRIAugmenter()

    matrix = tf.constant([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ], dtype=tf.float32)
    offset = tf.constant([0.0, 0.0, 0.0])
    augmented = augmenter.apply_transformation(image, matrix, offset,
                                               dims=tf.constant([1, 2, 0]))


    image = image.numpy()
    augmented = augmented.numpy()

    assert np.array_equal(np.swapaxes(image, 1, 2), augmented), \
           ('AffineMRIAugmenter.apply_transformation should apply the '
            'transformation to the image')

def test_affine_mri_augmenter_applies_offset():
    """Test that the affine MRI augmenter applies the given offset."""
    image = tf.random.uniform((3, 3, 3))
    augmenter = TestMRIAugmenter()

    matrix = tf.eye(3, dtype=tf.float32)
    offset = tf.constant([0.0, 1.0, 1.0])

    augmented = augmenter.apply_transformation(image, matrix, offset,
                                               dims=tf.constant([1, 2, 0]))

    image = image.numpy()
    augmented = augmented.numpy()

    assert np.array_equal(image[:,1:,1:], augmented[:,:-1,:-1,]), \
           ('AffineMRIAugmenter.apply_transformation should apply the '
           'offset to the image')

def test_affine_mri_augmenter_validates_matrix_shape():
    """Test that the affine MRI augmenter validates the matrix shape."""
    matrix = tf.random.uniform((2, 3), dtype=tf.float32)
    augmenter = TestMRIAugmenter()

    try:
        augmenter.apply_transformation(tf.zeros((3, 3, 3)), matrix,
                                       tf.zeros((3,)))
        assert False, ('AffineMRIAugmenter.apply_transformation should raise '
                       'an InvalidArgumentError if the transformation matrix '
                       'is not 3x3')
    except ValueError:
        pass

def test_affine_mri_augmenter_validates_matrix_dtype():
    """Test that the affine MRI augmenter validates the matrix dtype."""
    matrix = tf.eye(3, dtype=tf.int32)
    augmenter = TestMRIAugmenter()

    try:
        augmenter.apply_transformation(tf.zeros((3, 3, 3)), matrix,
                                       tf.zeros((3,)))
        assert False, ('AffineMRIAugmenter.apply_transformation should raise '
                       'an InvalidArgumentError if the transformation matrix '
                       'is not float32')
    except TypeError:
        pass

def test_affine_mri_augmenter_validates_offset_shape():
    """Test that the affine MRI augmenter validates the offset shape."""
    offset = tf.random.uniform((2,), dtype=tf.float32)
    augmenter = TestMRIAugmenter()

    try:
        augmenter.apply_transformation(tf.zeros((3, 3, 3)), tf.eye(3),
                                       offset)
        assert False, ('AffineMRIAugmenter.apply_transformation should raise '
                       'an InvalidArgumentError if the offset is not 3D')
    except ValueError:
        pass

def test_affine_mri_augmenter_validates_offset_dtype():
    """Test that the affine MRI augmenter validates the offset dtype."""
    offset = tf.zeros((3,), dtype=tf.int32)
    augmenter = TestMRIAugmenter()

    try:
        augmenter.apply_transformation(tf.zeros((3, 3, 3)), tf.eye(3),
                                       offset)
        assert False, ('AffineMRIAugmenter.apply_transformation should raise '
                       'an InvalidArgumentError if the offset is not float32')
    except TypeError:
        pass
