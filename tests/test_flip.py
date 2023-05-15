""" Tests for MRIAugmenter class. """
import numpy as np
import tensorflow as tf

from tensorflow.python.framework.errors_impl import InvalidArgumentError

from tensorflow_neuroimaging.augmentations import Flip


def test_mri_augmenter_flip_none():
    """Tests that the MRIAugmenter.flip method does not flip the image
    when no axes to flip are given.
    """
    image = tf.random.uniform((10, 10, 10))
    flips = tf.repeat(False, 3)
    matrix, offset = Flip.compute_transform(image, flips=flips)

    assert np.array_equal(matrix, np.eye(3)), \
           ('MRIAugmenter.flip should not flip the image when no axes to flip '
            'are given')
    assert np.array_equal(offset, np.zeros(3)), \
           ('MRIAugmenter.flip should not add an offset when no flips are '
            'given')

def test_mri_augmenter_flip_2d():
    """Tests that the MRIAugmenter.flip method raises an error if the
    image is not 3-dimensional.
    """
    image = tf.random.uniform((10, 10), dtype=tf.float32)
    flips = tf.repeat(False, 3)

    try:
        Flip.compute_transform(image, flips=flips)

        assert False, \
               ('MRIAugmenter.flip should raise an error when the image is '
                'not 3-dimensional')
    except InvalidArgumentError:
        pass

def test_mri_augmenter_flip_2d_vector():
    """Tests that the MRIAugmenter.flip method raises an error if the
    flips argument is not 3-dimensional.
    """
    image = tf.random.uniform((10, 10, 10), dtype=tf.float32)
    flips = tf.repeat(False, 2)
    try:
        Flip.compute_transform(image, flips=flips)

        assert False, \
               ('MRIAugmenter.flip should raise an error when the flips '
                'argument is not 3-dimensional')
    except ValueError:
        pass

def test_mri_augmenter_flip_integer_flips():
    """Tests that the MRIAugmenter.flip method raises an error if the
    flips argument is not boolean.
    """
    image = tf.random.uniform((10, 10, 10), dtype=tf.float32)
    flips = tf.constant([0, 0, 0])
    try:
        Flip.compute_transform(image, flips=flips)

        assert False, \
                ('MRIAugmenter.flip should raise an error when the flips '
                 'argument is not boolean')
    except TypeError:
        pass

def test_mri_augmenter_flip_y():
    """Tests that the MRIAugmenter.flip method flips the image across
    the y-axis when given the vector [True, False, False].
    """
    image = tf.random.uniform((10, 11, 12), dtype=tf.float32)
    flips = tf.constant([True, False, False])
    matrix, offset = Flip.compute_transform(image, flips=flips)

    assert np.array_equal(matrix, np.diag([-1, 1, 1])), \
           ('MRIAugmenter.flip should flip the image across the y-axis when '
            'given the vector [True, False, False]')
    assert np.array_equal(offset, np.array([10, 0, 0])), \
           ('MRIAugmenter.flip should add an offset equal to the image width '
            'when flipping across the y-axis')

def test_mri_augmenter_flip_x():
    """Tests that the MRIAugmenter.flip method flips the image across
    the x-axis when given the vector [False, True, False].
    """
    image = tf.random.uniform((10, 11, 12), dtype=tf.float32)
    flips = tf.constant([False, True, False])
    matrix, offset = Flip.compute_transform(image, flips=flips)

    assert np.array_equal(matrix, np.diag([1, -1, 1])), \
           ('MRIAugmenter.flip should flip the image across the x-axis when '
            'given the vector [False, True, False]')
    assert np.array_equal(offset, np.array([0, 11, 0])), \
           ('MRIAugmenter.flip should add an offset equal to the image height '
            'when flipping across the x-axis')

def test_mri_augmenter_flip_z():
    """Tests that the MRIAugmenter.flip method flips the image across
    the z-axis when given the vector [False, False, True].
    """
    image = tf.random.uniform((10, 11, 12), dtype=tf.float32)
    flips = tf.constant([False, False, True])
    matrix, offset = Flip.compute_transform(image, flips=flips)

    assert np.array_equal(matrix, np.diag([1, 1, -1])), \
           ('MRIAugmenter.flip should flip the image across the z-axis when '
            'given the vector [False, False, True]')
    assert np.array_equal(offset, np.array([0, 0, 12])), \
           ('MRIAugmenter.flip should add an offset equal to the image depth '
            'when flipping across the z-axis')
