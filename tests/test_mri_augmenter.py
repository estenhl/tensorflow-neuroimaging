import tensorflow as tf

from tensorflow_neuroimaging.augmentations import MRIAugmenter


def test_mri_augmenter_assert_shape():
    """Test that the MRI augmenter validates the image shape."""

    image = tf.random.uniform((3, 3), dtype=tf.float32)

    try:
        MRIAugmenter.validate_image(image)
        assert False, ('MRIAugmenter.validate_image should raise an '
                       'InvalidArgumentError if the image is not 3D')
    except tf.errors.InvalidArgumentError:
        pass

def test_mri_augmenter_assert_dtype():
    """Test that the MRI augmenter validates the image dtype."""

    image = tf.random.uniform((3, 3, 3), maxval=1, dtype=tf.int32)

    try:
        MRIAugmenter.validate_image(image)
        assert False, ('MRIAugmenter.validate_image should raise an '
                       'InvalidArgumentError if the image is not float32')
    except TypeError:
        pass
