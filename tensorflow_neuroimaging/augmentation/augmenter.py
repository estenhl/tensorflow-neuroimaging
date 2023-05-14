"""A stochastic MRI augmenter built using pure tensorflow. Implements a
range of transformations that can be applied to nifti images. The
transformations are configured through the constructor, and are then
applied stochastically to the images at runtime.

Example usage:
python augmenter.py --image <path/to/image> \
                    --flip_probabilities 0.5 0. 0. \
                    --shift_range 10 \
                    --zoom_range 0.05 \
                    --rotation_range 0.1 \
                    --shear_range 0.02 \
                    --occlusion_box_range 50 \
                    --repetitions 5
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from tensorflow.python.framework.dtypes import DType
from typing import Any, List


class StochasticMRIAugmenter:
    """Class for augmenting nifti images."""

    @staticmethod
    def _resolve_probabilities(probabilities: tf.Tensor, positive: float = 1.0,
                               negative: float = 0.0) -> tf.Tensor:
        generator = tf.random.get_global_generator()
        samples = generator.uniform((3,), minval=0, maxval=1, dtype=tf.float32)

        return tf.where(samples < probabilities, positive, negative)

    @staticmethod
    def _resolve_range(limit: List[Any], offset: float = 0.) -> tf.Tensor:
        generator = tf.random.get_global_generator()

        return generator.uniform(shape=(3,), minval=offset - limit,
                                 maxval=offset + limit, dtype=tf.float32)

    @staticmethod
    def _as_tensor(value: Any, dtype: DType = tf.float32) -> tf.Tensor:
        return None if value is None else \
               tf.convert_to_tensor(value, dtype=dtype)

    def __init__(self, image_shape: np.ndarray = None,
                 interpolation: str = 'BILINEAR',
                 flip_probabilities: List[float] = None,
                 shift_range: int = None,
                 zoom_range: float = None,
                 rotation_range: int = None,
                 shear_range: int = None,
                 occlusion_box_range: int = None):
        self.image_shape = self._as_tensor(image_shape, dtype=tf.int32)
        self.interpolation = interpolation

        #self.occlusion_box_size_range = None if occlusion_box_size_range \
        #                                is None else \
        #                                tf.convert_to_tensor(occlusion_box_size_range,
        #                                                     dtype=tf.int32)

        self.transformations = [
            (self.stochastic_flip, self._as_tensor(flip_probabilities)),
            (self.stochastic_shift, self._as_tensor(shift_range)),
            (self.stochastic_zoom, self._as_tensor(zoom_range)),
            (self.stochastic_rotate, self._as_tensor(rotation_range)),
            (self.stochastic_shear, self._as_tensor(shear_range))
        ]

        self.occlusion_box_range = self._as_tensor(occlusion_box_range, tf.int32)

    @staticmethod
    def stochastic_flip(probabilities: tf.Tensor, image_shape: tf.Tensor) -> tf.Tensor:
        image_shape = tf.cast(image_shape, dtype=tf.float32)
        image_center = image_shape / 2.0
        flips = MRIAugmenter._resolve_probabilities(probabilities,
                                                      positive=-1.0,
                                                      negative=1.0)
        matrix = tf.eye(3) * flips
        offset = image_center - (image_center * flips)

        return matrix, offset

    @staticmethod
    def stochastic_shift(limit: tf.Tensor, _) -> tf.Tensor:
        return tf.eye(3), MRIAugmenter._resolve_range(limit)

    @staticmethod
    def stochastic_zoom(limit: tf.Tensor, image_shape: tf.Tensor) -> tf.Tensor:
        image_shape = tf.cast(image_shape, dtype=tf.float32)
        ratios = MRIAugmenter._resolve_range(limit, offset=1)
        matrix = tf.eye(3) * ratios
        offset = (image_shape * (1.0 - ratios)) / 2.0

        return matrix, offset

    @staticmethod
    def _generate_rotation_matrix(angles: tf.Tensor) -> tf.Tensor:
        y, x, z = tf.unstack(angles, num=3, axis=0)

        # Generates rotation matrices along each axis
        rotation_y = tf.reshape(tf.stack([
            1, 0, 0,
            0, tf.cos(y), -tf.sin(y),
            0, tf.sin(y), tf.cos(y)
        ]), (3, 3))

        rotation_x = tf.reshape(tf.stack([
            tf.cos(x), 0, tf.sin(x),
            0, 1, 0,
            -tf.sin(x), 0, tf.cos(x)
        ]), (3, 3))

        rotation_z = tf.reshape(tf.stack([
            tf.cos(z), -tf.sin(z), 0,
            tf.sin(z), tf.cos(z), 0,
            0, 0, 1
        ]), (3, 3))

        rotation_xz = tf.tensordot(rotation_x, rotation_z, axes=1)
        rotation_yxz = tf.tensordot(rotation_y, rotation_xz, axes=1)

        return rotation_yxz

    @staticmethod
    def stochastic_rotate(limit: tf.Tensor, image_shape: tf.Tensor) -> tf.Tensor:
        angles = MRIAugmenter._resolve_range(limit)
        matrix = MRIAugmenter._generate_rotation_matrix(angles)

        image_shape = tf.cast(image_shape, dtype=tf.float32)
        image_center = image_shape / 2.0
        offset = image_center - tf.tensordot(image_center,
                                             tf.transpose(matrix),
                                             axes=1)

        return matrix, offset

    @staticmethod
    def stochastic_shear(limit: tf.Tensor, _) -> tf.Tensor:
        ratios = MRIAugmenter._resolve_range(limit)
        y, x, z = tf.unstack(ratios, num=3, axis=0)

        matrix = tf.reshape(tf.stack([
            1, y, y, x, 1, x, z, z, 1
        ]), (3, 3))

        return matrix, tf.zeros(3)

    @staticmethod
    def _stochastic_occlude(image: tf.Tensor, limit: tf.Tensor,
                            shape: tf.Tensor) -> tf.Tensor:
        generator = tf.random.get_global_generator()
        height, width, depth = tf.unstack(shape, num=3, axis=0)

        box_height = generator.uniform((), minval=1, maxval=limit,
                                       dtype=tf.int32)
        box_width = generator.uniform((), minval=1, maxval=limit,
                                      dtype=tf.int32)
        box_depth = generator.uniform((), minval=1, maxval=limit,
                                      dtype=tf.int32)

        ymin = generator.uniform((), minval=0, maxval=height - limit,
                                 dtype=tf.int32)
        ymax = ymin + box_height
        xmin = generator.uniform((), minval=0, maxval=width - limit,
                                 dtype=tf.int32)
        xmax = xmin + box_width
        zmin = generator.uniform((), minval=0, maxval=depth - limit,
                                 dtype=tf.int32)
        zmax = zmin + box_depth

        crop = image[ymin:ymax + 1, xmin:xmax + 1, zmin:zmax + 1]
        mean = tf.reduce_mean(crop)
        stddev = tf.math.reduce_std(crop)

        # Generate random values for the box and reshape into a vector
        values = generator.normal(tf.shape(crop), mean=mean, stddev=stddev)
        values = tf.reshape(values, [-1])

        # Generate all indices that should be filled
        idx = tf.meshgrid(
            tf.range(ymin, ymax + 1),
            tf.range(xmin, xmax + 1),
            tf.range(zmin, zmax + 1),
            indexing='ij'
        )
        idx = tf.reshape(tf.stack(idx, axis=-1), [-1, 3])

        image = tf.tensor_scatter_nd_update(image, idx, values)

        return image

    def __call__(self, image: tf.Tensor) ->  tf.Tensor:
        image_shape = self.image_shape if self.image_shape is not None \
                      else tf.shape(image)
        generator = tf.random.get_global_generator()

        matrix = tf.eye(3)
        offset = tf.zeros(3)

        for f, params in self.transformations:
            if params is not None:
                m, o = f(params, image_shape)
                matrix = tf.tensordot(matrix, m, axes=1)
                offset = offset + o

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

        output_shape = tf.gather(image_shape, dims)
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

        if self.occlusion_box_range is not None:
            image = self._stochastic_occlude(image, self.occlusion_box_range,
                                             shape=image_shape)

        return image

        def transform(image, matrix, offset):
            return affine_transform(image, matrix, offset=offset, order=1)

        return tf.py_function(transform, [image, matrix, offset], tf.float32)

        """
        translation_matrix = np.eye(3)
        offset = np.zeros(3)

        translations = [
            (self.flip, self.flip_probabilities, self._resolve_probabilities),
            (self.shift, self.shift_ranges, self._resolve_ranges),
            (self.zoom, self.zoom_ranges,
             lambda x: self._resolve_ranges(x, offset=1)),
            (self.rotate, self.rotation_ranges, self._resolve_ranges),
            (self.shear, self.shear_ranges, self._resolve_ranges)
        ]

        for translator, params, resolver in translations:
            if params is not None:
                m, o = translator(image, resolver(params))
                translation_matrix = translation_matrix.dot(m)
                offset += o

        image = affine_transform(image, translation_matrix, offset=offset,
                                 order=2)

        if self.crop_box_sides is not None:
            height = np.random.randint(1, self.crop_box_sides)
            width = np.random.randint(1, self.crop_box_sides)
            depth = np.random.randint(1, self.crop_box_sides)

            ymin = np.random.randint(0, image.shape[0] - (height + 1))
            xmin = np.random.randint(0, image.shape[1] - (width + 1))
            zmin = np.random.randint(0, image.shape[2] - (depth + 1))

            patch = image[ymin:ymin + height,
                          xmin:xmin + width,
                          zmin:zmin + depth]

            image[ymin:ymin + height,
                  xmin:xmin + width,
                  zmin:zmin + depth] = np.random.normal(np.mean(patch),
                                                        np.std(patch),
                                                        patch.shape)

        return image
        """

if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    import nibabel as nib

    from time import time

    parser = argparse.ArgumentParser('Augment a given nifti image')

    parser.add_argument('-i', '--image', type=str,
                        help='Path to the image to augment')
    parser.add_argument('-n', '--repetitions', required=True, type=int,
                        help='Number of times to augment the image')
    parser.add_argument('-f', '--flip_probabilities', nargs=3, type=float,
                        required=False, default=None,
                        help=('Probabilities of flipping the image along each '
                              'axis'))
    parser.add_argument('-si', '--shift_range', type=float, required=False,
                        default=None,
                        help=('Range of values to shift the image along each '
                                'axis'))
    parser.add_argument('-z', '--zoom_range', type=float, required=False,
                        default=None,
                        help=('Range of values to zoom the image along each '
                              'axis'))
    parser.add_argument('-r', '--rotation_range', type=float, required=False,
                        default=None,
                        help=('Range of values to rotate the image along each '
                              'axis'))
    parser.add_argument('-se', '--shear_range', type=float, required=False,
                        default=None,
                        help=('Range of values to shear the image along each '
                              'axis'))
    parser.add_argument('-o', '--occlusion_box_range', type=int,
                        required=False, default=None,
                        help=('Max side length of occlusion box introduced '
                                'into the image'))

    args = parser.parse_args()

    image = nib.load(args.image)
    image = image.get_fdata()
    image = tf.constant(image, dtype=tf.float32)
    height, width, depth = image.shape[:3]

    augmenter = MRIAugmenter(
        flip_probabilities=args.flip_probabilities,
        shift_range=args.shift_range,
        zoom_range=args.zoom_range,
        rotation_range=args.rotation_range,
        shear_range=args.shear_range,
        occlusion_box_range=args.occlusion_box_range
    )

    start = time()
    augmentations = [augmenter(image) for _ in range(args.repetitions)]
    print(f'Performed {args.repetitions} augmentations in {time() - start}s')

    fig, ax = plt.subplots(3, args.repetitions + 1, figsize=(15, 10))
    clim = (np.amin(image), np.amax(image))

    ax[0][0].imshow(image[height // 2, :, :], cmap='Greys_r', clim=clim)
    ax[0][0].axis('off')
    ax[0][0].set_title('Original')
    ax[1][0].imshow(image[:, width // 2, :], cmap='Greys_r', clim=clim)
    ax[1][0].axis('off')
    ax[2][0].imshow(image[:, :, depth // 2], cmap='Greys_r', clim=clim)
    ax[2][0].axis('off')

    for i, augmented in enumerate(augmentations):
        ax[0][i + 1].imshow(augmented[height // 2, :, :], cmap='Greys_r',
                            clim=clim)
        ax[0][i + 1].axis('off')
        ax[1][i + 1].imshow(augmented[:, width // 2, :], cmap='Greys_r',
                            clim=clim)
        ax[1][i + 1].axis('off')
        ax[2][i + 1].imshow(augmented[:, :, depth // 2], cmap='Greys_r',
                            clim=clim)
        ax[2][i + 1].axis('off')

    plt.show()
