import tensorflow as tf

from typing import Dict

_NIFTI_DTYPES = {
    4: {'dtype': tf.int16, 'bytes': 2},
    16: {'dtype': tf.float32, 'bytes': 4}
}
_VALID_NIFTI_DTYPES = list(_NIFTI_DTYPES.keys())
_NIFTI_BYTE_SIZES = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=_VALID_NIFTI_DTYPES,
        values=[_NIFTI_DTYPES[key]['bytes'] for key in _VALID_NIFTI_DTYPES]
    ),
    default_value=tf.constant(-1, dtype=tf.int32),
    name='nifti_dtypes'
)
def _parse_header(tfbytes: tf.Tensor) -> Dict[str, tf.Tensor]:
    headersize = tf.strings.substr(tfbytes, 0, 4)
    headersize = tf.io.decode_raw(headersize, tf.int32)

    dims = [tf.strings.substr(tfbytes, 40 + i, 2) \
            for i in range(0, 16, 2)]
    dims = [tf.io.decode_raw(dim, tf.uint16) \
            for dim in dims]

    dtype = tf.strings.substr(tfbytes, 70, 2)
    dtype = tf.io.decode_raw(dtype, tf.int16)

    return {
        'headersize': headersize,
        'dims': dims,
        'dtype': dtype
    }

def _decode(imagebytes: tf.Tensor, dtype: tf.Tensor):
    cases = [
        (tf.equal(dtype, key),
         lambda: tf.io.decode_raw(imagebytes,
                                  _NIFTI_DTYPES[key]['dtype'])) \
        for key in _VALID_NIFTI_DTYPES
    ]
    return tf.case(cases,
                   default=lambda: tf.Assert(False,
                                             [f'Unknown data type: {dtype}']))

def load_nifti(path: tf.Tensor) -> tf.Tensor:
    raise NotImplementedError('load_nifti is not implemented yet')
    niftidata = tf.io.read_file(path)
    niftidata = tf.cond(tf.strings.regex_full_match(path, '.*\.gz$'),
                        lambda: tf.io.decode_compressed(niftidata, 'GZIP'),
                        lambda: niftidata)

    headerbytes = tf.strings.substr(niftidata, 0, 348)
    extensionbytes = tf.strings.substr(niftidata, 348, 4)
    imagebytes = tf.strings.substr(niftidata, tf.constant([352]),
                                   num_image_bytes)

    header = _parse_header(headerbytes)
    spatial_dims = tf.cast(tf.concat(header['dims'][1:4], axis=0),
                           tf.int32)
    num_voxels = tf.reduce_prod(spatial_dims)
    bytes_per_voxel = _NIFTI_BYTE_SIZES.lookup(tf.cast(header['dtype'],
                                                       tf.int32))
    tf.Assert(tf.not_equal(bytes_per_voxel, -1),
              [f'Unknown data type: {header["dtype"]}'])
    num_image_bytes = num_voxels * bytes_per_voxel
    imagebytes = _decode(imagebytes, header['dtype'])
    image = tf.reshape(imagebytes, tf.reverse(spatial_dims, axis=[0]))

    return image


if __name__ == '__main__':
    import argparse
    import numpy as np
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser('Loads a nifti image')

    parser.add_argument('-p', '--path', type=str,
                        help='Path to the nifti image')

    args = parser.parse_args()

    path = tf.constant(args.path)
    image = load_nifti(path=tf.constant(args.path))
    image = image.numpy()
    center = np.asarray(image.shape) // 2

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))

    ax[0].imshow(image[center[0], :, :], cmap='Greys_r')
    ax[0].axis('off')
    ax[1].imshow(image[:, center[1], :], cmap='Greys_r')
    ax[1].axis('off')
    ax[2].imshow(image[:, :, center[2]], cmap='Greys_r')
    ax[2].axis('off')

    plt.show()

