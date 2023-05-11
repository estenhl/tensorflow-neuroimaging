import tensorflow as tf

from typing import Dict


def _parse_header(tfbytes: tf.Tensor) -> Dict[str, tf.Tensor]:
    headersize = tf.strings.substr(tfbytes, 0, 4)
    dims = [tf.strings.substr(tfbytes, 40 + i, 2) \
            for i in range(0, 16, 2)]
    dims = [tf.cast(tf.io.decode_raw(dim, tf.uint16), tf.int32) \
            for dim in dims]

    return {'headersize': headersize, 'dims': dims}

def load_nifti(path: tf.Tensor, image_shape: tf.Tensor = None) -> tf.Tensor:
    if image_shape is None:
        raise NotImplementedError()

    niftidata = tf.io.read_file(path)
    niftibytes = tf.io.decode_raw(niftidata, tf.uint8)

    headerbytes = tf.strings.substr(niftidata, 0, 348)
    header = _parse_header(headerbytes)
    voxels = header['dims'][1] * header['dims'][2] * header['dims'][3]
    print(voxels)
    extensionbytes = tf.strings.substr(niftidata, 348, 4)
    imagebytes = tf.strings.substr(niftidata, 352,
                                   voxels)

    imagedata = tf.io.decode_raw(imagebytes, tf.int16)
    imagedata = tf.reshape(imagedata, (324, 256, 256))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Loads a nifti image')

    parser.add_argument('-p', '--path', type=str,
                        help='Path to the nifti image')
    parser.add_argument('-s', '--shape', required=False, nargs=3, type=int,
                        help='Shape of the image')

    args = parser.parse_args()

    path = tf.constant(args.path)
    image = load_nifti(path=tf.constant(args.path),
                       image_shape=tf.constant(args.shape))

