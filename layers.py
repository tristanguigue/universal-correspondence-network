import tensorflow as tf
import tensorflow.contrib.layers as layers
from spatial_transformer import transformer
from tools import unravel_index


def conv_spatial_transfo(x, thetas, kernel_size):
    """Run the spatial transformer for each patch of kernel_size * kernel_size.

    Args:
        thetas: the parameters of each spatial transformer
        kernel_size: the size of each patch

    Return:
        The patches glued together after having been
        spatially transformed
    """
    size_x = int(x.get_shape()[1])
    size_y = int(x.get_shape()[2])
    channels = int(x.get_shape()[3])
    # Flatten the parameters
    thetas = tf.reshape(thetas, [-1, size_x * size_y, 6])

    # Extract patches of kernel_size * kernel_size at each
    # pixel of the input image
    x = tf.extract_image_patches(
        x, ksizes=[1, kernel_size, kernel_size, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding='SAME')
    # Flatten the patches
    x = tf.reshape(x, [-1, kernel_size, kernel_size, channels])
    # Run through the spatial transformer
    res = transformer(x, thetas, (kernel_size, kernel_size))
    # Reform the image
    res = tf.reshape(res, [-1, size_x * kernel_size, size_y * kernel_size, channels])
    return res


def inception(x, conv11_size, conv33_reduce_size, conv33_size,
              conv55_reduce_size, conv55_size, pool11_size):
    """Run the inception layer for input x.

    Args:
        x: the input
        conv11_size: number of channels for the 1*1 convolution
        conv33_reduce_size: number of channels for the 3*3 reduce convolution
        conv33_size: number of channels for the 3*3 convolution
        conv55_reduce_size: number of channels for the 5*5 reduce convolution
        conv55_size: number of channels for the 5*5 convolution
        pool11_size: number of channels for the max pooling

    Return:
        The concatenated results of the convolutions
    """
    with tf.variable_scope('conv_11'):
        conv11 = layers.conv2d(x, conv11_size, [1, 1])

    with tf.variable_scope('conv_33'):
        conv33_reduce = layers.conv2d(x, conv33_reduce_size, [1, 1], activation_fn=tf.nn.relu)
        conv33 = layers.conv2d(conv33_reduce, conv33_size, [3, 3], activation_fn=tf.nn.relu)

    with tf.variable_scope('conv_55'):
        conv55_reduce = layers.conv2d(x, conv55_reduce_size, [1, 1], activation_fn=tf.nn.relu)
        conv55 = layers.conv2d(conv55_reduce, conv55_size, [5, 5], activation_fn=tf.nn.relu)

    with tf.variable_scope('pool_proj'):
        pool_proj = layers.max_pool2d(x, [3, 3], stride=1, padding='SAME')
        pool11 = layers.conv2d(pool_proj, pool11_size, [1, 1], activation_fn=tf.nn.relu)

    return tf.concat([conv11, conv33, conv55, pool11], 3)


def project_at_points(feature_proj, points, image_indices):
    """Get the values of the projected features at the given points.

    Args:
        feature_proj: the values to extract
        points: the indices of the points to be extracted
        image_indices: the id of the images

    Return:
        The values of the projected features at the given points
    """
    stacked_indices1 = tf.stack((image_indices, points[:, :, 0], points[:, :, 1]), -1)
    feature_proj_at_points = tf.gather_nd(feature_proj, stacked_indices1)
    return feature_proj_at_points


def knn(feature1_proj_at_corres, feature2):
    """Find nearest neighours of feature1_proj_at_corres in feature2.

    Args:
        feature1_proj_at_corres: the reference vectors
        feature2: the list of neighbours

    Return:
        The indices of the nearest neighbours
    """
    feature_size_x = int(feature2.get_shape()[1])
    feature_size_y = int(feature2.get_shape()[2])
    feature_channels = int(feature2.get_shape()[3])

    flat_feature2 = tf.reshape(feature2, (-1, feature_size_x * feature_size_y, feature_channels))
    distance = tf.negative(
        tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(
            tf.expand_dims(feature1_proj_at_corres, 2), tf.expand_dims(flat_feature2, 1))), reduction_indices=3)))
    flat_indices_nearest = tf.cast(tf.argmin(distance, 2), tf.int32)
    indices_nearest = unravel_index(flat_indices_nearest, (feature_size_x, feature_size_y))

    return indices_nearest, flat_indices_nearest
