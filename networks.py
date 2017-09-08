from layers import conv_spatial_transfo, inception
import tensorflow as tf
import tensorflow.contrib.layers as layers
from layers import project_at_points, inception, knn


class FullyConvNetwork():
    """Fully Convolutional Network including convolutional spatial transformer.

    Attributes:
        feature: The output features of the network
    """

    def __init__(self, x):
        conv1 = layers.conv2d(x, 64, [7, 7], stride=2, scope='conv1', activation_fn=tf.nn.relu)
        pool1 = layers.max_pool2d(conv1, [3, 3], scope='pool1')
        pool1_lrn = tf.nn.local_response_normalization(pool1, name='pool1_lrn', alpha=0.0001,
                                                       beta=0.75)
        conv2_reduce = layers.conv2d(pool1_lrn, 64, [1, 1], scope='conv2_reduce',
                                     activation_fn=tf.nn.relu)
        conv2 = layers.conv2d(conv2_reduce, 192, [3, 3], scope='conv2', activation_fn=tf.nn.relu)
        conv2_lrn = tf.nn.local_response_normalization(conv2, name='pool1_lrn', alpha=0.0001,
                                                       beta=0.75)
        pool2 = layers.max_pool2d(conv2_lrn, [3, 3], scope='pool2')

        with tf.variable_scope('inception_3a'):
            inception_3a = inception(pool2, 64, 96, 128, 16, 32, 32)

        with tf.variable_scope('inception_3b'):
            inception_3b = inception(inception_3a, 128, 128, 192, 32, 96, 64)
            pool3 = layers.max_pool2d(inception_3b, [3, 3], scope='pool3')

        with tf.variable_scope('inception_4a'):
            conv11 = layers.conv2d(pool3, 192, [1, 1], activation_fn=tf.nn.relu)

            conv33_reduce = layers.conv2d(pool3, 96, [1, 1], activation_fn=tf.nn.relu)
            conv33_reduce_param = layers.conv2d(conv33_reduce, 32, [1, 1], activation_fn=tf.nn.relu)
            conv33_reduce_param2 = layers.conv2d(conv33_reduce_param, 32, [1, 1],
                                                 activation_fn=tf.nn.relu)
            conv33_reduce_param3 = layers.conv2d(conv33_reduce_param2, 6, [1, 1])
            conv33_spatial_transfo = conv_spatial_transfo(conv33_reduce, conv33_reduce_param3, 3)
            conv33 = layers.conv2d(conv33_spatial_transfo, 208, [3, 3], stride=3)

            conv55_reduce = layers.conv2d(pool3, 16, [1, 1], activation_fn=tf.nn.relu)
            conv55_reduce_param = layers.conv2d(conv55_reduce, 32, [1, 1], activation_fn=tf.nn.relu)
            conv55_reduce_param2 = layers.conv2d(conv55_reduce_param, 32, [1, 1],
                                                 activation_fn=tf.nn.relu)
            conv55_reduce_param3 = layers.conv2d(conv55_reduce_param2, 6, [1, 1])
            conv55_spatial_transfo = conv_spatial_transfo(conv55_reduce, conv55_reduce_param3, 5)
            conv55 = layers.conv2d(conv55_spatial_transfo, 48, [5, 5], stride=5)

            pool_proj = layers.max_pool2d(pool3, [3, 3], stride=1, padding='SAME')
            pool11 = layers.conv2d(pool_proj, 64, [1, 1])

            inception4a = tf.concat([conv11, conv33, conv55, pool11], 3)

        feature_unnorm = layers.conv2d(inception4a, 128, [1, 1], scope='feature_unnorm')
        self.feature = tf.nn.l2_normalize(feature_unnorm, dim=3)


class UniversalCorrepondenceNetwork():
    """Network including the siamese network and the projection of features on the initial image."""

    def __init__(self, img_shape, nb_corres):
        self.nb_corres = nb_corres
        self.image_size_x = img_shape[0]
        self.image_size_y = img_shape[1]
        self.image_channels = img_shape[2]

        self.x1 = tf.placeholder(
            tf.float32, [None, self.image_size_x, self.image_size_y, self.image_channels], name='x1')
        self.x2 = tf.placeholder(
            tf.float32, [None, self.image_size_x, self.image_size_y, self.image_channels], name='x2')
        self.correspondences = tf.placeholder(tf.float32, [None, self.nb_corres, 4], name='corres')

        batch_size = tf.shape(self.x1)[0]

        with tf.variable_scope('siamese') as scope:
            self.network1 = FullyConvNetwork(self.x1)
            scope.reuse_variables()
            self.network2 = FullyConvNetwork(self.x2)

        with tf.variable_scope('projection') as scope:
            # Project the output feature back into the intput space using bilinear interpolation
            feature1_proj = tf.image.resize_bilinear(
                self.network1.feature, [self.image_size_x, self.image_size_y])
            feature2_proj = tf.image.resize_bilinear(
                self.network2.feature, [self.image_size_x, self.image_size_y])

            # Hack to ensure the correspondences can be interpreted as pixel coordinates
            self.pos_corres = tf.nn.relu(tf.cast(self.correspondences, tf.int32))

            image_indices = tf.reshape(tf.range(batch_size), (batch_size, 1))
            image_indices = tf.tile(image_indices, [1, nb_corres])
            # Values of the features at the correspondence points
            self.projection_corres_1 = project_at_points(
                feature1_proj, self.pos_corres[:, :, :2], image_indices)
            self.projection_corres_2 = project_at_points(
                feature2_proj, self.pos_corres[:, :, 2:], image_indices)

        with tf.variable_scope('knn') as scope:
            # Get the nearest neighbours of the projected points of the features from the first
            # network at the correspondence points in the output features of the second network
            self.indices_nearest, self.flat_indices_nearest = knn(
                self.projection_corres_1, self.network2.feature)
            # Get the values of the nearest neighbours at their indices
            self.projection_nearest = project_at_points(
                feature2_proj, self.indices_nearest[:, :, :2], image_indices)
