import tensorflow as tf
from abc import ABC, abstractmethod


class Learner(ABC):
    """Generic learner.

    Attributes:
        net: the network to use for learning
        lr: the learning rate of the optimiser
        loss_op: the loss to be defined by the learners
    """

    def __init__(self, network, learning_rate):
        self.lr = tf.placeholder(tf.float32)
        self.net = network
        self.learning_rate = learning_rate
        self.loss_op = self.loss()

        with tf.name_scope('train'):
            self.train_step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_op)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    @abstractmethod
    def loss(self):
        pass

    def train_network(self, image1_batch, image2_batch, correspondences, learning_rate):
        """Train the network using the two inputs.

        Args:
            image1_batch: input of the first part of the siamese network
            image2_batch: intput of the second part of the siamese network
            correspondences: the correspondence points
            learning_rate: the learning rate for the optimiser

        Return:
            The current loss
        """
        if learning_rate:
            self.learning_rate = learning_rate
        feed_dict = {
            self.net.x1: image1_batch,
            self.net.x2: image2_batch,
            self.net.correspondences: correspondences,
            self.lr: self.learning_rate
        }

        _, current_loss = self.sess.run(
            [self.train_step, self.loss_op],
            feed_dict=feed_dict)

        return current_loss


class CorrespondenceLearner(Learner):
    """Learner specific of the correspondence contrastive loss.

    Attributes:
        margin: the negative pairs are encouraged to be at least of the margin
            apart
    """

    def __init__(self, network, learning_rate, margin):
        self.margin = margin
        super().__init__(network, learning_rate)

    def loss(self):
        """The correspondence contrastive loss function."""
        feature_size_x = self.net.network1.feature.get_shape()[1]
        flat_indices_corres = self.net.pos_corres[:, :, 2] * feature_size_x + self.net.pos_corres[:, :, 3]

        # Loss for the nearest neighbour, negative pairs and positive paris are separated in the
        # where
        nearest_loss = tf.where(
            tf.equal(flat_indices_corres, self.net.flat_indices_nearest),
            tf.norm(self.net.projection_corres_1 - self.net.projection_nearest, axis=2),
            tf.square(tf.maximum(0., self.margin - tf.sqrt(
                tf.norm(self.net.projection_corres_1 - self.net.projection_nearest, axis=2)))))

        # The second part of the loss correspond to only positives given by the correspondence points
        return 0.5 * tf.reduce_mean(nearest_loss) + tf.reduce_mean(tf.norm(
            self.net.projection_corres_2 - self.net.projection_corres_1, axis=2))
