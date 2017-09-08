import argparse
import pykitti
import numpy as np
from learners import CorrespondenceLearner
from networks import UniversalCorrepondenceNetwork
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def main(last_frame, step_frame, nb_corres_points, learning_rate):
    """Train the universal correspondence network .

    Args:
        last_frame: last image to be loaded.
        step_frame: load every step_frame images
        nb_corres_points: Number of velodyne correspondence points to load
        learning_rate: learning rate of the optimiser
    """
    logger = logging.getLogger(__name__)
    data = pykitti.raw('data', '2011_09_26', '0002', frames=range(0, last_frame, step_frame))

    # Load the velodyne correspondence points
    correspondences = []
    for i, velo_points in enumerate(data.velo):
        velo_points = velo_points[:nb_corres_points]

        # Project the points in the images of the left and right cameras
        cam2 = velo_points.dot(np.transpose(data.calib.T_cam2_velo))
        cam3 = velo_points.dot(np.transpose(data.calib.T_cam3_velo))

        # Only gather the x and y position on the images
        correspondences.append(np.concatenate((cam2[:, :2], cam3[:, :2]), axis=1))

    ucn = UniversalCorrepondenceNetwork([375, 1242, 3], nb_corres_points)
    learner = CorrespondenceLearner(ucn, 0.001, 10)

    loss = learner.train_network(np.array(list(data.cam2)), np.array(list(data.cam3)),
                                 np.array(correspondences), learning_rate)
    logger.info('Loss: {loss}'.format(loss=loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_stop', type=int, default=50,
                        help='last frame')
    parser.add_argument('--frames_step', type=int, default=5,
                        help='frames step')
    parser.add_argument('--corres', type=int, default=1000,
                        help='number of correspondences points')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')

    args = parser.parse_args()
    main(args.frames_stop, args.frames_step, args.corres, args.learning_rate)
