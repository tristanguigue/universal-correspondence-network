import tensorflow as tf


def unravel_index(indices, shape):
    output_list = []
    output_list.append(indices // (shape[0] * shape[1]))
    output_list.append(indices % (shape[0] * shape[1]) // shape[1])
    return tf.transpose(tf.stack(output_list), [1, 2, 0])
