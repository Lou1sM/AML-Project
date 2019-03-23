import tensorflow as tf
import numpy as np
import os
from tensorflow.python import debug as tf_debug
from math import floor

def new_weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def new_bias_variable(shape,positive=True):
    initial = tf.truncated_normal(shape, stddev=.05)
    """if positive:
        initial = tf.math.abs(initial)"""
    return tf.Variable(initial)


def weight_variable(shape, name="weights"):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable(name=name, initializer=initial)

#tf.get_variable(name=name, initializer=initial)


def bias_variable(shape, name="biases", positive=True):
    initial = tf.truncated_normal(shape, stddev=.05)
    """if positive:
        initial = tf.math.abs(initial)"""
    return tf.get_variable(name=name, initializer=initial)


def autoencoder(autoencodee, latent_dims):
    with tf.variable_scope("encoder"):
        autoencodee_shape = autoencodee.shape
        colour_channels = tf.cast(autoencodee_shape[-1], tf.int32)
        with tf.variable_scope("1"):
            w_conv1_e = weight_variable([8,8,colour_channels,16])
            b_conv1_e = bias_variable([16])
            h_conv1_e = tf.nn.conv2d(autoencodee, filter=w_conv1_e, strides=(1,2,2,1), padding="VALID")
            h_conv1_e = tf.nn.leaky_relu(tf.nn.bias_add(h_conv1_e, b_conv1_e))
        with tf.variable_scope("2"):
            w_conv2_e = weight_variable([5,5,16,32])
            b_conv2_e = bias_variable([32])
            h_conv2_e = tf.nn.conv2d(h_conv1_e, filter=w_conv2_e, strides=(1,2,2,1), padding="VALID")
            h_conv2_e = tf.nn.leaky_relu(tf.nn.bias_add(h_conv2_e, b_conv2_e))

        #print('e1:',h_conv1_e.get_shape())
        #print('e2:',h_conv2_e.get_shape())
        #h_conv2_e = tf.nn.max_pool(h_conv2_e, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

    latent_size = np.prod(h_conv2_e.get_shape().as_list()[1:])
    latent_1 = tf.reshape(h_conv2_e, [-1,latent_size])
                                              
    with tf.variable_scope("fc"):
        with tf.variable_scope("1"):
            w_fc1 = weight_variable([latent_size, latent_size//3])
            b_fc1 = bias_variable([latent_size//3])
            latent_2 = tf.nn.leaky_relu(tf.matmul(latent_1, w_fc1) + b_fc1)
        with tf.variable_scope("2"):
            w_fc2 = weight_variable([latent_size//3, latent_dims])
            b_fc2 = bias_variable([latent_dims])
            latent_3 = tf.nn.leaky_relu(tf.matmul(latent_2, w_fc2) + b_fc2)
        with tf.variable_scope("3"):
            w_fc3 = weight_variable([latent_dims, latent_size//3])
            b_fc3 = bias_variable([latent_size//3])
            latent_4 = tf.nn.leaky_relu(tf.matmul(latent_3, w_fc3) + b_fc3)
        with tf.variable_scope("4"):
            w_fc4 = weight_variable([latent_size//3, latent_size])
            b_fc4 = bias_variable([latent_size])
            latent_5 = tf.nn.leaky_relu(tf.matmul(latent_4, w_fc4) + b_fc4)


    fc_out = tf.reshape(latent_5, [-1] + h_conv2_e.get_shape().as_list()[1:])

    
    with tf.variable_scope("decoder"):
        with tf.variable_scope("1"):
            w_conv1_d = weight_variable([5,5,16,32])
            b_conv1_d = bias_variable([16])
            h_conv1_d = tf.nn.conv2d_transpose(fc_out, filter=w_conv1_d, output_shape=tf.shape(h_conv1_e), strides=(1,2,2,1), padding="VALID")
            h_conv1_d = tf.nn.leaky_relu(tf.nn.bias_add(h_conv1_d, b_conv1_d))
        with tf.variable_scope("2"):

            w_conv2_d = weight_variable([8,8,colour_channels,16])
            b_conv2_d = bias_variable([colour_channels])
            h_conv2_d = tf.nn.conv2d_transpose(h_conv1_d, filter=w_conv2_d, output_shape=tf.shape(autoencodee), strides=(1,2,2,1), padding="VALID")
            h_conv2_d = tf.nn.leaky_relu(tf.nn.bias_add(h_conv2_d,b_conv2_d))

    return h_conv2_d, h_conv1_e, h_conv2_e, fc_out, h_conv1_d


def time_format(x):
    """Convert a float of seconds to hhmmss format and return."""
    x = round(x, 1)
    if x < 60:
        return str(x) + 's'
    elif x < 3600:
        minutes = floor(x/60.0)
        x -= 60*minutes
        x = round(x, 1)
        return str(minutes) + 'm' + str(x) + 's'
    else:
        hours = floor(x/3600.0)
        x -= 3600*hours
        minutes = floor(x/60.0)
        x -= 60*minutes
        x = round(x, 1)
        return str(hours) + 'h' + str(minutes) + 'm' + str(x) + 's'


def compute_f1_from_indices(start_pred, end_pred, start_gs, end_gs):
    assert end_gs >= start_gs

    # Deal with degenerate case separately, NB: this may need
    # modification if using non-answerable questions
    if start_pred > end_pred:
        return 0

    tp = max(0, min(end_pred, end_gs) - max(start_pred, start_gs) + 1)
    fp = max(0, end_pred - end_gs) + max(0, start_gs - start_pred)
    fn = max(0, end_gs - end_pred) + max(0, start_pred - start_gs)

    if tp == 0:
        return 0
    # Total counts should add up to the union of words in pred and gs
    assert tp + fp + fn == max(0, max(end_pred, end_gs) - min(start_pred, start_gs) + 1), 'error counts dont add up'
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    f1 = 2/((1/prec) + (1/rec))
    return f1

# Creates summaries for a set of varaiables
def variable_summaries(var):
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.histogram('histogram', var)


if __name__ == "__main__":
    print(compute_f1_from_indices(1,2,3,4))
    print(compute_f1_from_indices(1,3,2,4))
    print(compute_f1_from_indices(1,4,2,4))
    print(compute_f1_from_indices(2,2,2,5))
    print(compute_f1_from_indices(100,2,3,4))
    print(compute_f1_from_indices(1,2,3,4))
