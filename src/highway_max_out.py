import tensorflow as tf
import numpy as np
from utils import weight_variable, bias_variable

from utils import variable_summaries

h_size=200
pool_size=32
# Shape=(batch_size, 5*hidden_state, document_length)
dummy = tf.random.uniform([600,32,400])
#print(dummy)

def HMN(current_words, lstm_hidden_state, prev_start_point_guess, prev_end_point_guess, name):

    with tf.name_scope(name):

        r = tf.concat([lstm_hidden_state, prev_start_point_guess, prev_end_point_guess], axis=1)

        with tf.variable_scope("layer1", reuse=tf.AUTO_REUSE):
            wd = weight_variable([5 * h_size, h_size])
            variable_summaries(wd)
            r = tf.nn.tanh(tf.matmul(r, wd))
            print('r shape:', r.get_shape())

        with tf.variable_scope("layer2", reuse=tf.AUTO_REUSE):
            w1 = weight_variable([3 * h_size, h_size, pool_size])
            variable_summaries(w1)
            b1 = bias_variable([h_size, pool_size])
            variable_summaries(b1)
            concated_words = tf.map_fn(lambda x: tf.concat([x, r], axis=1), current_words)
            mt1 = tf.math.add(tf.tensordot(concated_words, w1, axes=[[2], [0]]), b1)
            mt1 = tf.reduce_max(mt1, reduction_indices=[3])
            print('mt1 shape:', mt1.get_shape())

        with tf.variable_scope("layer3", reuse=tf.AUTO_REUSE):
            w2 = weight_variable([h_size, h_size, pool_size])
            variable_summaries(w2)
            b2 = bias_variable([h_size, pool_size])
            variable_summaries(b2)
            mt2 = tf.math.add(tf.tensordot(mt1, w2, axes=[[2], [0]]), b2)
            mt2 = tf.reduce_max(mt2, reduction_indices=[3])
            print('mt2 shape:', mt2.get_shape())

        with tf.variable_scope("layer4", reuse=tf.AUTO_REUSE):
            w3 = weight_variable([h_size, 1, pool_size])
            variable_summaries(w3)
            b3 = bias_variable([pool_size])
            variable_summaries(b3)
            mt3 = tf.math.add(tf.tensordot(mt2, w3, axes=[[2], [0]]), b3)
            mt3 = tf.reduce_max(mt3, reduction_indices=[3])
            print('mt2 shape:', mt3.get_shape())

        return mt3



if __name__ == "__main__":
    hmn_out = HMN(dummy, tf.random.uniform([32,200]), tf.random.uniform([32,2*h_size]), tf.random.uniform([32,2*h_size]))
    
    print('output shape:', hmn_out.get_shape())
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run(hmn_out))

