import tensorflow as tf
import highway_max_out
import encoder
import ciprian_data_prep_script
import tfrecord_converter
import argparse
import time
import datetime
import os
import utils

start_time = time.time()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('tensorflow').setLevel(50)


parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", default=1, type=int, help="Number of epochs to train for")
parser.add_argument("--restore", action="store_true", default=False, help="Whether to restore weights from previous run")
parser.add_argument("--num_units", default=200, type=int, 
	help="Number of recurrent units for the first lstm, which is deteriministic and is only used in both training and testing")
parser.add_argument("--test", "-t", default=False, action="store_true", help="Whether to run in test mode")
parser.add_argument("--batch_size", default=10, type=int, help="Size of each training batch")
parser.add_argument("--dataset", choices=["SQuAD"],default="SQuAD", type=str, help="Dataset to train and evaluate on")
parser.add_argument("--hidden_size", default=200, type=int, help="Size of the hidden state")
parser.add_argument("--keep_prob", default=1, type=float, help="Keep probability for question and document encodings.")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--short_test", "-s", default=False, action="store_true", help="Whether to run in short test mode")

ARGS = parser.parse_args()

"""
with tf.name_scope("data_prep"):
    input_d_vecs, input_q_vecs, ground_truth_labels, documents_lengths, questions_lengths = ciprian_data_prep_script.get_data()
    print("In train.py: get_data finished.")
    start_l = list(map(lambda x: x[0], ground_truth_labels))
    end_l = list(map(lambda x: x[1], ground_truth_labels))
    d = tf.placeholder(tf.float64, [ARGS.batch_size, len(input_d_vecs[0]), len(input_d_vecs[0][0])])
    q = tf.placeholder(tf.float64, [ARGS.batch_size, len(input_q_vecs[0]), len(input_q_vecs[0][0])])
    starting_labels = tf.placeholder(tf.int64, [ARGS.batch_size])
    ending_labels = tf.placeholder(tf.int64, [ARGS.batch_size])
    doc_l = tf.placeholder(tf.int64, [ARGS.batch_size])
    que_l = tf.placeholder(tf.int64, [ARGS.batch_size])
"""

dataset = tfrecord_converter.read_tfrecords(file_names=('test.tfrecord'))
#dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensorslices(x))
dataset = dataset.batch(ARGS.batch_size)
iter_ = dataset.make_initializable_iterator()
dd = iter_.get_next()
print(dd)
print(type(dd))
d = dd['D']
q = dd['Q']
a = dd['A']
doc_l = dd['DL']
que_l = dd['QL']

print(d.get_shape())
print(q.get_shape())
with tf.name_scope("encoder"):
    encoded = encoder.encoder(
        document=d,
        question=q,
        documents_lengths=doc_l,
        questions_lengths=que_l,
        hyperparameters=ARGS
    )
    print('enc', encoded.get_shape())
    # Create single nodes for labels
    print(a.get_shape())
    print(a[0].get_shape())
    start_labels = tf.one_hot(a[:,0], 766)
    end_labels = tf.one_hot(a[:,1], 766)

with tf.name_scope("decoder"):
    # Static tensor to be re-used in main loop iterations
    batch_indices = tf.range(start=0, limit=ARGS.batch_size, dtype=tf.int32)

    # Create and initialize decoding LSTM
    decoding_lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
        num_layers=1,
        num_units=ARGS.hidden_size)

    # first guess for start- and end-points
    u_s = tf.zeros([ARGS.batch_size, 2 * ARGS.hidden_size])  # Dummy guess start point
    u_e = tf.zeros([ARGS.batch_size, 2 * ARGS.hidden_size])  # Dummy guess end point

    with tf.name_scope("decoding_loop"):
        for i in range(4):
            # LSTM input is concatenation of previous guesses
            usue = tf.concat([u_s, u_e], axis=1)
            print('usue', usue.get_shape())
            # CudnnLSTM expects input with shape [time_len, batch_size, input_size]
            # As the inputs across time depend on the previous outputs, we simply set
            # time_len = 0 and repeat the calculation multiple times while setting
            # the 'initial' state to the previous output. CudnnLSTMs output is
            # similarly time-major (has a first dimension that spans time).

            # an initial_state of None corresponds to initialisation with zeroes
            with tf.variable_scope("decoding_lstm", reuse=tf.AUTO_REUSE):
                lstm_output, h = decoding_lstm(
                    inputs=tf.expand_dims(input=usue, axis=0),
                    initial_state=None if i == 0 else h,
                    training=True)
                lstm_output_reshaped = tf.squeeze(lstm_output, [0])

            # Make an estimation of start- and end-points. Define the
            # set of graph nodes for the HMN twice in two different namespaces
            # so that two copies are created and they can be trained separately

            with tf.variable_scope('start_estimator'):
                alphas = highway_max_out.HMN(
                    current_words=encoded,
                    lstm_hidden_state=lstm_output_reshaped,
                    prev_start_point_guess=u_s,
                    prev_end_point_guess=u_e,
                    hyperparameters=ARGS,
                    name="HMN_start"
                )

            with tf.variable_scope('end_estimator'):
                betas = highway_max_out.HMN(
                    current_words=encoded,
                    lstm_hidden_state=lstm_output_reshaped,
                    prev_start_point_guess=u_s,
                    prev_end_point_guess=u_e,
                    hyperparameters=ARGS,
                    name="HMN_end"
                )

            s = tf.argmax(alphas, axis=1, output_type=tf.int32)
            s_indices = tf.transpose(tf.stack([s, batch_indices]))
            u_s = tf.gather_nd(encoded, s_indices)

            e = tf.argmax(betas, axis=1, output_type=tf.int32)
            e_indices = tf.transpose(tf.stack([e, batch_indices]))
            u_e = tf.gather_nd(encoded, e_indices)

            # Each guess contributes to loss,
            # even the very first
            print('start_labels:', start_labels.get_shape())
            print('alphas:', alphas.get_shape())
            s_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=start_labels,
                logits=alphas
            )
            e_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=end_labels,
                logits=betas
            )

            with tf.name_scope("iteration_" + str(i) + "_loss"):
                iteration_loss = s_loss + e_loss
                tf.summary.scalar('loss', tf.reduce_mean(iteration_loss))

            loss = iteration_loss if i == 0 else loss + iteration_loss

mean_loss = tf.reduce_mean(loss)
tf.summary.scalar('total_loss', mean_loss)
optimizer = tf.train.AdamOptimizer(ARGS.learning_rate)
train_step = optimizer.minimize(mean_loss)

# For Tensorboard
merged = tf.summary.merge_all()
summaryDirectory = "summaries/start_" + str(datetime.datetime.now())
summaryDirectory = summaryDirectory.replace('.', '_').replace(':', '-').replace(' ', '_')
tf.gfile.MkDir(summaryDirectory)
writer = tf.summary.FileWriter(summaryDirectory)
writer.add_graph(tf.get_default_graph())

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch_size = ARGS.batch_size
    #for i in range(ARGS.num_epochs):
    train_start_time = time.time()
    print("Time elapsed from beginning until right before starting train is: ", utils.time_format(train_start_time - start_time))
    for i in range(ARGS.num_epochs):
        while True:
            try:
                summary, _, loss_val, meanloss = sess.run([merged, train_step, loss, mean_loss], feed_dict)
                print("Epoch: ", i, ", Batch: ",j,", Loss: ",meanloss)
            except tf.errors.OutOfRangeError:
                writer.add_summary(summary, i)
                break

    train_end_time = time.time()

    print("Total training time (without data reading): ", utils.time_format(train_end_time - train_start_time))

    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)

