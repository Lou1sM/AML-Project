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
from tensorflow.python.client import timeline

start_time = time.time()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('tensorflow').setLevel(50)


parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", default=100, type=int, help="Number of epochs to train for")
parser.add_argument("--restore", action="store_true", default=False, help="Whether to restore weights from previous run")
#parser.add_argument("--num_units", default=200, type=int,help="Number of recurrent units for the first lstm, which is deteriministic and is only used in both training and testing")
parser.add_argument("--test", "-t", default=False, action="store_true", help="Whether to run in test mode")
parser.add_argument("--batch_size", default=256, type=int, help="Size of each training batch")
parser.add_argument("--dataset", choices=["SQuAD"],default="SQuAD", type=str, help="Dataset to train and evaluate on")
parser.add_argument("--hidden_size", default=100, type=int, help="Size of the hidden state")
parser.add_argument("--keep_prob", default=1, type=float, help="Keep probability for question and document encodings.")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--short_test", "-s", default=False, action="store_true", help="Whether to run in short test mode")
parser.add_argument("--pool_size", default=8, type=int, help="Number of units to pool over in HMN sub-network")
parser.add_argument("--validate", default=False, action="store_true", help="Whether to apply validation.")
parser.add_argument("--early_stop", default=None, type=int, help="Number of epochs without improvement before applying early-stopping. Defaults to num_epochs, which amounts to no early-stopping.")

ARGS = parser.parse_args()

if ARGS.early_stop != None: 
    ARGS.validate = True



with tf.name_scope("data_prep"):
    input_d_vecs, input_q_vecs, ground_truth_labels, documents_lengths, questions_lengths, questions_ids = ciprian_data_prep_script.get_data(ARGS.test)
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
print(ARGS.validate)
dataset = tfrecord_converter.read_tfrecords(file_names=('test.tfrecord'))
validation_dataset = tfrecord_converter.read_tfrecords(file_names=('test.tfrecord'))
validation_dataset = validation_dataset.shuffle(1000)
validation_dataset = validation_dataset.batch(ARGS.batch_size).prefetch(1)
#dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensorslices(x))
#dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensorslices(x))
dataset = dataset.shuffle(1000)
dataset = dataset.batch(ARGS.batch_size).prefetch(1)
iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)

train_init_op = iterator.make_initializer(dataset)
val_init_op = iterator.make_initializer(validation_dataset)
data_dict = iterator.get_next()
d = data_dict['D']
q = data_dict['Q']
a = data_dict['A']
doc_l = data_dict['DL']
que_l = data_dict['QL']

"""

with tf.name_scope("encoder"):
    encoded = encoder.encoder(
        document=d,
        question=q,
        documents_lengths=doc_l,
        questions_lengths=que_l,
        hyperparameters=ARGS
    )

    # Create single nodes for labels, data version
    #start_labels = tf.one_hot(a[:,0], 766)
    #end_labels = tf.one_hot(a[:,1], 766)

    # Create single nodes for labels, feed_dict version
    start_labels = tf.one_hot(starting_labels, 766)
    end_labels = tf.one_hot(ending_labels, 766)

with tf.name_scope("decoder"):
    # Static tensor to be re-used in main loop iterations
    batch_indices = tf.range(start=0, limit=ARGS.batch_size, dtype=tf.int32)

    # Create and initialize decoding LSTM
    decoding_lstm = tf.contrib.rnn.LSTMCell(num_units=ARGS.hidden_size)
    h = decoding_lstm.zero_state(ARGS.batch_size, dtype=tf.float32)

    # first guess for start- and end-points
    u_s = tf.zeros([ARGS.batch_size, 2 * ARGS.hidden_size])  # Dummy guess start point
    u_e = tf.zeros([ARGS.batch_size, 2 * ARGS.hidden_size])  # Dummy guess end point

    with tf.name_scope("decoding_loop"):
        for i in range(4):
            # LSTM input is concatenation of previous guesses
            usue = tf.concat([u_s, u_e], axis=1)
            # CudnnLSTM expects input with shape [time_len, batch_size, input_size]
            # As the inputs across time depend on the previous outputs, we simply set
            # time_len = 0 and repeat the calculation multiple times while setting
            # the 'initial' state to the previous output. CudnnLSTMs output is
            # similarly time-major (has a first dimension that spans time).

            # an initial_state of None corresponds to initialisation with zeroes
            lstm_output, h = decoding_lstm(inputs=usue, state=h)
            # Make an estimation of start- and end-points. Define the
            # set of graph nodes for the HMN twice in two different namespaces
            # so that two copies are created and they can be trained separately

            with tf.variable_scope('start_estimator'):
                alphas = highway_max_out.HMN(
                    current_words=encoded,
                    lstm_hidden_state=lstm_output,
                    prev_start_point_guess=u_s,
                    prev_end_point_guess=u_e,
                    pool_size=ARGS.pool_size,
                    h_size=ARGS.hidden_size,
                    name="HMN_start"
                )

            with tf.variable_scope('end_estimator'):
                betas = highway_max_out.HMN(
                    current_words=encoded,
                    lstm_hidden_state=lstm_output,
                    prev_start_point_guess=u_s,
                    prev_end_point_guess=u_e,
                    pool_size=ARGS.pool_size,
                    h_size=ARGS.hidden_size,
                    name="HMN_end"
                )

            s = tf.argmax(alphas, axis=1, output_type=tf.int32)
            s_indices = tf.transpose(tf.stack([batch_indices, s]))
            u_s = tf.gather_nd(encoded, s_indices)

            e = tf.argmax(betas, axis=1, output_type=tf.int32)
            e_indices = tf.transpose(tf.stack([batch_indices, e]))
            u_e = tf.gather_nd(encoded, e_indices)

            # Each guess contributes to loss,
            # even the very first
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
loss_val = -1
if ARGS.early_stop != None:
    tolerance = ARGS.early_stop
else:
    tolerance = ARGS.num_epochs
best_loss_val = -1

if(!ARGS.test):
	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    train_start_time = time.time()
	    print("Graph-build time: ", utils.time_format(train_start_time - start_time))

	    dataset_length = len(input_d_vecs)
	    num_batchs = dataset_length//ARGS.batch_size 

	    for epoch in range(ARGS.num_epochs):
	        print("\nEpoch:", epoch)
	        for batch_num in range(0,dataset_length,ARGS.batch_size):
	            if batch_num+ARGS.batch_size >= len(input_d_vecs):
	                break
	            batch_time = time.time()
	            feed_dict={
	                d:input_d_vecs[batch_num:batch_num+ARGS.batch_size], 
	                q:input_q_vecs[batch_num: batch_num+ARGS.batch_size], 
	                starting_labels: start_l[batch_num: batch_num+ARGS.batch_size], 
	                ending_labels: end_l[batch_num: batch_num+ARGS.batch_size], 
	                doc_l: documents_lengths[batch_num: batch_num+ARGS.batch_size], 
	                que_l: questions_lengths[batch_num: batch_num+ARGS.batch_size]
	                }

	            if batch_num//ARGS.batch_size % 10 == 0:
	                _, loss_val, summary_val = sess.run([train_step, mean_loss, merged], feed_dict=feed_dict)
	                writer.add_summary(summary_val, i)
	                print("\tBatch: {}\tloss: {}".format(batch_num//ARGS.batch_size, loss_val)) 
	            else:
	                sess.run([train_step], feed_dict=feed_dict) 
	            print(time.time()-batch_time)
	        save_path = saver.save(sess, "checkpoints/model.ckpt.{}".format(epoch))
	        print("Model saved in path: %s" % save_path)

	train_end_time = time.time()

	print("Train time", utils.time_format(train_end_time - train_start_time))
else:
	with tf.Session() as sess:
	    saver = tf.train.import_meta_graph('checkpoints/model.ckpt.55.meta')
	    print("Loaded graph")
	    
	    saver.restore(sess, tf.train.latest_checkpoint('checkpoints/'))
	    print("Loaded weights")

        for batch_num in range(0, dataset_length,ARGS.batch_size):
	        if batch_num + ARGS.batch_size >= len(input_d_vecs):
	            break

		    feed_dict = {
	            d: input_d_vecs[batch_num:batch_num + ARGS.batch_size], 
	            q: input_q_vecs[batch_num: batch_num + ARGS.batch_size], 
	            starting_labels: start_l[batch_num: batch_num + ARGS.batch_size], 
	            ending_labels: end_l[batch_num: batch_num + ARGS.batch_size], 
	            doc_l: documents_lengths[batch_num: batch_num + ARGS.batch_size], 
	            que_l: questions_lengths[batch_num: batch_num + ARGS.batch_size]
	            }

	        _, loss_val, start_predict, end_predict = sess.run([train_step, mean_loss, s_indices, e_indices], feed_dict = feed_dict)
	        start_correct = start_l[batch_num: batch_num + ARGS.batch_size]
	        end_correct = end_l[batch_num: batch_num + ARGS.batch_size]

	        for i in range(ARGS.batch_size):
	        	print("Question with ID: ", questions_ids[batch_num + i])
	        	print("Correct (start, end): ", (start_correct[i], end_correct[i]))
	        	print("Predicted (start, end): ", (start_predict[i], end_predict[i]))
	        	print("___________________________")

"""
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_start_time = time.time()
    print("Graph-build time: ", utils.time_format(train_start_time - start_time))
    for i in range(ARGS.num_epochs):
        sess.run(train_init_op)
        profileFirstBatch = False
        while True:
            try:
                if profileFirstBatch:
                    profileFirstBatch = False
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                    summary, _, loss_val = sess.run([merged, train_step, mean_loss],
                                                    options=options,
                                                    run_metadata=run_metadata)

                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open('profiling_timeline.json', 'w') as f:
                        f.write(chrome_trace)
                else:
                    summary, _, loss_val = sess.run([merged, train_step, mean_loss])
                    #print(loss_val)
                    if ARGS.test:
                        break
            except tf.errors.OutOfRangeError:
                writer.add_summary(summary, i)
                if ARGS.validate:
                    print('\nComputing validiation loss:')
                    sess.run(val_init_op)
                    new_loss_val = 0
                    while True:
                        try:
                            new_loss_val += sess.run(mean_loss)
                        except tf.errors.OutOfRangeError:
                            print(new_loss_val)
                            if new_loss_val < best_loss_val or best_loss_val == -1:
                                best_loss_val = new_loss_val
                                save_path = saver.save(sess, "checkpoints/model.ckpt")
                                print("Model saved in path: %s" % save_path)
                            else:
                                print("No improvement in loss, not saving")
                                tolerance -= 1
                            break
                break           

        if tolerance == 0:
            print('Tolerance threshold reached, early-stopping')
            break

    train_end_time = time.time()

    print("Train time", utils.time_format(train_end_time - train_start_time))

"""
