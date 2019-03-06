import tensorflow as tf
import highway_max_out
import encoder
import ciprian_data_prep_script
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", default=1, type=int, help="Number of epochs to train for")
parser.add_argument("--restore", action="store_true", default=False, help="Whether to restore weights from previous run")
parser.add_argument("--num_units", default=200, type=int, help="Number of recurrent units for the first lstm, which is deteriministic and is only used in both training and testing")
parser.add_argument("--test", "-t", default=False, action="store_true", help="Whether to run in test mode")
parser.add_argument("--batch_size", default=10, type=int, help="Size of each training batch")
parser.add_argument("--dataset", choices=["SQuAD"],default="SQuAD", type=str, help="Dataset to train and evaluate on")
parser.add_argument("--hidden_size", default=200, type=int, help="Size of the hidden state")
parser.add_argument("--keep_prob", default=1, type=float, help="Keep probability for question and document encodings.")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate.")
ARGS = parser.parse_args()

input_d_vecs, input_q_vecs, ground_truth_labels, documents_lengths, questions_lengths = ciprian_data_prep_script.get_data()

# dataset = tf.data.Dataset.from_tensor_slices((input_d_vecs,input_q_vecs, ground_truth_labels))
# feed_dict = {d: input_d_vecs, q: input_q_vecs, a: ground_truth_labels, doc_l: documents_lengths, que_l: questions_lengths}
# dataset = tf.data.Dataset.from_tensor_slices(
# 		(input_d_vecs, input_q_vecs, ground_truth_labels, documents_lengths, questions_lengths)
# 	).batch(ARGS.batch_size)

# iterator = dataset.make_initializable_iterator()
# init = iterator.make_initializer(dataset)
# d, q, a, doc_l, que_l = iterator.get_next()


d = tf.placeholder(tf.float64, [ARGS.batch_size, len(input_d_vecs[0])])
q = tf.placeholder(tf.float64, [ARGS.batch_size, len(input_q_vecs[0])])
a = tf.placeholder(tf.float64, [ARGS.batch_size, len(ground_truth_labels[0])])
documents_lengths = tf.placeholder(tf.float64, [ARGS.batch_size])
questions_lengths = tf.placeholder(tf.float64, [ARGS.batch_size])

encoded = encoder.encoder(
					document=d,
					question=q,
					documents_lengths = doc_l,
					questions_lengths = que_l,
                    hyperparameters = ARGS
				)

'''
# Testing code
encoded = tf.random.uniform([600, batch_size, 2*ARGS.hidden_size])
new_ground_truth_labels = tf.random.uniform([2, batch_size, 600])
'''

# Static tensor to be re-used in main loop iterations
batch_indices = tf.range(start=0, limit=ARGS.batch_size, dtype=tf.int32)

# Create single nodes for labels
a = tf.transpose(a)
start_labels = tf.squeeze(tf.gather(a, [0]), [0])
end_labels = tf.squeeze(tf.gather(a, [1]), [0])
start_labels = tf.one_hot(start_labels, 766)
end_labels = tf.one_hot(end_labels, 766)

# Create and initialize decoding LSTM
decoding_lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
    num_layers=1,
    num_units=ARGS.hidden_size)

# first guess for start- and end-points
u_s = tf.zeros([ARGS.batch_size, 2*ARGS.hidden_size])  # Dummy guess start point
u_e = tf.zeros([ARGS.batch_size, 2*ARGS.hidden_size])  # Dummy guess end point

for i in range(4):

    # LSTM input is concatenation of previous guesses
    usue = tf.concat([u_s, u_e], axis=1)

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
            name="HMN_start"
        )

    #alphas = tf.squeeze(tf.transpose(alphas), [0])

    with tf.variable_scope('end_estimator'):
        betas = highway_max_out.HMN(
            current_words=encoded,
            lstm_hidden_state=lstm_output_reshaped,
            prev_start_point_guess=u_s,
            prev_end_point_guess=u_e,
            name="HMN_end"
        )
    #betas = tf.squeeze(tf.transpose(betas), [0])

    s = tf.argmax(alphas, axis=1, output_type=tf.int32)
    s_indices = tf.transpose(tf.stack([s, batch_indices]))
    u_s = tf.gather_nd(encoded, s_indices)

    e = tf.argmax(betas, axis=1, output_type=tf.int32)
    e_indices = tf.transpose(tf.stack([e, batch_indices]))
    u_e = tf.gather_nd(encoded, e_indices)

    # Define loss and optimizer, each guess contributes to loss,
    # even the very first
    print('alphas:', alphas.get_shape())
    print('start_labels:', start_labels.get_shape())
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
        #tf.summary.scalar('loss', iteration_loss)

    loss = iteration_loss if i == 0 else loss + iteration_loss

# Keep track of loss
#tf.summary.scalar('total_loss', loss)

# Set up learning process
optimizer = tf.train.AdamOptimizer(ARGS.learning_rate)  # They don't give learning rate!
train_step = optimizer.minimize(loss)

# For Tensorboard
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("summaries/")
writer.add_graph(tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch_size = ARGS.batch_size
    #for i in range(ARGS.num_epochs):
    for i in range(100):
        sess.run(init)
        feed_dict = {d: input_d_vecs[:batch_size], q: input_q_vecs[:batch_size], a: ground_truth_labels[:batch_size], doc_l: documents_lengths[:batch_size], que_l: questions_lengths[:batch_size]}
        for _ in range(1):  # for now
            summary, _, loss_val = sess.run([merged, train_step, loss], feed_dict)
            print(loss_val)
       # currently write summary for each epoch
        writer.add_summary(summary, i)
