import tensorflow as tf
import highway_max_out
import andrei_encoder_script
import ciprian_data_prep_script

# Global variables
# We may want to move these to an object and pass it to components

# Learning Parameters
num_epochs = 100
num_batches = 32
learning_rate = 1e-3
dropout = 0

# Architecture Parameters
hidden_state_size = 200
num_decoding_lstm_layers = 1  # assumed to be 1 in paper

# Whatever the format of the data preparation, the output will
# be a document-length sequence of glove word vectors, and a
# question-length sequence of the same. Ideally it these would
# be generator objects with the faster tf.data.Dataset, but
# another format is ok if that's all that's possible
#
# For example:
input_d_vecs, input_q_vecs, ground_truth_labels = ciprian_data_prep_script.get_data()


# Encode into the matrix U, using notation from the paper
# The output should be of shape [2*hidden_size, document_length]
encoded = andrei_encoder_script.encoder(input_d_vecs, input_q_vecs)

# Create and initialize decoding LSTM
decoding_lstm = tf.contrib.rnn.LSTMCell(num_units = hidden_state_size)
h = decoding_lstm.zer_state(num_batches, dtype=tf.float32)


# Get lstm_dec hidden state values from U in order to make the
# first guess for start- and end-points
# Output should be of shape [2*hidden_size]
# Each guess is based on the previous guess so it seems we need
# to start with some random guess, eg first and last document words

u_s = encoded[0, :, :]  # Dummy guess start point
u_e = encoded[-1, :, :]  # Dummy guess end point

for i in range(4):

    usue = tf.concat([u_s, u_e], axis=1)
    # Here, "h" is the concatenation of cell output and cell state
    # (h_t and C_t in colah.github.io/posts/2015-08-Understanding-LSTMs/ )
    # decoding_lstm returns (h_t, (h_t, C_t))
    cell_output, h = decoding_lstm(inputs=usue, state=h)

    # Make the first estimation of start- and end-points. Define the
    # set of graph nodes for the HMN twice in two different namespaces
    # so that two copies are created and they can be trained separately
    with tf.variable_scope('start_estimator'):
        first_guess_start_points = highway_max_out.HMM(
            current_words=encoded,
            lstm_hidden_state=h,
            prev_start_point_guess=u_s,
            prev_end_point_guess=u_e
        )

    with tf.variable_scope('end_estimator'):
        first_guess_end_points = highway_max_out.HMM(
            current_words=encoded,
            lstm_hidden_state=h,
            prev_start_point_guess=u_s,
            prev_end_point_guess=u_e
        )

    s = tf.argmax(first_guess_start_points)
    u_s = encoded[s, :, :]
    e = tf.argmax(first_guess_start_points)
    u_e = encoded[e, :, :]

    # Define loss and optimizer, each guess contributes to loss,
    # even the very first
    s_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=ground_truth_labels[0],
        logits=first_guess_start_points
    )

    e_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=ground_truth_labels[1],
        logits=first_guess_end_points
    )

    iteration_loss = s_loss + e_loss

    loss = iteration_loss if i == 0 else loss + iteration_loss

# Set up learning process
optimizer = tf.train.AdamOptimizer(learning_rate)  # They don't give learning rate!
train_step = optimizer.minimize(loss)

# For Tensorboard
writer = tf.summary.FileWriter("summaries/")
writer.add_graph(tf.get_default_graph())


# Define the session, which will use default_graph, ie the
# graph that contains all the nodes defined in the current
# script (I think that's what default_graph is anyway)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # This bit will depend on data format, we can also
    # add summaries with 'writer' every so often, for
    # tensorboard loss visualization
    for _ in range(num_epochs):
        for _ in range(num_batches):
            sess.run(train_step)
