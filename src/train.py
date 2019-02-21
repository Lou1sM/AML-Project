import tensorflow as tf
import highway_max_out
import andrei_encoder_script
import vlad_decoder_script1
import ciprian_data_prep_script

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
encoded = andrei_encode_script.encoder(input_d_vecs, input_q_vecs)


# Get lstm_dec hidden state values from U in order to make the
# first guess for start- and end-points
# Output should be of shape [2*hidden_size]
# Each guess is based on the previous guess so it seems we need
# to start with some random guess, eg first and last document words

h0 = tf.Variable(tf.zeros([hidden_size])) # Initialize LSTM to 0
u_s0 = encoded[0,:,:] # Dummy guess start point
u_e0 = encoded[-1,:,:] # Dummy guess end point
h1 = vlad_script1.lstm_dec(u_s0, u_e0, lstm_hidden_init)

# Make the first estimation of start- and end-points. Define the
# set of graph nodes for the HMN twice in two different namespaces
# so that two copies are created and they can be trained separately
with tf.variable_scope('start_estimator'):
    first_guess_start_points = highway_max_out_network.HMM(
        current_words=encoded,
        lstm_hidden_state=h1,
        prev_start_point_guess=dummy_guess_start_point,
        prev_end_point_guess=dummy_guess_end_point
        )

with tf.variable_scope('end_estimator'):
    first_guess_end_points = highway_max_out_network.HMM(
        current_words=encoded,
        lstm_hidden_state=h1,
        prev_start_point_guess=dummy_guess_start_point,
        prev_end_point_guess=dummy_guess_end_point
        )

s1 = tf.argmax(first_guess_start_points)
u_s1 = encoded[s1,:,:]
e1 = tf.argmax(first_guess_start_points)
u_e1 = encoded[s1,:,:]

# Feed these guesses back to lstm to make next guess, note this
# will reuse the weights from from above because the graph node
# names will be the same
h2 = vlad_script1.lstm_dec(u_s1, u_e1, h1)
with tf.variable_scope('start_estimator'):
    second_guess_start_points = highway_max_out_network.HMM(
        current_words=encoded,
        lstm_hidden_state=h2,
        prev_start_point_guess=first_guess_start_point,
        prev_end_point_guess=first_guess_end_point
        )

with tf.variable_scope('end_estimator'):
    second_guess_end_points = highway_max_out_network.HMM(
        current_words=encoded,
        lstm_hidden_state=h2,
        prev_start_point_guess=first_guess_start_point,
        prev_end_point_guess=first_guess_end_point
        )

s2 = tf.argmax(first_guess_start_points)
u_s2 = encoded[s2,:,:]
e2 = tf.argmax(first_guess_start_points)
u_e2 = encoded[s2,:,:]
    

# ..repeat as many times as we want to iterate, the authors say
# they use 4; of course we can implement a for loop, but just
# writing explicitly here for clarity

# Define loss and optimizer, each guess contributes to loss,
# even the very first
s_loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=labels[0],
    logits=first_guess_start_points
    )

e_loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=labels[1],
    logits=first_guess_end_points
    )

s_loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=labels[0],
    logits=first_guess_start_points
    )

e_loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=labels[1],
    logits=first_guess_end_points
    )

#..etc
loss = s_loss1 + e_loss1 + s_loss_2 + e_loss_2
optimizer = tf.train.AdamOptimizer(1e-3) # They don't give learning rate!
train_step = optimizer.minimize(loss)

# For tensorboard
writer = tf.summary.FileWriter("summaries/")
writer.add_graph(tf.get_default_graph())


# Define the session, which will use defaul_graph, ie the
# graph that contains all the nodes defined in the current
# script (I think that's what default_graph is anyway
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # This bit will depend on data format, we can also
    # add summaries with 'writer' every so often, for
    # tensorboard loss visualization
    for epoch in range(num_epochs):
        for d in range(num_batches):
            sess.run(train_step)
