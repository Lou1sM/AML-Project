import numpy as np
import tensorflow as tf

# Note: can't use CudNN as it is not yet adapated (in TF)
# to take batches of variable sizes (NVidia recently added this feature to their ML API ~ 1 month ago),
# meanwhile, the classic tf.nn.dynamic_rnn supports adding a vector of batch_size elements, each
# describing the length of the documents/questions to process.

# Provide hyperparameters to functions below as dictionary with keys "num_units", "keep_prob", "batch_size"

def build_lstm_cell(num_units = 128, keep_prob = 1, batch_size = 32): 
    lstm = tf.contrib.rnn.BasicLSTMCell(num_units)
    cell = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob = keep_prob)
    initial_state = cell.zero_state(batch_size, tf.float32)
    return initial_state, cell
 

def dynamic_lstm(embed, hyperparameters):
    num_units = hyperparameters["num_units"]
    keep_prob = hyperparameters["keep_prob"]
    batch_size = hyperparameters["batch_size"]
    initial_state, cell = build_lstm_cell(num_units, keep_prob, batch_size)
    lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state = initial_state)
    return lstm_outputs, final_state
 
def dynamic_bilstm(embed, hyperparameters):
    num_units = hyperparameters["num_units"]
    keep_prob = hyperparameters["keep_prob"]
    batch_size = hyperparameters["batch_size"]
    initial_fw_state, fw_cell = build_lstm_cell(num_units, keep_prob, batch_size)
    initial_bw_state, bw_cell = build_lstm_cell(num_units, keep_prob, batch_size)
    lstm_outputs, final_state = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(fw_cell, bw_cell, embed,
                                initial_states_fw = initial_fw_state, initial_states_bw = initial_bw_state)
    return lstm_outputs, final_state
 

def doc_que_encoder(document_columns, question_columns, hyperparameters, document_lengths=np.array([100]), question_lengths=np.array([4])):
    # Use batch_size from hyperparameters, dropout, num_cells
    # Data needs to come padded, also need the length 
    with tf.variable_scope('lstm') as scope:
        document_enc, final_state_doc = dynamic_lstm(document_columns, document_lengths, hyperparameters)
        scope.reuse_variables()
        que_lstm_outputs, final_state_que = dynamic_lstm(question_columns, question_lengths, hyperparameters)
    with tf.variable_scope('tanhlayer') as scope:
        linear_model = tf.layers.Dense(units = 128)
        question_enc = tf.math.tanh(linear_model(que_lstm_outputs))
 
    return document_enc, question_enc

# Once we agree on shapes, use code below to write tensor as vector
# of length timesteps, with 2D tensors as elements:
# x=tf.placeholder("float",[None,time_steps,n_input])
# input=tf.unstack(x ,time_steps,1)


# TODO: add sentinel below, now we do not add the sentinel at the end anymore,
# we should add it between the padded sequence of each question and the question content
def coattention_encoder(D, Q, hyperparameters):
    # D[i] = document i in the batch, Q[i] = question i in the batch
    L = tf.matmul(D, Q, transpose_a = True)
    A_Q = tf.nn.softmax(L)
    A_D = tf.nn.softmax(tf.transpose(L))
    C_Q = tf.matmul(D, A_Q)
    concat_1 = tf.concat([Q, C_Q], 0)
    C_D = tf.matmul(concat_1, A_D)
    concat_2 = tf.concat([D, C_D], 0)
    BiLSTM_outputs, BiLSTM_final_state = dynamic_bilstm(concat_2, hyperparameters)
    # Make output format agree with the other components!
    return BiLSTM_outputs
 
def encoder(document, question, hyperparameters):
    D, Q = doc_que_encoder(document, question, hyperparameters)
    return coattention_encoder(D, Q, hyperparameters)
