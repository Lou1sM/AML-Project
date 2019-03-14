import numpy as np
import tensorflow as tf

# Note: can't use CudNN as it is not yet adapated (in TF)
# to take batches of variable sizes (NVidia recently added this feature to their ML API ~ 1 month ago),
# meanwhile, the classic tf.nn.dynamic_rnn supports adding a vector of batch_size elements, each
# describing the length of the documents/questions to process.

# Provide hyperparameters to functions below as dictionary with keys "hidden_size", "keep_prob", "batch_size"

def build_lstm_cell(hidden_size = 200, keep_prob = 1, batch_size = 10): 
    # lstm = tf.contrib.rnn.BasicLSTMCell(hidden_size) # deprecated
    lstm = tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell', num_units = hidden_size)
    cell = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob = keep_prob)
    initial_state = cell.zero_state(batch_size, tf.float32)
    return initial_state, cell


def dynamic_lstm(embed, sequence_lengths, hyperparameters):
    hidden_size = hyperparameters.hidden_size
    keep_prob = hyperparameters.keep_prob
    batch_size = hyperparameters.batch_size
    initial_state, cell = build_lstm_cell(hidden_size, keep_prob, batch_size)
    embed = tf.cast(embed,tf.float32)
    lstm_outputs, final_state = tf.nn.dynamic_rnn(cell = cell, inputs = embed, sequence_length = sequence_lengths, initial_state = initial_state)
    return lstm_outputs, final_state


def dynamic_bilstm(embed, sequence_lengths, hyperparameters):
    hidden_size = hyperparameters.hidden_size
    keep_prob = hyperparameters.keep_prob
    batch_size = hyperparameters.batch_size
    initial_fw_state, fw_cell = build_lstm_cell(hidden_size, keep_prob, batch_size)
    initial_bw_state, bw_cell = build_lstm_cell(hidden_size, keep_prob, batch_size)
    embed = tf.cast(embed,tf.float32)
    lstm_outputs, final_fw_state, final_bw_state = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                                                        cells_fw = [fw_cell], 
                                                        cells_bw = [bw_cell], 
                                                        inputs = embed, 
                                                        sequence_length = sequence_lengths,
                                                        initial_states_fw = [initial_fw_state],
                                                        initial_states_bw = [initial_bw_state],
                                                    )
    return lstm_outputs, final_fw_state, final_bw_state
 

def doc_que_encoder(document_columns, question_columns, documents_lengths, questions_lengths, hyperparameters):
    # Use batch_size from hyperparameters, dropout, num_cells
    # Data needs to come padded, also need the length 
    hidden_size = hyperparameters.hidden_size
    with tf.variable_scope('lstm') as scope:
        document_enc, final_state_doc = dynamic_lstm(document_columns, documents_lengths, hyperparameters)
        scope.reuse_variables()
        que_lstm_outputs, final_state_que = dynamic_lstm(question_columns, questions_lengths, hyperparameters)
    with tf.variable_scope('tanhlayer') as scope:
        linear_model = tf.layers.Dense(units = hidden_size)
        question_enc = tf.math.tanh(linear_model(que_lstm_outputs))
 
    return document_enc, question_enc

# Once we agree on shapes, use code below to write tensor as vector
# of length timesteps, with 2D tensors as elements:
# x=tf.placeholder("float",[None,time_steps,n_input])
# input=tf.unstack(x ,time_steps,1)


# TODO: add sentinel below, now we do not add the sentinel at the end anymore,
# we should add it between the padded sequence of each question and the question content
def coattention_encoder(D, Q, documents_lengths, questions_lengths, hyperparameters):
    # D[i] = document i in the batch, Q[i] = question i in the batch
    L = tf.matmul(D, tf.transpose(Q, perm = [0,2,1]))

    A_Q = tf.nn.softmax(L)
    A_D = tf.nn.softmax(tf.transpose(L, perm = [0,2,1]))
    C_Q = tf.matmul(tf.transpose(D, perm = [0,2,1]), A_Q)

    concat_1 = tf.concat([tf.transpose(Q, perm = [0,2,1]), C_Q], 1)
    C_D = tf.matmul(concat_1, A_D)

    concat_2 = tf.concat([tf.transpose(D, perm = [0,2,1]), C_D], 1)
    concat_2 = tf.transpose(concat_2, perm = [0,2,1])
    BiLSTM_outputs, BiLSTM_final_fw_state, BiLSTM_final_bw_state = dynamic_bilstm(concat_2, documents_lengths, hyperparameters)

    return BiLSTM_outputs
 
 
def encoder(document, question, documents_lengths, questions_lengths, hyperparameters):
    D, Q = doc_que_encoder(document, question, documents_lengths, questions_lengths, hyperparameters)
    return coattention_encoder(D, Q, documents_lengths, questions_lengths, hyperparameters)
