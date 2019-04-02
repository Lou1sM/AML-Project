import numpy as np
import tensorflow as tf
from utils import bias_variable, variable_summaries

# Note: can't use CudNN as it is not yet adapated (in TF)
# to take batches of variable sizes (NVidia recently added this feature to their ML API ~ 1 month ago),
# meanwhile, the classic tf.nn.dynamic_rnn supports adding a vector of batch_size elements, each
# describing the length of the documents/questions to process.

# Provide hyperparameters to functions below as dictionary with keys "hidden_size", "keep_prob", "batch_size"

def build_lstm_cell(hidden_size = 200, keep_prob = 1, batch_size = 10, use_dropout = True):
    # lstm = tf.contrib.rnn.BasicLSTMCell(hidden_size) # deprecated
    lstm = tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell', num_units = hidden_size)
    if use_dropout:
        cell = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob = keep_prob)
    else:
        cell = lstm
    initial_state = cell.zero_state(batch_size, tf.float32)
    return initial_state, cell


def dynamic_lstm(embed, sequence_lengths, hyperparameters, use_dropout = True):
    hidden_size = hyperparameters.hidden_size
    keep_prob = hyperparameters.keep_prob
    batch_size = hyperparameters.batch_size
    initial_state, cell = build_lstm_cell(hidden_size, keep_prob, batch_size, use_dropout)
    embed = tf.cast(embed,tf.float32)
    lstm_outputs, final_state = tf.nn.dynamic_rnn(cell = cell, inputs = embed, sequence_length = sequence_lengths, initial_state = initial_state)
    return lstm_outputs, final_state


def dynamic_bilstm(embed, sequence_lengths, hyperparameters):
    hidden_size = hyperparameters.hidden_size
    keep_prob = hyperparameters.keep_prob
    batch_size = hyperparameters.batch_size
    if hyperparameters.bi_lstm_dropout:
        initial_fw_state, fw_cell = build_lstm_cell(hidden_size, keep_prob=keep_prob, batch_size=batch_size)
        initial_bw_state, bw_cell = build_lstm_cell(hidden_size, keep_prob=keep_prob, batch_size=batch_size)
    else:
        initial_fw_state, fw_cell = build_lstm_cell(hidden_size, keep_prob=1, batch_size=batch_size)
        initial_bw_state, bw_cell = build_lstm_cell(hidden_size, keep_prob=1, batch_size=batch_size)
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
    with tf.variable_scope('lstm', reuse = tf.AUTO_REUSE) as scope:
        document_enc, final_state_doc = dynamic_lstm(document_columns, documents_lengths, hyperparameters)
        # No dropout for when questions pass
        que_lstm_outputs, final_state_que = dynamic_lstm(question_columns, questions_lengths, hyperparameters, use_dropout=hyperparameters.q_lstm_dropout)
    with tf.variable_scope('tanhlayer') as scope:
        linear_model = tf.layers.Dense(units = hidden_size)
        question_enc = tf.math.tanh(linear_model(que_lstm_outputs))
        # add dropout after tanh
        if hyperparameters.q_tanh_dropout:
            question_enc = tf.nn.dropout(question_enc, keep_prob=hyperparameters.keep_prob)
 
    return document_enc, question_enc

# Once we agree on shapes, use code below to write tensor as vector
# of length timesteps, with 2D tensors as elements:
# x=tf.placeholder("float",[None,time_steps,n_input])
# input=tf.unstack(x ,time_steps,1)

def coattention_encoder(D, Q, documents_lengths, questions_lengths, hyperparameters):
    # D[i] = document i in the batch, Q[i] = question i in the batch
    with tf.name_scope("sentinels"):
        with tf.variable_scope("sentinel_d"):
            sentinel_d = bias_variable([hyperparameters.hidden_size])
            variable_summaries(sentinel_d)
        with tf.variable_scope("sentinel_q"):
            sentinel_q = bias_variable([hyperparameters.hidden_size])
            variable_summaries(sentinel_q)
        # append sentinels at the end of documents
        expanded_sentinel_d = tf.expand_dims(tf.expand_dims(sentinel_d, 0), 0)
        tiled_sentinel_d = tf.tile(expanded_sentinel_d, [hyperparameters.batch_size, 1, 1])
        D = tf.concat([D, tiled_sentinel_d], axis=1)
        # append sentinels at the end of questions
        expanded_sentinel_q = tf.expand_dims(tf.expand_dims(sentinel_q, 0), 0)
        tiled_sentinel_q = tf.tile(expanded_sentinel_q, [hyperparameters.batch_size, 1, 1])
        Q = tf.concat([Q, tiled_sentinel_q], axis=1)

    #print('D', D.shape)
    #print('Q', Q.shape)
    L = tf.matmul(D, tf.transpose(Q, perm = [0,2,1]))
    if hyperparameters.padding_mask:
        document_end_indices = tf.subtract(documents_lengths, 1)
        question_end_indices = tf.subtract(questions_lengths, 1)
        doc_words_mask = tf.math.cumsum(tf.one_hot(document_end_indices, 600), axis=1, reverse=True)
        que_words_mask = tf.math.cumsum(tf.one_hot(question_end_indices, 60), axis=1, reverse=True)
        # add sentinels
        sentinel_mask = tf.ones([hyperparameters.batch_size, 1])
        doc_words_mask = tf.concat([doc_words_mask, sentinel_mask], axis=1)
        que_words_mask = tf.concat([que_words_mask, sentinel_mask], axis=1)
        words_mask = tf.matmul(tf.expand_dims(doc_words_mask, axis=2), tf.expand_dims(que_words_mask, axis=1))
        negative_padding_mask = tf.subtract(words_mask, 1)
        min_float_at_padding = tf.multiply(negative_padding_mask, tf.cast(-0.5*tf.float32.min, tf.float32))
        L = tf.add(L, min_float_at_padding)

    #print('L', L.shape)
    A_Q = tf.nn.softmax(L, axis=1, name="softmaxed_L")
    A_D = tf.nn.softmax(tf.transpose(L, perm = [0,2,1]), axis=1, name="softmaxed_L_transpose")
    C_Q = tf.matmul(tf.transpose(D, perm = [0,2,1]), A_Q)

    concat_1 = tf.concat([tf.transpose(Q, perm = [0,2,1]), C_Q], 1)
    C_D = tf.matmul(concat_1, A_D)

    concat_2 = tf.concat([tf.transpose(D, perm = [0,2,1]), C_D], 1)
    concat_2 = tf.transpose(concat_2, perm = [0,2,1])
    concat_2 = concat_2[:, :-1, :]  # remove sentinels

    BiLSTM_outputs, BiLSTM_final_fw_state, BiLSTM_final_bw_state = dynamic_bilstm(concat_2, documents_lengths, hyperparameters)
    return L, BiLSTM_outputs
 
 
def encoder(document, question, documents_lengths, questions_lengths, hyperparameters):
    with tf.variable_scope("doc_que_encoder"):
        D, Q = doc_que_encoder(document, question, documents_lengths, questions_lengths, hyperparameters)
    with tf.variable_scope("coattention_encoder"):
        return coattention_encoder(D, Q, documents_lengths, questions_lengths, hyperparameters)



