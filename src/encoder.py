import numpy as np
import tensorflow as tf
from utils import bias_variable, variable_summaries

# Note: can't use CudNN as it is not yet adapated (in TF)
# to take batches of variable sizes (NVidia recently added this feature to their ML API ~ 1 month ago),
# meanwhile, the classic tf.nn.dynamic_rnn supports adding a vector of batch_size elements, each
# describing the length of the documents/questions to process.

def build_lstm_cell(hidden_size = 200, keep_prob = 1, batch_size = 10, use_dropout = True):
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

def dynamic_lstm_with_hidden_size(embed, sequence_lengths, hyperparameters, hidden_size, use_dropout = True):
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
        document_enc, final_state_doc = dynamic_lstm(document_columns, documents_lengths, hyperparameters, use_dropout = hyperparameters.doc_lstm_dropout)
        if hyperparameters.doc_encoding_dropout:
            document_enc = tf.nn.dropout(document_enc, keep_prob=hyperparameters.keep_prob)
        # No dropout for when questions pass
        que_lstm_outputs, final_state_que = dynamic_lstm(question_columns, questions_lengths, hyperparameters, use_dropout = hyperparameters.que_lstm_dropout)
    with tf.variable_scope('tanhlayer') as scope:
        linear_model = tf.layers.Dense(units = hidden_size)
        question_enc = tf.math.tanh(linear_model(que_lstm_outputs))
        # add dropout after tanh
        if hyperparameters.que_encoding_dropout:
            question_enc = tf.nn.dropout(question_enc, keep_prob=hyperparameters.keep_prob)
 
    return document_enc, question_enc

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

    L = tf.matmul(D, tf.transpose(Q, perm = [0,2,1]))
    if hyperparameters.padding_mask:
        document_end_indices = tf.subtract(documents_lengths, 1)
        question_end_indices = tf.subtract(questions_lengths, 1)
        doc_words_mask = tf.math.cumsum(tf.one_hot(document_end_indices, hyperparameters.max_doc_len), axis=1, reverse=True)
        que_words_mask = tf.math.cumsum(tf.one_hot(question_end_indices, hyperparameters.max_que_len), axis=1, reverse=True)
        # add sentinels
        sentinel_mask = tf.ones([hyperparameters.batch_size, 1])
        doc_words_mask = tf.concat([doc_words_mask, sentinel_mask], axis=1)
        que_words_mask = tf.concat([que_words_mask, sentinel_mask], axis=1)
        words_mask = tf.matmul(tf.expand_dims(doc_words_mask, axis=2), tf.expand_dims(que_words_mask, axis=1))
        negative_padding_mask = tf.subtract(words_mask, 1)
        min_float_at_padding = tf.multiply(negative_padding_mask, tf.cast(-0.5*tf.float32.min, tf.float32))
        L = tf.add(L, min_float_at_padding)

    A_Q = tf.nn.softmax(L, axis=int(hyperparameters.softmax_axis), name="softmaxed_L")
    A_D = tf.nn.softmax(tf.transpose(L, perm = [0,2,1]), axis=int(hyperparameters.softmax_axis), name="softmaxed_L_transpose")
    C_Q = tf.matmul(tf.transpose(D, perm = [0,2,1]), A_Q)

    C_D_2 = tf.matmul(C_Q, A_D)
    C_Q_2 = tf.matmul(C_D_2, A_Q)
    #print('C_D_2', C_D_2.shape)
    #print('C_Q_2', C_Q_2.shape)
    concat_1 = tf.concat([tf.transpose(Q, perm = [0,2,1]), C_Q], 1)
    concat_1_1 = tf.concat([tf.transpose(Q, perm=[0,2,1]), C_Q, C_Q_2], 1)
    if int(hyperparameters.coattention) == 0:
        C_D = tf.matmul(tf.transpose(Q, perm=[0,2,1]), A_D)
    elif int(hyperparameters.coattention)== 1:
        C_D = tf.matmul(concat_1, A_D)
    elif int(hyperparameters.coattention)== 2:
        C_D = tf.matmul(concat_1_1, A_D)
    concat_2 = tf.concat([tf.transpose(D, perm = [0,2,1]), C_D], 1)
    concat_2 = tf.transpose(concat_2, perm = [0,2,1])
    concat_2 = concat_2[:, :-1, :]  # remove sentinels

    BiLSTM_outputs, BiLSTM_final_fw_state, BiLSTM_final_bw_state = dynamic_bilstm(concat_2, documents_lengths, hyperparameters)

    if hyperparameters.bi_lstm_encoding_dropout:
        BiLSTM_outputs = tf.nn.dropout(BiLSTM_outputs, keep_prob=hyperparameters.keep_prob)

    '''
    if (hyperparameters.squad2_vector or hyperparameters.squad2_lstm):
        with tf.name_scope("SQuAD_2"):
            if (hyperparameters.squad2_vector):
                impossible_encoding = bias_variable([2 * hyperparameters.hidden_size])
                variable_summaries(impossible_encoding)
                impossible_encoding = tf.expand_dims(tf.expand_dims(impossible_encoding, axis=0), axis=0)
                impossible_encoding = tf.tile(impossible_encoding, [hyperparameters.batch_size, 1, 1])
            elif (hyperparameters.squad2_lstm):
                encodings, final_state = dynamic_lstm_with_hidden_size(concat_2, documents_lengths, hyperparameters,
                                                                       2 * hyperparameters.hidden_size)
                impossible_encoding = encodings[:, -1]
                variable_summaries(impossible_encoding)
                impossible_encoding = tf.expand_dims(impossible_encoding, axis=1)
        BiLSTM_outputs = tf.concat([BiLSTM_outputs, impossible_encoding], axis=1)
'''
    return L, BiLSTM_outputs
 
 
def encoder(document, question, documents_lengths, questions_lengths, hyperparameters):
    with tf.variable_scope("doc_que_encoder"):
        D, Q = doc_que_encoder(document, question, documents_lengths, questions_lengths, hyperparameters)
    with tf.variable_scope("coattention_encoder"):
        L, encodings = coattention_encoder(D, Q, documents_lengths, questions_lengths, hyperparameters)
        return L, encodings



