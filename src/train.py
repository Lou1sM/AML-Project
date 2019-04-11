import tensorflow as tf
import highway_max_out
import encoder
import ciprian_data_prep_script
import tfrecord_converter
import argparse
import time
from datetime import datetime
import os
import utils
import random
from tensorflow.python.client import timeline
from tensorflow.python import debug as tf_debug
import json
import log_reader


start_time = time.time()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('tensorflow').setLevel(50)

prob = tf.placeholder_with_default(0.7, shape=())
max_doc_len = tf.placeholder_with_default(600, shape=())
max_que_len = tf.placeholder_with_default(60, shape=())
parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", default=200, type=int, help="Number of epochs to train for")
parser.add_argument("--test", "-t", default=False, action="store_true", help="Whether to run in test mode")
parser.add_argument("--batch_size", default=64, type=int, help="Size of each training batch")
parser.add_argument("--dataset", choices=["SQuAD"],default="SQuAD", type=str, help="Dataset to train and evaluate on")
parser.add_argument("--hidden_size", default=200, type=int, help="Size of the hidden state")
parser.add_argument("--keep_prob", default=prob, type=float, help="Keep probability for question and document encodings.")
parser.add_argument("--max_doc_len", default=max_doc_len, type=int, help="The maximum length of a padded document.")
parser.add_argument("--max_que_len", default=max_que_len, type=int, help="The maximum length of a padded question.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--short_test", "-s", default=False, action="store_true", help="Whether to run in short test mode")
parser.add_argument("--pool_size", default=16, type=int, help="Number of units to pool over in HMN sub-network")
parser.add_argument("--tfdbg", default=False, action="store_true", help="Whether to enter tf debugger")
parser.add_argument("--restore", default=None, type=str, help="File path for the checkpoint to restore from. If None then don't restore.")
parser.add_argument("--log_folder", default=False, action="store_true", help="Whether to generate a folder with experimental results.")
parser.add_argument("--num_iterations_hmn", default=4, type=int, help="The number of HMN decoding loops")

parser.add_argument("--padding_mask", default=True, action="store_true", help="Whether to apply padding masks.")
parser.add_argument("--converge", default=False, action="store_true", help="Whether to stop iteration upon convergence.")

parser.add_argument("--bi_lstm_dropout", default=False, action="store_true", help="Whether to use bi-LSTM dropout.")
parser.add_argument("--bi_lstm_encoding_dropout", default=False, action="store_true", help="Whether to use bi-LSTM dropout.")
parser.add_argument("--doc_lstm_dropout", default=False, action="store_true", help="Whether to use dropout in the document lstm.")
parser.add_argument("--que_lstm_dropout", default=False, action="store_true", help="Whether to use dropout in the question lstm.")
parser.add_argument("--que_encoding_dropout", default=False, action="store_true", help="Whether to apply dropout to question encodings.")
parser.add_argument("--doc_encoding_dropout", default=False, action="store_true", help="Whether to apply dropout to document encodings.")
parser.add_argument("--squad2_vector", default=False, action="store_true", help="Whether to assume SQuAD 2 operation and add learnable vector.")
parser.add_argument("--squad2_lstm", default=False, action="store_true", help="Whether to assume SQuAD 2 operation and add LSTM.")
parser.add_argument("--exp_name", default="", help="Name of current experiment, in form 'a.b'")

ARGS = parser.parse_args()

if ARGS.exp_name == "" and ARGS.test == False:
    print("WARNING: no experiment name specified")

if ARGS.test:
    print("Running in test mode")
    ARGS.num_iterations_hmn=1
    ARGS.hidden_size=10
    ARGS.pool_size=1
    ARGS.batch_size=1
    ARGS.num_epochs=3

with tf.variable_scope("data_prep"):
    input_d_vecs, input_q_vecs, ground_truth_labels, documents_lengths, questions_lengths, _, _ = ciprian_data_prep_script.get_data("train")
    input_d_vecs_validation, input_q_vecs_validation, ground_truth_labels_validation, documents_lengths_validation, questions_lengths_validation, questions_ids_validation, all_answers_validation = ciprian_data_prep_script.get_data("test")
    #print("In train.py: get_data finished.")

    start_l = list(map(lambda x: x[0], ground_truth_labels))
    end_l = list(map(lambda x: x[1], ground_truth_labels))

    start_l_validation = list(map(lambda x: x[0], ground_truth_labels_validation))
    end_l_validation = list(map(lambda x: x[1], ground_truth_labels_validation))

    d = tf.placeholder(tf.float64, [ARGS.batch_size, None, len(input_d_vecs[0][0])])
    q = tf.placeholder(tf.float64, [ARGS.batch_size, None, len(input_q_vecs[0][0])])
    starting_labels = tf.placeholder(tf.int64, [ARGS.batch_size])
    ending_labels = tf.placeholder(tf.int64, [ARGS.batch_size])
    doc_l = tf.placeholder(tf.int64, [ARGS.batch_size])
    que_l = tf.placeholder(tf.int64, [ARGS.batch_size])


with tf.variable_scope("performance_metrics"):
    em_score_log = tf.placeholder(tf.float32, ())
    f1_score_log = tf.placeholder(tf.float32, ())
    tf.summary.scalar('em_score', em_score_log)
    tf.summary.scalar('f1_score', f1_score_log)

with tf.variable_scope("parameters"):
    tf.summary.scalar('batch_size', tf.constant(ARGS.batch_size))
    tf.summary.scalar('hidden_size', tf.constant(ARGS.hidden_size))
    tf.summary.scalar('pool_size', tf.constant(ARGS.pool_size))
    tf.summary.scalar('keep_prob', ARGS.keep_prob)
    tf.summary.scalar('learning_rate', tf.constant(ARGS.learning_rate))

with tf.variable_scope("encoder"):
    L, encoded = encoder.encoder(
        document=d,
        question=q,
        documents_lengths=doc_l,
        questions_lengths=que_l,
        hyperparameters=ARGS
    )

    # Create single nodes for labels, data version
    #start_labels = tf.one_hot(a[:,0], 600)
    #end_labels = tf.one_hot(a[:,1], 600)

    # if we are running for SQuAD 2, accommodate IMPOSSIBlE encoding
    '''
    if (ARGS.squad2_vector or ARGS.squad2_lstm):
        max_doc_len += 1
    '''
    # Create single nodes for labels, feed_dict version
    start_labels = tf.one_hot(starting_labels, max_doc_len)
    end_labels = tf.one_hot(ending_labels, max_doc_len)

with tf.variable_scope("decoder"):
    # Calculate padding mask
    if ARGS.padding_mask:
        after_padding_mark = tf.one_hot(doc_l, max_doc_len)
        padding_mask = tf.math.cumsum(after_padding_mark, axis=1)
        min_float_at_padding = tf.multiply(padding_mask, tf.cast(0.5*tf.float32.min, tf.float32))

    # Initialize variables if convergence is enabled
    if ARGS.converge:
        # Persistent loss mask that remembers whether convergence has already taken place
        loss_mask = tf.constant(True, shape=[ARGS.batch_size])
        s_prev = tf.constant(-1, dtype=tf.int32, shape=[ARGS.batch_size])
        e_prev = tf.constant(-1, dtype=tf.int32, shape=[ARGS.batch_size])

    # Static tensor to be re-used in main loop iterations
    batch_indices = tf.range(start=0, limit=ARGS.batch_size, dtype=tf.int32)

    # Create and initialize decoding LSTM
    decoding_lstm = tf.contrib.rnn.LSTMCell(num_units=ARGS.hidden_size)
    h = decoding_lstm.zero_state(ARGS.batch_size, dtype=tf.float32)

    # First guess for start- and end-points
    u_s = tf.zeros([ARGS.batch_size, 2 * ARGS.hidden_size], tf.float32)  # Dummy guess start point
    u_e = tf.zeros([ARGS.batch_size, 2 * ARGS.hidden_size], tf.float32)  # Dummy guess end point

    if ARGS.converge:
        total_iteration_count = tf.zeros([1])

    with tf.variable_scope("decoding_loop"):
        for i in range(ARGS.num_iterations_hmn):
            # LSTM input is concatenation of previous guesses
            usue = tf.concat([u_s, u_e], axis=1)
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

            if ARGS.padding_mask:
                alphas = tf.add(alphas, min_float_at_padding)
            s = tf.argmax(alphas, axis=1, output_type=tf.int32)
            s_encoding_indices = tf.transpose(tf.stack([batch_indices, s]))
            u_s = tf.gather_nd(encoded, s_encoding_indices)
 
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

            if ARGS.padding_mask:
                betas = tf.add(betas, min_float_at_padding)
            e = tf.argmax(betas, axis=1, output_type=tf.int32)
            e_encoding_indices = tf.transpose(tf.stack([batch_indices, e]))
            u_e = tf.gather_nd(encoded, e_encoding_indices)

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

            iteration_loss = s_loss + e_loss
            if ARGS.converge:
                with tf.variable_scope("iteration_" + str(i) + "_loss"):
                    s_mask = tf.equal(s, s_prev)
                    e_mask = tf.equal(e, e_prev)
                    output_same = tf.logical_and(s_mask, e_mask)
                    s_prev = s
                    e_prev = e
                    loss_mask = tf.logical_and(loss_mask, tf.logical_not(output_same))
                    masked_iteration_loss = tf.multiply(iteration_loss, tf.cast(loss_mask, tf.float32))
                    total_iteration_count = total_iteration_count + tf.reduce_sum(tf.cast(loss_mask, tf.float32))
                    tf.summary.scalar('loss', tf.reduce_mean(iteration_loss))
                    tf.summary.scalar('masked_it_loss_summary', tf.reduce_mean(masked_iteration_loss))

                #loss = iteration_loss if i == 0 else loss + iteration_loss
                with tf.control_dependencies([tf.assert_greater_equal(iteration_loss, masked_iteration_loss)]):
                    loss = masked_iteration_loss if i == 0 else loss + masked_iteration_loss

                final_s = s if i == 0 else final_s + tf.multiply(tf.cast(loss_mask, tf.int32), s - final_s)
                final_e = e if i == 0 else final_e + tf.multiply(tf.cast(loss_mask, tf.int32), e - final_e)

            else:
                loss = iteration_loss if i == 0 else loss + iteration_loss
if not ARGS.converge:
    final_s = s
    final_e = e
    mean_loss = tf.reduce_mean(loss)
else:
    mean_loss_per_iteration_per_datapoint = tf.divide(tf.reduce_sum(loss), total_iteration_count)
    # normalize to make it comparable to non-convergence variant
    mean_loss = tf.multiply(mean_loss_per_iteration_per_datapoint, ARGS.num_iterations_hmn)
    mean_loss = tf.reduce_mean(mean_loss) # convert shape from (1,) to ()
tf.summary.scalar('total_loss', mean_loss)
optimizer = tf.train.AdamOptimizer(ARGS.learning_rate)
train_step = optimizer.minimize(mean_loss)

#time_now = datetime.datetime.now()
datetime_stamp = str(datetime.now()).split()[0][5:] + '_'+str(datetime.now().time()).split()[0][:-7]
exp_dir_name = datetime_stamp if ARGS.exp_name == "" else ARGS.exp_name +'@'+datetime_stamp

logFolder = ""
if(ARGS.log_folder):
    #log_dir = os.path.join("/home/shared/logs", str(time_now).replace(':', '-').replace(' ', '_'))
    log_dir = os.path.join("/home/shared/logs", exp_dir_name)
    print(log_dir)
    orig_mask = os.umask(0)
    os.makedirs(log_dir, mode=0o777)
    os.umask(orig_mask)
    #logFolder.replace(':', '-').replace(' ', '_')
    #tf.gfile.MkDir(logFolder)
else:
    log_dir = ""

#tf.gfile.MkDir(log_dir)

# For Tensorboard
merged = tf.summary.merge_all()
#summaryDirectory = "/home/shared/summaries/start_" + str(datetime.datetime.now())
summary_dir = os.path.join(log_dir, "summaries")
if not os.path.isdir(summary_dir):
    orig_mask = os.umask(0)
    os.makedirs(summary_dir, mode=0o777)
    os.umask(orig_mask)

checkpoint_dir = os.path.join(log_dir, "checkpoints")
if not os.path.isdir(checkpoint_dir):
    orig_mask = os.umask(0)
    os.makedirs(checkpoint_dir, mode=0o777)
    os.umask(orig_mask)



#tf.gfile.MkDir(summaryDirectory)
#dateSummary = str(datetime.datetime.now())
#dateSummary = dateSummary.replace('.', '_').replace(':', '-').replace(' ', '_')
#summaryDirectory += dateSummary
#tf.gfile.MkDir(summaryDirectory)
#writer = tf.summary.FileWriter(summaryDirectory)
writer = tf.summary.FileWriter(summary_dir)
writer.add_graph(tf.get_default_graph())

saver = tf.train.Saver(max_to_keep = 100)

#logDirectory = logFolder + "logs/"
#tf.gfile.MkDir(logDirectory)
#file_name = "log" + str(time_now) + ".txt"
#file_name = file_name.replace(':', '-').replace(' ', '_')
#file = open(logDirectory + file_name, "w")
file = open(os.path.join(log_dir, "log.txt"), "w")
log_file_path = os.path.join(log_dir, "log.txt")
os.chmod(log_file_path, 0o777)
file = open(log_file_path, "w")


#fileEM_name = "logEM" + str(time_now) + ".txt"
#fileEM_name = fileEM_name.replace(':', '-').replace(' ', '_')
#fileEM = open(logDirectory + fileEM_name, "w")
logEM_file_path = os.path.join(log_dir, "logEM.txt")
f = open(logEM_file_path, "w").close()
os.chmod(logEM_file_path, 0o777)
fileEM = open(logEM_file_path,"w")
fileEM.write("Hyperparameters:" + str(ARGS))

param_file_path = os.path.join(log_dir, "params_used.txt")
with open(param_file_path, "w") as param_file:
    for key in sorted(vars(ARGS).keys()):
        param_file.write(str(key) + ": " + str(vars(ARGS)[key]) + "\n")
os.chmod(param_file_path, 0o777)


if ARGS.tfdbg:
    chosen_session = tf_debug.LocalCLIDebugWrapperSession(tf.Session())
else:
    chosen_session = tf.Session()
with chosen_session as sess:
    if ARGS.restore == None:
        sess.run(tf.global_variables_initializer())
    else:
        restore_path = saver.restore(sess, os.path.join(checkpoint_dir, ARGS.restore))
        print("Restoring from checkpoint at", restore_path)
    train_start_time = time.time()
    print("Graph-build time: ", utils.time_format(train_start_time - start_time))

    dataset_length = len(input_d_vecs)
    num_batchs = dataset_length // ARGS.batch_size 

    dataset_length_validation = len(input_d_vecs_validation)
    num_batchs_validation = dataset_length_validation // ARGS.batch_size

    best_em_score = 0.0
    best_avg_f1 = 0.0
    global_batch_num = 0
    new_avg_f1 = 0
    new_em_score = 0
    for epoch in range(ARGS.num_epochs):

        json_predictions = {}
        json_predictions['epoch'] = int(epoch)
        json_predictions['pred'] = []

        jsonFileName = "predictions_epoch_" + str(epoch) + ".json"
        #jsonFile = open(logDirectory + jsonFileName, "w")
        json_file_path = os.path.join(log_dir, jsonFileName)
        jsonFile = open(json_file_path, "w")

        print("\nEpoch:", epoch)
        file.write("\nEpoch: " + str(epoch) + "\n")

        batch_size = ARGS.batch_size
        shuffling = list(zip(input_d_vecs, input_q_vecs, start_l, end_l, documents_lengths, questions_lengths))
        random.shuffle(shuffling) 
        input_d_vecs, input_q_vecs, start_l, end_l, documents_lengths, questions_lengths = zip(*shuffling)

        prev_time = time.time()
        epoch_loss = 0
        for dp_index in range(0, dataset_length, batch_size):
            if dp_index + batch_size > dataset_length:
                break
            #batch_time = time.time()

            doc = list.copy(list(input_d_vecs[dp_index:dp_index + batch_size]))
            que = list.copy(list(input_q_vecs[dp_index:dp_index + batch_size]))
            doc_len = documents_lengths[dp_index: dp_index + batch_size]
            que_len = questions_lengths[dp_index: dp_index + batch_size]
            batch_doc_len = max(doc_len)
            batch_que_len = max(que_len)
            doc = [elem[:batch_doc_len] for elem in doc]
            que = [elem[:batch_que_len] for elem in que]

            feed_dict = {
                max_doc_len: batch_doc_len,
                max_que_len: batch_que_len,
                d: doc,
                q: que,
                starting_labels: start_l[dp_index: dp_index + batch_size],
                ending_labels: end_l[dp_index: dp_index + batch_size],
                doc_l: doc_len,
                que_l: que_len,
                f1_score_log: new_avg_f1,
                em_score_log: new_em_score
                }

            '''
            invalid_batch = False
            for bi in range(0, batch_size):
                answer_end_index = feed_dict[ending_labels][bi]
                document_length = feed_dict[doc_l][bi]
                if not answer_end_index < document_length:
                    invalid_batch = True
                    break
            if invalid_batch:
                print("Batch with invalid datapoint encounterd, skipping...")
                continue
            '''

            if (dp_index//batch_size + 1)% 10 == 0: #or batch_size == dataset_length-batch_num:
                _, loss_val, summary_val = sess.run([train_step, mean_loss, merged], feed_dict=feed_dict)
                dp_time = (time.time() - prev_time)/(10.0*batch_size)
                prev_time = time.time()
                writer.add_summary(summary_val, global_batch_num)
                #loss_val = round(loss_val, 5)
                print("\tBatch: {}\tloss: {}\t Time per data point: {}".format((dp_index//batch_size)+1,loss_val, utils.time_format(dp_time)))
                epoch_loss += loss_val
            else:
                sess.run([train_step], feed_dict=feed_dict)

            global_batch_num += 1
            del(doc)
            del(que)
            del(doc_len)
            del(que_len)

            #print(time.time()-batch_time)
            if ARGS.test:
                break

        epoch_loss = 10*epoch_loss/(int(dataset_length/batch_size))
        total_count = 0.1
        exact_matches = 0.1
        num_just_start_right = 0.1
        running_f1 = 0.1

        total_epoch_val_loss = 0.0

        for dp_index_validation in range(0, dataset_length_validation, ARGS.batch_size):
            if(dp_index_validation % 500 == 0):
                print("Validation Batch: ", dp_index_validation, "\n")
                file.write("Validation Batch: " + str(dp_index_validation) + "\n")
                file.flush()
            if dp_index_validation + ARGS.batch_size >= len(input_d_vecs_validation):
                break

            doc = list.copy(list(input_d_vecs_validation[dp_index_validation:dp_index_validation + ARGS.batch_size]))
            que = list.copy(list(input_q_vecs_validation[dp_index_validation: dp_index_validation + ARGS.batch_size]))
            doc_len = documents_lengths_validation[dp_index_validation: dp_index_validation + ARGS.batch_size]
            que_len = questions_lengths_validation[dp_index_validation: dp_index_validation + ARGS.batch_size]
            batch_doc_len = max(doc_len)
            batch_que_len = max(que_len)
            doc = [elem[:batch_doc_len] for elem in doc]
            que = [elem[:batch_que_len] for elem in que]

            feed_dict_validation = {
                prob: 1,
                max_doc_len: batch_doc_len,
                max_que_len: batch_que_len,
                d: doc,
                q: que,
                starting_labels: start_l_validation[dp_index_validation: dp_index_validation + ARGS.batch_size],
                ending_labels: end_l_validation[dp_index_validation: dp_index_validation + ARGS.batch_size],
                doc_l: doc_len,
                que_l: que_len
            }

            loss_val_validation, start_predict_validation, end_predict_validation = sess.run([mean_loss, final_s, final_e], feed_dict = feed_dict_validation)
            start_correct_validation = start_l_validation[dp_index_validation: dp_index_validation + ARGS.batch_size]
            end_correct_validation = end_l_validation[dp_index_validation: dp_index_validation + ARGS.batch_size]

            total_epoch_val_loss += loss_val_validation
            for i in range(ARGS.batch_size):
                total_count += 1.0
                got_exact_match = False
                best_f1_dp = 0.0
                for ans in all_answers_validation[dp_index_validation + i]:
                    new_f1_dp = utils.compute_f1_from_indices(start_predict_validation[i], end_predict_validation[i], ans[0], ans[1])
                    if new_f1_dp > best_f1_dp:
                        best_f1_dp = new_f1_dp
                    if ans == [start_predict_validation[i], end_predict_validation[i]]:
                        got_exact_match = True
                    elif ans[0] == start_predict_validation[i]:
                        num_just_start_right += 1
                if got_exact_match:
                    exact_matches += 1
                running_f1 += best_f1_dp

                file.write("Question with ID: " + str(questions_ids_validation[dp_index_validation + i]))
                file.write("\n")
                file.write("Correct (start, end): " + str(all_answers_validation[dp_index_validation + i]))
                file.write("\n")
                file.write("Predicted (start, end): " + str((start_predict_validation[i], end_predict_validation[i])))
                file.write("\n")
                file.write("___________________________\n")
                file.flush()

                json_dp = {}
                json_dp["id"] = str(questions_ids_validation[dp_index_validation + i])
                json_dp["start"] = int(start_predict_validation[i])
                json_dp["end"] = int(end_predict_validation[i])
                json_predictions['pred'].append(json_dp)


            del(doc)
            del(que)
            del(doc_len)
            del(que_len)

            if ARGS.test:
                break
        total_epoch_val_loss = total_epoch_val_loss/(int(dataset_length_validation/batch_size))
        new_em_score = exact_matches / total_count
        new_avg_f1 = running_f1 / total_count
        frac_just_start_right = num_just_start_right / total_count
        if new_avg_f1 > best_avg_f1:
            best_avg_f1 = new_avg_f1

        print("New avg f1: %f Best avg f1: %f." % (new_avg_f1, best_avg_f1))
        json.dump(json_predictions, jsonFile)
        jsonFile.close()

        os.chmod(json_file_path, 0o777)
        if new_em_score > best_em_score:
            #save_path = saver.save(sess, "checkpoints/model.ckpt")
            #save_path = saver.save(sess, logFolder + "checkpoints/model{}.ckpt".format(epoch))
            save_path = saver.save(sess, os.path.join(checkpoint_dir, "model{}.ckpt".format(epoch)))
            print("EM score improved from %f to %f. Model saved in path: %s" % (best_em_score, new_em_score, save_path,))
            print("Epoch validation loss: %f" % total_epoch_val_loss)
            fileEM.write("\nEpoch number:" + str(epoch))
            fileEM.write("\n")
            fileEM.write("\nNew EM score:" + str(new_em_score) + " Best EM score: " + str(best_em_score) + ". Model saved in path: " + str(save_path))
            fileEM.write("\nFraction with just start prediction correct: {}".format(frac_just_start_right))
            fileEM.write("\nNew avg F1:" + str(new_avg_f1) + " Best avg f1: " + str(best_avg_f1) + ".")
            #fileEM.write("\n")
            fileEM.write("\nEpoch loss value:" + str(epoch_loss))
            fileEM.write("\nEpoch validation loss: %f" % total_epoch_val_loss)
            fileEM.write("\n\n")
            fileEM.flush()
            best_em_score = new_em_score 
        else:
            print("No improvement in EM score, not saving")
            fileEM.write("Epoch number:" + str(epoch))
            fileEM.write("\n")
            fileEM.write("\nNew EM score:" + str(new_em_score) + " Best EM score: " + str(best_em_score) + ". No improvement, model not saved.")
            fileEM.write("\nNew avg F1:" + str(new_avg_f1) + " Best avg f1: " + str(best_avg_f1) + ".")
            #fileEM.write("\n")
            fileEM.write("\nEpoch loss value:" + str(epoch_loss))
            fileEM.write("\nEpoch validation loss: %f" % total_epoch_val_loss)
            fileEM.write("\n\n")
            fileEM.flush()
 
train_end_time = time.time()

train_loss_scores, val_loss_scores = log_reader.get_train_val_scores(logEM_file_path)
loss_graph_file_path = os.path.join(log_dir, 'train_val_loss.png')
log_reader.plot_losses(train_losses=train_loss_scores, val_losses=val_loss_scores, filepath=loss_graph_file_path)
print("Train time", utils.time_format(train_end_time - train_start_time))
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
                    print(loss_val)
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
                                save_path = saver.save(sess, "/home/shared/checkpoints/model.ckpt")
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
