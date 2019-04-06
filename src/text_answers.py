import json
import numpy as np
import nltk 
import datetime

dev_json_filename = 'data/squad/dev-v1.1.json'
default_predicted_dev_json_filename = 'predictions_epoch_2.json'

default_additional_text = ''

time_now = datetime.datetime.now()
out_file_name = "compare_predicted" + str(time_now) + ".txt"
out_file_name = out_file_name.replace(':', '-').replace(' ', '_')
out_file = open(out_file_name, "w")

def load_data(filename):
    with open(filename, encoding='utf-8') as data_file:
        data = json.loads(data_file.read())
    return data

total_count = 0.1
correct_predictions = 0.1

def process_actual_answers(predicted_dev_json_filename = default_predicted_dev_json_filename, additional_text = default_additional_text):
    global total_count
    global correct_predictions

    actual_answers_data = load_data(dev_json_filename)
    qid_to_answers = {}
    qid_to_question = {}
    qid_to_paragraph = {}

    answer = list(map(lambda x: x['paragraphs'], actual_answers_data['data']))
    answer = [item for sublist in answer for item in sublist]

    answer = list(map(lambda x: (x['context'], x['qas']), answer))
    answer = [(item,text) for (text, sublist) in answer for item in sublist]

    for dp in answer:
        qid = dp[0]['id']
        qid_to_answers[qid] = list(map(lambda x: x['text'], dp[0]['answers']))
        qid_to_question[qid] = dp[0]['question']
        qid_to_paragraph[qid] = dp[1]

    predicted_answers_data = load_data(predicted_dev_json_filename)
    epoch = predicted_answers_data['epoch']
    predictions = predicted_answers_data['pred']
    out_file.write("Epoch: " + str(epoch) + "\n")
    out_file.write("Extra data: " + str(additional_text) + "\n\n\n")

    for pred in predictions:
        total_count += 1.0

        pred_qid = pred['id']
        pred_start = pred['start']
        pred_end = pred['end']

        pred_paragraph = qid_to_paragraph[pred_qid]
        pred_question = qid_to_question[pred_qid]
        tokenized_pred_paragraph = nltk.word_tokenize(pred_paragraph)

        pred_answer = tokenized_pred_paragraph[pred_start : pred_end + 1]
        correct_answers = qid_to_answers[pred_qid]

        out_file.write("\n_______________________________________________________\n")

        out_file.write("Question ID: " + str(pred_qid) + "\n")
        out_file.write("Paragraph: " + pred_paragraph + "\n")
        out_file.write("Question: " + pred_question + "\n")

        out_file.write("Correct answers: " + str(correct_answers) + "\n")
        out_file.write("Model prediction: " + str(pred_answer) + "\n")

        out_file.write("\n_______________________________________________________\n")
        out_file.flush()

        ok = 0
        for correct_answer in correct_answers:
        	tokenized_correct_answer = nltk.word_tokenize(correct_answer)
        	if(tokenized_correct_answer == pred_answer):
        		ok = 1

        if(ok == 1):
        	correct_predictions += 1.0


    print("EM SCORE: ", correct_predictions / total_count)
    out_file.write("Correct count: " + str(correct_predictions) + "\n")
    out_file.write("Total count: " + str(total_count) + "\n")
    out_file.write("EM SCORE: " + str(correct_predictions / total_count))
    out_file.write("\n")
    out_file.close()



process_actual_answers()
