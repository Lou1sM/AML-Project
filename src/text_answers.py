import json
import numpy as np
import nltk 
import datetime

dev_json_filename = 'data/squad/dev-v1.1.json'
default_predicted_dev_json_filename = 'predictions.json'

default_additional_text = ''

time_now = datetime.datetime.now()
out_file_name = "compare_predicted" + str(time_now) + ".txt"
out_file_name = out_file_name.replace(':', '-').replace(' ', '_')
out_file = open(out_file_name, "w")

def load_data(filename):
    with open(filename, encoding='utf-8') as data_file:
        data = json.loads(data_file.read())
    return data

count_questions = 0

def process_actual_answers(predicted_dev_json_filename = default_predicted_dev_json_filename, additional_text = default_additional_text):
    global count_questions
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
        count_questions += 1

    predicted_answers_data = load_data(predicted_dev_json_filename)
    epoch = predicted_answers_data['epoch']
    predictions = predicted_answers_data['pred']
    out_file.write("Epoch: " + str(epoch) + "\n")
    out_file.write("Extra data: " + str(additional_text) + "\n\n\n")

    for pred in predictions:
        pred_qid = pred['qid']
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

    out_file.close()



process_actual_answers()
