import json
import numpy as np
import nltk 
import datetime
import argparse
import string
import os

version = 2
if version == 1:
	dev_json_filename = 'data/squad/dev-v1.1.json'
elif version == 2:
	dev_json_filename = 'data/squad/dev-v2.0.json'

default_predicted_dev_json_filename = 'predictions_epoch_2.json'
default_additional_text = ''

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", default=default_predicted_dev_json_filename, type=str, help="Json predictions file name")
ARGS = parser.parse_args()

file_name = ARGS.file_name

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

codalab_evaluation_data = {}
codalab_json_filename = "codalab_" + file_name

json_file_path = os.path.join("", codalab_json_filename)
jsonFile = open(json_file_path, "w")

pct = set(string.punctuation)

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
		# if False:
		if version == 2 and pred_start >= len(tokenized_pred_paragraph):
			codalab_evaluation_data[pred_qid] = ""
			# print("HERE")
		else:

			pred_answer = tokenized_pred_paragraph[pred_start : pred_end + 1]
			correct_answers = qid_to_answers[pred_qid]

			pred_string = ""
			for token in pred_answer:
				if token == ',' or token == '.' or token ==';' or token == ':':
					token_space = token
				else:
					token_space = " " + token
				pred_string = pred_string + token_space

			pred_string = pred_string[1:]

			codalab_evaluation_data[pred_qid] = pred_string

			out_file.write("\n_______________________________________________________\n")

			out_file.write("Question ID: " + str(pred_qid) + "\n")
			out_file.write("Paragraph: " + pred_paragraph + "\n")
			out_file.write("Question: " + pred_question + "\n")

			out_file.write("Correct answers: " + str(correct_answers) + "\n")
			out_file.write("Model prediction: " + str(pred_answer) + "\n")

			pred_is_correct = False
			new_pred_answer = []
			for token in pred_answer:
					if token not in pct and token.lower() != "the" and token.lower() != "a" and token.lower!="an":
						new_pred_answer.append(token)

			for correct_answer in correct_answers:
				tokenized_correct_answer = nltk.word_tokenize(correct_answer)
				new_tokenized_correct_answer = []
				for token in tokenized_correct_answer:
					if token not in pct and token.lower() != "the" and token.lower() != "a" and token.lower!="an":
						new_tokenized_correct_answer.append(token)
				if(new_pred_answer == new_tokenized_correct_answer):
					pred_is_correct = True
					break
			if pred_is_correct:
				correct_predictions += 1.0
				out_file.write("Verdict: Correct\n")
			else:
				out_file.write("Verdict: Wrong\n")

			out_file.write("\n_______________________________________________________\n")
			out_file.flush()

	if version == 2:
		answer = list(map(lambda x: x['paragraphs'], actual_answers_data['data']))
		answer = [item for sublist in answer for item in sublist]

		answer = list(map(lambda x: x['qas'], answer))
		answer = [item for sublist in answer for item in sublist]

		answer_ids = set(list(map(lambda x: x['id'], answer)))

		for ID in answer_ids:
			if not(ID in codalab_evaluation_data):
				codalab_evaluation_data[ID] = ""

		for ID in codalab_evaluation_data:
			if not(ID in answer_ids):
				codalab_evaluation_data.pop(ID, None)


	print("EM SCORE: ", correct_predictions / total_count)
	out_file.write("Correct count: " + str(correct_predictions) + "\n")
	out_file.write("Total count: " + str(total_count) + "\n")
	out_file.write("EM SCORE: " + str(correct_predictions / total_count))
	out_file.write("\n")
	out_file.close()

	json.dump(codalab_evaluation_data, jsonFile)
	jsonFile.close()




process_actual_answers(predicted_dev_json_filename = file_name)
