import numpy as np
import tensorflow as tf
import json
import itertools
import time
import nltk 
import pickle 
import random 

filename = 'data/embedding/glove.840B.300d.txt'
train_json_filename = 'data/squad/train-v1.1.json'
test_json_filename = 'data/squad/dev-v1.1.json'
time1 = time.time()
gloveDimension = 300
q_length = 60
d_length = 766
 
def load_glove(filename):
	count = 0
	embedding = {}
	file = open(filename,'r')
	for line in file.readlines():
		row = line.strip().split(' ')
		embedding[row[0]] = list(map(float, row[1:]))
	print('Loaded GloVe!')
	file.close()
	return embedding

# (embedding, index) = load_glove(filename)
# a = tf.nn.embedding_lookup(embedding, index['?'])
# sess = tf.Session()
# print(sess.run(a))

def load_train_data(filename):

	with open(filename, encoding='utf-8') as data_file:
		data = json.loads(data_file.read())
	return data

def save_titles():
	#embedding,index = load_glove(filename)
	data = load_train_data(json_filename)
	answer = map(lambda x: x['title'], data['data'])
	return list(answer)

def process_squad(x, multiple_answers):
	global max_count
	context = nltk.word_tokenize(x['context'])
	qas = x['qas']
	answers = list(map(lambda x: x['answers'], qas))
	answer_interval = []
	for i in range(len(answers)):
		if(multiple_answers):
			intervals = []
			for j in range(len(answers[i])):
				answer = answers[i][j]
				len_answer = len(nltk.word_tokenize(answer['text']))
				answer_character_position = answer['answer_start']
				answer = len(nltk.word_tokenize(x['context'][:answer_character_position]))
				intervals.append([answer, answer+len_answer-1])
			answer_interval.append(intervals)
		else:		
			answer = answers[i][0]
			len_answer = len(nltk.word_tokenize(answer['text']))
			answer_character_position = answer['answer_start']
			answer = len(nltk.word_tokenize(x['context'][:answer_character_position]))
			answer_interval.append([answer, answer+len_answer-1])
	question = list(map(lambda x: nltk.word_tokenize(x['question']), qas))
	ids = list(map(lambda x: x['id'], qas))
	qa = list(zip(question, answer_interval, ids))
	product = [context, qa]
	return product

def apply_embd(dic, context): 

	document = list(map(lambda x: dic[x.lower()] if x.lower() in dic else [0.0]*gloveDimension, context[0]))
	answers = list(map(lambda x: [list(map(lambda y: dic[y.lower()] if y.lower() in dic else [0.0]*gloveDimension, x[0])),x[1], x[2]], context[1]))
	answers = [[document, x[0], x[1], x[2]] for x in answers]

	return answers

def save_embeddings(type_of_embeddings):
	multiple_answers = False
	json_filename = train_json_filename

	if(type_of_embeddings == 'padded_test_data' or type_of_embeddings == 'unpadded_test_data'):
		multiple_answers = True
		json_filename = test_json_filename

	padded_data = False
	if(type_of_embeddings == 'padded_test_data' or type_of_embeddings == 'padded_train_data'):
		padded_data = True

	embedding = load_glove(filename)
	data = load_train_data(json_filename)
	answer = map(lambda x: x['paragraphs'], data['data'])
	answer = list(answer)
	answer = [item for sublist in answer for item in sublist]
	data = list(map(lambda x: process_squad(x, multiple_answers), answer))
	process_data = list(map(lambda x: apply_embd(embedding, x), data))
	data_array = [item for sublist in process_data for item in sublist]
	data_array = list(filter(lambda x: len(x[0]) < 600, data_array))	
	random.shuffle(data_array)
	if(padded_data):
		for j in range(0,len(data_array), 640):
			documents = []
			questions = []
			answers = []
			ids = []
			lengths_doc = []
			lengths_que = []
			for i in range(min(640, len(data_array)-j)):
				doc = list.copy(data_array[j+i][0])
				que = list.copy(data_array[j+i][1])
				pad = gloveDimension * [0.0]
				lengths_doc.append(len(doc))
				lengths_que.append(len(que))
				doc.extend([pad] * (d_length - len(doc)))
				que.extend([pad] * (q_length - len(que)))
				ans = data_array[j+i][2]
				q_id = data_array[j+i][3]
				documents.append(doc)
				questions.append(que)
				answers.append(ans)
				ids.append(q_id)
			documents = np.asarray(documents)
			print('doc done')
			questions = np.asarray(questions)
			print('que done')
			answers = np.asarray(answers)
			print('ans done')
			lengths_doc = np.asarray(lengths_doc)
			lengths_que = np.asarray(lengths_que)
			output_data_array = [[documents, questions, answers, ids], [lengths_doc, lengths_que]]
			np.savez_compressed('../../../shared/data/batched_data/'+type_of_embeddings+str(j/640), output_data_array)
			print("iteration %d", j)
	else:
		documents = list(map (lambda x: np.array(x[0]), data_array))
		questions = list(map (lambda x: np.array(x[1]), data_array))
		answers = list(map (lambda x: np.array(x[2]), data_array))
		data_array = [documents,questions,answers] 


# the types are 'padded_train_data', 'unpadded_train_data', 'padded_test_data', 'unpadded_test_data'
# the padded version also contain the lenghts of the original data points 
type_of_embeddings = 'padded_train_data'
save_embeddings(type_of_embeddings)