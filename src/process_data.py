import numpy as np
import tensorflow as tf
import json
import itertools
import time
import nltk 
import pickle 
import random

filename = 'data/embedding/glove.6B/glove.6B.300d.txt'
train_json_filename = 'data/squad/train-v1.1.json'
test_json_filename = 'data/squad/dev-v1.1.json'
time1 = time.time()
gloveDimension = 300
q_length = 60
d_length = 600
 
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
max_count = 0
def process_squad(x, multiple_answers):
	global max_count
	qas = x['qas']
	answers = list(map(lambda x: x['answers'], qas))
	answer_interval = []
	positions = []
	for i in range(len(answers)):
		if(multiple_answers):
			intervals = []
			for j in range(len(answers[i])):
				answerin = answers[i][j]
				text_answer = answerin['text']
				len_answer = len(nltk.word_tokenize(text_answer))
				answer_character_position = answerin['answer_start']
				text = x['context'][:answer_character_position]
				# if(answer_character_position>0):
				# 	positions.append(answer_character_position-1)
				answer = len(nltk.word_tokenize(text))
				intervals.append([answer, answer+len_answer-1])
				#positions.append(answer_character_position+len(answerin['text']))
			answer_interval.append(intervals)
		else:		
			answerin = answers[i][0]
			text_answer = answerin['text']
			len_answer = len(nltk.word_tokenize(text_answer))
			answer_character_position = answerin['answer_start']
			text = x['context'][:answer_character_position]
			if(not (answer_character_position>0 and text[answer_character_position-1]!=" " 
					and text[answer_character_position-1]!="(" 
					and text[answer_character_position-1]!="$" 
					and text[answer_character_position-1]!='"'
					and text[answer_character_position-1]!='“'
					and text[answer_character_position-1]!="'"
					and text[answer_character_position-1]!="⟨"
					and text[answer_character_position-1]!="[")):
			# text = text.replace("-", " - ")
			# text = text.replace("–", " – ")
			# text = text.replace(chr(8212)," "+chr(8212)+" ")
			# if(answer_character_position>0):
			# 	positions.append(answer_character_position-1)
				answer = len(nltk.word_tokenize(text))
				answer_interval.append([answer, answer+len_answer-1])
			#positions.append(answer_character_position+len(answerin['text']))

	# positions = list(set(positions))
	# positions.sort()
	text = x['context']
	# for i in range(len(positions)):
	# 	pos = positions[i] + i
	# 	text = text[:pos] + ' ' + text[pos:]


	def que_tra(y):
		question = y['question']
		return nltk.word_tokenize(question)
	question = list(map(lambda y: que_tra(y) , qas))
	context = nltk.word_tokenize(text)
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
		documents = []
		questions = []
		answers = []
		ids = []
		lengths_doc = []
		lengths_que = []
		for i in range(0,len(data_array)):
			doc = list.copy(data_array[i][0])
			que = list.copy(data_array[i][1])
			pad = gloveDimension * [0.0]
			lengths_doc.append(len(doc))
			lengths_que.append(len(que))
			doc.extend([pad] * (d_length - len(doc)))
			que.extend([pad] * (q_length - len(que)))
			ans = data_array[i][2]
			q_id = data_array[i][3]
			documents.append(doc)
			questions.append(que)
			answers.append(ans)
			ids.append(q_id)

		print("HERE")
		for i in range(len(documents)):
			if(answers[i][1]>=lengths_doc[i]):
				print(i)
		output_data_array = [[documents, questions, answers, ids], [lengths_doc, lengths_que]]
		np.save('data/'+type_of_embeddings+'_shuffled', output_data_array)
	else:
		documents = list(map (lambda x: np.array(x[0]), data_array))
		questions = list(map (lambda x: np.array(x[1]), data_array))
		answers = list(map (lambda x: np.array(x[2]), data_array))
		data_array = [documents,questions,answers] 


# the types are 'padded_train_data', 'unpadded_train_data', 'padded_test_data', 'unpadded_test_data'
# the padded version also contain the lenghts of the original data points 
type_of_embeddings = 'padded_train_data'
save_embeddings(type_of_embeddings)
