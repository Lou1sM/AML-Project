import numpy as np
import tensorflow as tf
import json
import itertools
import time
import nltk 
import pickle 
filename = 'embedding/glove.6B/glove.6B.300d.txt'
json_filename = 'squad/train-v1.1.json'
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
	print('Loaded GloVe mothafucka!')
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
def process_squad(x):
	global max_count
	context = nltk.word_tokenize(x['context'])
	qas = x['qas']
	answers = list(map(lambda x: x['answers'], qas))
	answer_start = []
	for i in range(len(answers)):
		# if(len(answers[i])==1):
		answer = answers[i][0]
		len_answer = len(nltk.word_tokenize(answer['text']))
		answer = answer['answer_start']
		answer_start.append([answer, answer+len_answer])
		# elif(len(answer[i])==2):
		# 	print("pulalala")
		# 	answer = answers[i]
		# 	answer1 = answer[0]['answer_start']
		# 	len_answer = len(nltk.word_tokenize(answer[0]['text']))
		# 	answer1 = [answer1, answer1+len_answer]
		# 	answer2 = answer[1]['answer_start']
		# 	len_answer = len(nltk.word_tokenize(answer[1]['text']))
		# 	answer2 = [answer2, answer2+len_answer]
		# 	answer_start[i].append([answer1, answer2, answer1])
		# else:
		# 	answer = answers[i]
		# 	len_answer = len(nltk.word_tokenize(answer[0]['text']))
		# 	answer1 = answer[0]['answer_start']
		# 	answer1 = [answer1, answer1 + len_answer]
		# 	len_answer = len(nltk.word_tokenize(answer[1]['text']))
		# 	answer2 = answer[1]['answer_start']
		# 	answer2 = [answer2, answer2 + len_answer]
		# 	len_answer = len(nltk.word_tokenize(answer[2]['text']))
		# 	answer2 = answer[2]['answer_start']
		# 	answer3 = [answer3, answer3 + len_answer]
		#	answer_start[i].append([answer1, answer2, answer3])
	question = list(map(lambda x: nltk.word_tokenize(x['question']), qas))
	qa = list(zip(question, answer_start))
	product = [context, qa]
	return list(product)

def apply_embd(dic, context): 

	document = list(map(lambda x: dic[x.lower()] if x.lower() in dic else [0.0]*gloveDimension, context[0]))
	answers = list(map(lambda x: [list(map(lambda y: dic[y.lower()] if y.lower() in dic else [0.0]*gloveDimension, x[0])),x[1]], context[1]))
	answers = [[document, x[0], x[1]] for x in answers]
	return answers

def save_embeddings():
	embedding = load_glove(filename)
	data = load_train_data(json_filename)
	answer = map(lambda x: x['paragraphs'], data['data'])
	answer = list(answer)
	answer = [item for sublist in answer for item in sublist]
	data = list(map(lambda x: process_squad(x), answer))
	process_data = list(map(lambda x: apply_embd(embedding, x), data))
	data_array = [item for sublist in process_data for item in sublist]
	padded_array = []
	print('am inceput bossule..intr-un for cum intr-un ma-ta')
	for i in range(len(data_array)):
		print(i)
		doc = data_array[i][0]
		que = data_array[i][1]
		pad = gloveDimension * [0.0]
		doc.extend([pad] * (d_length - len(doc)))
		que.extend([pad] * (q_length - len(que)))
		ans = data_array[i][2][0]
		element = np.array([doc, que, ans])
		padded_array.append(element)
		data_array[i] = []

	padded_array = np.array(padded_array)
	np.save("data", padded_array)


save_embeddings()