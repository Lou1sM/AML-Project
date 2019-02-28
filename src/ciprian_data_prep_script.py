import numpy as np

def get_data():
	filename = '../data/data.npy'
	data_array = np.load(filename)
	documents = np.array(list(map (lambda x: x[0], data_array)))
	questions = np.array(list(map (lambda x: x[1], data_array)))
	answers = np.array(list(map (lambda x: x[2][0], data_array)))
	return documents, questions, answers
