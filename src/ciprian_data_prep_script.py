import numpy as np

def get_data():
	filename = 'data/small_data.npy'
	data_array = np.load(filename)
	documents = np.array(list(map (lambda x: np.array(x[0]), data_array)))
	questions = np.array(list(map (lambda x: np.array(x[1]), data_array)))
	answers = np.array(list(map (lambda x: np.array(x[2]), data_array)))
	return documents, questions, answers
