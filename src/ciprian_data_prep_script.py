import numpy as np

def get_data():
	filename = 'data/small_data.npy'
	# Taking only a slice of size 1 from the data (just one element)
	# The code will work with all the data once we feed into it the 
	# padded sequences, i.e. once all tensor elements agree on sizes
	# Keep it like this until we get one point running through the whole architecture
	data_array = np.load(filename)[:1]
	documents = np.array(list(map (lambda x: np.array(x[0]), data_array)))
	questions = np.array(list(map (lambda x: np.array(x[1]), data_array)))
	answers = np.array(list(map (lambda x: np.array(x[2]), data_array)))
	return documents, questions, answers
