import numpy as np

def get_data():
        filename = 'data/padded_train_d.npy'
        # Taking only a slice of size 1 from the data (just one element)
        # The code will work with all the data once we feed into it the 
        # padded sequences, i.e. once all tensor elements agree on sizes
        # Keep it like this until we get one point running through the whole architecture
        try:
            data_array = np.load(filename)
        except:
            data_array = np.load('../data/padded_train_data.npy')
        print("Get Data stage: ciprian_data_prep read ", filename, " of shape: ", data_array.shape)
        documents = data_array[0][0]
        questions = data_array[0][1]
        answers = data_array[0][2]
        lengths_documents = data_array[1][0]
        lengths_questions = data_array[1][1]
        size = 1000
        return documents[:size], questions[:size], answers[:size], lengths_documents[:size], lengths_questions[:size] 
