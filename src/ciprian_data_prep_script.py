import numpy as np

def get_data():
        filename = 'data/padded_train_data.npy'
        # Taking only a slice of size 1 from the data (just one element)
        # The code will work with all the data once we feed into it the 
        # padded sequences, i.e. once all tensor elements agree on sizes
        # Keep it like this until we get one point running through the whole architecture
        try:
            data_array = np.load(filename)
        except:
            data_array = np.load('../data/padded_train_data.npy')
        print(data_array.shape)
        documents = data_array[0][0]
        questions = data_array[0][1]
        answers = data_array[0][2]
        lengths_documents = data_array[1][0]
        lengths_questions = data_array[1][1]
        return documents[:100], questions[:100], answers[:100], lengths_documents[:100], lengths_questions[:100] 
