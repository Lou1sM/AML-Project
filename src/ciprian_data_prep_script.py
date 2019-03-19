import numpy as np

def get_data(test_case):
        # filename = 'data/padded_train_data_shuffled.npy'
        filename = 'data/padded_test_data_shuffled.npy'
        # Taking only a slice of size 1 from the data (just one element)
        # The code will work with all the data once we feed into it the 
        # padded sequences, i.e. once all tensor elements agree on sizes
        # Keep it like this until we get one point running through the whole architecture
        try:
            data_array = np.load(filename)
        except:
            # data_array = np.load('../data/padded_train_data.npy')
            data_array = np.load('../data/padded_test_data_shuffled.npy')

        print("Get Data stage: ciprian_data_prep read ", filename, " of shape: ", data_array.shape)
        documents = data_array[0][0]
        questions = data_array[0][1]
        answers0 = list(map(lambda x: x[0], data_array[0][2]))
        all_answers = data_array[0][2]
        lengths_documents = data_array[1][0]
        lengths_questions = data_array[1][1]
        if test_case:
            ids_questions = data_array[0][3] # or [1][2] depending on data format
        else:
            ids_questions = []
        size = len(lengths_documents)
        # if(test_case):
        #     size = 65
        return documents[:size], questions[:size], answers0[:size], lengths_documents[:size], lengths_questions[:size], ids_questions[:size], all_answers[:size]
