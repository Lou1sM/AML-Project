import numpy as np

def get_data(typeData, squad2):
    if not squad2:
        if (typeData == "train"):
            filename = 'data/padded_train_data_shuffled.npy'
        elif (typeData == "test"):
            filename = 'data/padded_test_data_shuffled.npy'
    else:
        if (typeData == "train"):
            filename = 'data/padded_train_data_shuffled_squad2.npy'
        elif (typeData == "test"):
            filename = 'data/padded_test_data_shuffled_squad2.npy'

    try:
        data_array = np.load(filename)
    except:
        data_array = np.load('/home/shared/' + filename)
    #print("Get Data stage: anonymous2_data_prep read ", filename, " of shape: ", data_array.shape)

    documents = data_array[0][0]
    questions = data_array[0][1]
    answers = data_array[0][2]
    lengths_documents = data_array[1][0]
    lengths_questions = data_array[1][1]
    size = len(lengths_documents)
    all_answers = []
    ids_questions = []

    if(typeData == "test"):
        answers = list(map(lambda x: x[0], data_array[0][2]))
        all_answers = data_array[0][2]
        ids_questions = data_array[0][3]

    return documents[:size], questions[:size], answers[:size], lengths_documents[:size], lengths_questions[:size], ids_questions[:size], all_answers[:size]
