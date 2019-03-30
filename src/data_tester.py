import ciprian_data_prep_script

input_d_vecs, input_q_vecs, ground_truth_labels, documents_lengths, questions_lengths, question_ids, _ = ciprian_data_prep_script.get_data("train")

'''
print(input_d_vecs[0])
print(input_q_vecs[0])
print(ground_truth_labels[0])
print(documents_lengths[0])
print(questions_lengths[0])
'''

dataset_length = len(input_d_vecs)
batch_size = 1
print(dataset_length)
nerrors = 0
for i in range(0, dataset_length):
    d = input_d_vecs[i],
    q = input_q_vecs[i],
    ground_truth = ground_truth_labels[i],
    doc_l = documents_lengths[i],
    doc_l = doc_l[0]
    que_l = questions_lengths[i]
    qid = question_ids[i]
    for answer in ground_truth:
        if not(answer[0] < doc_l and
               answer[0] <= answer[1] and
               answer[1] < doc_l and
               answer[0] <= answer[1]):
            print("ERROR at", i, ":  answer:", answer, "   doc_l:", doc_l, "   qid:", qid)
            nerrors += 1
print(nerrors, " errors")
