import kNN

group, labels = kNN.create_dataset()

print(kNN.classify0([0, 0], group, labels, 3))

dating_data_mat, dating_labels = kNN.file2matrix('./CH02/datingTestSet.txt')
