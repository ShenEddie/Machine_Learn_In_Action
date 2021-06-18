import kNN

group, labels = kNN.create_dataset()

print(kNN.classify0([0, 0], group, labels, 3))

dating_data_mat, dating_labels = kNN.file2matrix('./CH02/datingTestSet.txt')

norm_mat, ranges, min_values = kNN.auto_norm(dating_data_mat)

kNN.dating_class_test('./CH02/datingTestSet.txt')
