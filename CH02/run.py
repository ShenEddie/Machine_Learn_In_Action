import kNN

group, labels = kNN.create_dataset()

print(kNN.classify0([0, 0], group, labels, 3))

dating_data_mat, dating_labels = kNN.file2matrix('./CH02/datingTestSet.txt')

norm_mat, ranges, min_values = kNN.auto_norm(dating_data_mat)

kNN.dating_class_test('./CH02/datingTestSet.txt')

kNN.classify_person('./CH02/datingTestSet2.txt')

test_vec = kNN.img2vector('./CH02/digits/testDigits/0_13.txt')

kNN.handwriting_class_test('./CH02/digits/trainingDigits', './CH02/digits/testDigits')
