import kNN

group, labels = kNN.create_dataset()

print(kNN.classify0([0, 0], group, labels, 3))
