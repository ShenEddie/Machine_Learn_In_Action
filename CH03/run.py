import trees

my_data, _ = trees.create_dataset()
print(trees.cal_shannon_ent(my_data))

my_data[0][-1] = 'maybe'
print(trees.cal_shannon_ent(my_data))

my_data, _ = trees.create_dataset()
trees.split_dataset(my_data, 0, 1)
trees.split_dataset(my_data, 0, 0)

trees.choose_best_feature_to_split(my_data)
