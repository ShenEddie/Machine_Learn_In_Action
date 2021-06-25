"""
The functions for decision tree algorithm.
"""
__all__ = ['cal_shannon_ent', 'create_dataset', 'split_dataset',
           'choose_best_feature_to_split', 'majority_count', 'create_tree']

import operator
from math import log2
from typing import List, Union, Tuple, Dict


def cal_shannon_ent(dataset: List[List[Union[int, str]]]) -> float:
    """
    Calculate the Shannon Entropy.

    Args:
        dataset: The dataset with classes in last column.

    Returns:
        The Shannon Entropy.
    """
    num_entries = len(dataset)
    label_counts = {}
    for feat_vec in dataset:
        current_label = feat_vec[-1]
        label_counts[current_label] = label_counts.get(current_label, 0) + 1
    shannon_ent = 0
    for key in label_counts.keys():
        prob = label_counts[key] / num_entries
        shannon_ent -= prob * log2(prob)
    return shannon_ent


def create_dataset() -> Tuple[List[List[Union[int, str]]], List[str]]:
    """
    Create dataset.

    Returns:
        The dataset with classes in last columns and the names of features.
    """
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


def split_dataset(dataset: List[List[Union[int, str]]],
                  axis: int,
                  value: Union[int, str]
                  ) -> List[List[Union[int, str]]]:
    """
    Split dataset.

    Args:
        dataset: The dataset to be split.
        axis: The axis of chosen feature.
        value: The value of chosen feature to be returned.

    Returns:
        The split dataset.
    """
    ret_dataset = []
    for feat_vec in dataset:
        if feat_vec[axis] == value:
            reduced_feat_vac = feat_vec[:axis]
            reduced_feat_vac.extend(feat_vec[axis + 1:])
            ret_dataset.append(reduced_feat_vac)
    return ret_dataset


def choose_best_feature_to_split(dataset: List[List[Union[int, str]]]) -> int:
    """
    Choose the best feature for splitting data.

    Args:
        dataset: The dataset to be split.

    Returns:
        The axis of the best feature.
    """
    num_features = len(dataset[0]) - 1
    base_entropy = cal_shannon_ent(dataset)
    best_info_gain = 0
    best_feature = -1
    for i in range(num_features):
        feat_list = [example[i] for example in dataset]
        unique_values = set(feat_list)
        new_entropy = 0
        for value in unique_values:
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset) / len(dataset)
            new_entropy += prob * cal_shannon_ent(sub_dataset)  # Weighted avg.
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_count(class_list: List) -> Union[int, str]:
    class_count = {}
    for vote in class_list:
        class_count[vote] = class_count.get(vote, 0) + 1
    sorted_class_count = sorted(class_count.items(),
                                key=operator.itemgetter(1),
                                reverse=True)
    return sorted_class_count[0][0]


def create_tree(dataset: List[List[Union[int, str]]],
                labels: List[str]) -> Union[Dict, int, str]:
    """
    Create Decision Tree.

    Args:
        dataset: The dataset with classes in last column.
        labels: The names of the features in the dataset.

    Returns:
        The class of the leaf node or the recursive tree.
    """
    class_list = [example[-1] for example in dataset]
    # Condition 1: All classes are the same.
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # Condition 2: No available features.
    if len(dataset[0]) == 1:
        return majority_count(class_list)
    best_feat = choose_best_feature_to_split(dataset)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    del labels[best_feat]
    feat_value = [example[best_feat] for example in dataset]
    unique_values = set(feat_value)
    for value in unique_values:
        sub_labels = labels[:]
        my_tree[best_feat_label][value] = create_tree(
            split_dataset(dataset, best_feat, value), sub_labels
        )
    return my_tree
