"""
The functions for decision tree algorithm.
"""
__all__ = ['cal_shannon_ent', 'create_dataset', 'split_dataset',
           'choose_best_feature_to_split']

from math import log2
from typing import List, Union, Tuple


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


def choose_best_feature_to_split(dataset: List[List[Union[int, str]]]):
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
