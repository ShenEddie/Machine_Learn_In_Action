"""
The functions for decision tree algorithm.
"""
__all__ = ['cal_shannon_ent']

from math import log2
from typing import List, Union, Tuple


def cal_shannon_ent(dataset: List[List[Union[int, str]]]) -> float:
    """
    Calculate the Shannon Entropy.

    Args:
        dataset: The dataset with labels in last column.

    Returns:
        The Shannon Entropy.
    """
    num_entries = len(dataset)
    label_counts = {}
    for feat_vec in dataset:
        current_label = feat_vec[-1]
        label_counts[current_label] += label_counts.get(current_label, 0)
    shannon_ent = 0
    for key in label_counts.keys():
        prob = label_counts[key] / num_entries
        shannon_ent -= prob * log2(prob)
    return shannon_ent


def create_dataset() -> Tuple[List[List[Union[int, str]]], List[str]]:
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels
