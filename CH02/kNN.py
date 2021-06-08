"""
The functions for KNN algorithm.
"""
__all__ = ['create_dataset', 'classify0', 'file2matrix']

import numpy as np
import operator
from typing import Tuple, List, Union, Iterable


def create_dataset() -> Tuple[np.ndarray, List[str]]:
    """
    Create dataset for classification.

    Returns:
        Groups of features and labels for each group.
    """
    group = np.array([
        [1.0, 1.1],
        [1.0, 1.0],
        [0, 0],
        [0, 0.1]
    ])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(in_X: Union[np.ndarray, Iterable, int, float],
              dataset: np.ndarray,
              labels: List[Union[str, int]],
              k: int) -> Union[str, int]:
    """
    kNN algorithm.

    Args:
        in_X: The features for the item to be classified.
        dataset: The train set.
        labels: The labels of each group in train set.
        k: The parameter k for kNN method.

    Returns:
        The predicted class of "in_X".
    """
    dataset_size = dataset.shape[0]
    diff_mat = np.tile(in_X, (dataset_size, 1)) - dataset
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances: np.ndarray = sq_distances ** 0.5
    sorted_dist_indices = distances.argsort()
    class_count: dict = {}
    for i in range(k):
        vote_i_label = labels[sorted_dist_indices[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def file2matrix(filename: str) -> Tuple[np.ndarray, List[int]]:
    fr = open(filename)
    lines = fr.readlines()
    number_of_lines = len(lines)
    number_of_features = len(lines[0].split('\t')) - 1
    return_mat = np.empty((number_of_lines, number_of_features))
    class_label_vector = []
    index = 0
    for line in lines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0:number_of_features]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector
