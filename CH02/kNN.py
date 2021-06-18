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


def file2matrix(filename: str) -> Tuple[np.ndarray, List[Union[int, str]]]:
    """
    Read features and labels from a txt file.

    Args:
        filename: The path of the txt file which contains the dataset.

    Returns:
        The array of features and the list of labels.
    """
    fr = open(filename)
    lines = fr.readlines()
    number_of_lines = len(lines)
    number_of_features = len(lines[0].split('\t')) - 1
    return_mat = np.empty((number_of_lines, number_of_features))
    class_label_vector: List[Union[int, str]] = []
    index = 0
    for line in lines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0:number_of_features]
        if list_from_line[-1].isdigit():
            class_label_vector.append(int(list_from_line[-1]))
        else:
            class_label_vector.append(list_from_line[-1])
        index += 1
    return return_mat, class_label_vector


def auto_norm(dataset: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize the dataset.

    Args:
        dataset: The dataset of add features.

    Returns:
        Normalized dataset, Ranges between maximums and minimums, minimums.
    """
    min_values = dataset.min(0, initial=None)
    max_values = dataset.max(0, initial=None)
    ranges = max_values - min_values
    m = dataset.shape[0]
    normed_dataset = dataset - np.tile(min_values, (m, 1))
    normed_dataset = normed_dataset / np.tile(ranges, (m, 1))
    return normed_dataset, ranges, min_values


def dating_class_test(filename: str, ho_ratio: float = 0.1) -> float:
    """
    Calculate the error rate for kNN.

    Args:
        filename: The path of the txt file which contains the dataset.
        ho_ratio: The proportion of data to be placed to the test set.

    Returns:
        The total error rate.
    """
    dating_data_mat, dating_labels = file2matrix(filename)
    normed_mat, ranges, min_values = auto_norm(dating_data_mat)
    m = normed_mat.shape[0]
    num_test_vectors = int(m * ho_ratio)
    error_count = 0
    for i in range(num_test_vectors):
        classified_res = classify0(
            normed_mat[i, :],
            normed_mat[num_test_vectors:m, :],
            dating_labels[num_test_vectors:m],
            k=3
        )
        print("{}: The classifier came back with: {}, The real answer is: {}".format(
            i, classified_res, dating_labels[i]
        ))
        if classified_res != dating_labels[i]:
            error_count += 1
    print("The total error rate is: {}".format(error_count / num_test_vectors))
    return error_count / num_test_vectors
