"""
The pictures for dating example.
"""
# %% Import packages.
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import kNN

# %% Read data.
dating_data_mat, dating_labels = kNN.file2matrix('./CH02/datingTestSet2.txt')

# %% Draw picture.
fig: plt.Figure = plt.figure()
ax: plt.Axes = fig.add_subplot(111)
ax.scatter(
    dating_data_mat[:, 1],
    dating_data_mat[:, 2],
    15.0 * np.array(dating_labels),
    15.0 * np.array(dating_labels)
)
ax.set_xlabel('Percentage of Time Spent Playing Video Games')
ax.set_ylabel('Liters of Ice Cream Consumed Per Week')
ax.axis([-2, 25, -0.2, 2.0])
fig.show()

# %% Draw picture.
fig: plt.Figure = plt.figure()
ax: plt.Axes = fig.add_subplot(111)
scatter = ax.scatter(
    dating_data_mat[:, 0],
    dating_data_mat[:, 1],
    15.0 * np.array(dating_labels),
    15.0 * np.array(dating_labels)
)
ax.set_ylabel('Frequent Flyer Miles Earned Per Year')
ax.set_xlabel('Percentage of Time Spent Playing Video Games')
fig.show()
