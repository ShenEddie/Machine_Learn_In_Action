import matplotlib.pyplot as plt
from typing import Tuple, Dict

decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")
ax: plt.Axes


def plot_node(node_text: str,
              center_point: Tuple[float, float],
              parent_point: Tuple[float, float],
              node_type: Dict):
    global ax
    ax.annotate(node_text, xy=parent_point,
                xycoords='axes fraction',
                xytext=center_point, textcoords='axes fraction',
                va="center", ha="center", bbox=node_type,
                arrowprops=arrow_args)


def create_plot():
    fig: plt.Figure = plt.figure(1, facecolor='white')
    fig.clf()
    global ax
    ax = plt.subplot(111, frameon=False)
    plot_node('a decision node', (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node('a leaf node', (0.8, 0.1), (0.3, 0.8), leaf_node)
    plt.show()
