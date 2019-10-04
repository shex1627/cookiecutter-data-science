

from graphviz import Digraph
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# +
# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    return dataset[dataset[:, index] == value,
                   ], dataset[dataset[:, index] != value, ]


def proportion_explained(groups):
    return groups[0].shape[0] / (groups[0].shape[0] + groups[1].shape[0])


# Select the best split point for a dataset
def get_split(dataset, col_indices):
    b_index, b_value, b_score, b_groups = 999, 999, -1, None
    for index in col_indices:
        mode, count = pd.Series(dataset[:, index]).value_counts().head(
            1).reset_index().iloc[0, ]
        proportion = count / dataset.shape[0]
        groups = test_split(index, mode, dataset)
        if proportion > b_score:
            b_index, b_value, b_score, b_groups = index, mode, proportion, groups
    return {
        'index': b_index,
        'value': b_value,
        'groups': b_groups,
        'col_indices': col_indices,
        'proportion': b_score,
        'proportion_str': f"{np.round(b_score,2)} ({groups[0].shape[0]}/{dataset.shape[0]})"}


def to_terminal(group):
    return group.shape[0]


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left.shape[0] or not right.shape[0] or not node['col_indices']:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        du_col_indices = list(node['col_indices'])
        du_col_indices.remove(node['index'])
        if du_col_indices:
            node['left'] = get_split(left, du_col_indices)
            # print(node['left'])
            split(node['left'], max_depth, min_size, depth + 1)
        else:
            node['left'] = to_terminal(left)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, node['col_indices'])
        split(node['right'], max_depth, min_size, depth + 1)


# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train, list(range(train.shape[1])))
    split(root, max_depth, min_size, 1)
    return root


# Print a decision tree
def print_tree(node, depth=0, col_dict=None):
    if not col_dict:
        col_dict = {index: 'X' + str(index + 1)
                    for index in node['col_indices']}
    if isinstance(node, dict):
        print('%s[%s = %s] %.3f' % (
            (depth * '   ', (col_dict[node['index']]), node['value'], node['proportion'])))
        print_tree(node['left'], depth + 1, col_dict)
        print_tree(node['right'], depth + 1, col_dict)
    else:
        print('%s leaf [%s]' % ((depth * '   ', node)))


# +


def make_dot(tree, col_dict):
    dot = Digraph('test_tree')
    num_nodes = 0

    def add_data(node, col_dict=None, depth=0):
        global num_nodes
        num_nodes = num_nodes + 1
        if not col_dict:
            col_dict = {index: 'X' + str(index + 1)
                        for index in node['col_indices']}
        if isinstance(node, dict):
            node_curr = (str(num_nodes), '[%s = %s] \\n %s' % (
                ((col_dict[node['index']]), node['value'], node['proportion_str'])))

            left_node = add_data(node['left'], col_dict, depth + 1)
            right_node = add_data(node['right'], col_dict, depth + 1)

            dot.node(node_curr[0], node_curr[1])
            dot.edge(node_curr[0], left_node[0], 'yes')
            dot.edge(node_curr[0], right_node[0], 'no')
        else:
            node_curr = (str(num_nodes), f"leaf [{node}]")
        return node_curr

    add_data(tree, col_dict)
    return dot

# +
# to visualize things on note book
# import pydotplus
# dot = make_dot(tree, col_dict)
# graph = pydotplus.graph_from_dot_data(dot_source
# from IPython.display import Image
# Image(graph.create_png())
