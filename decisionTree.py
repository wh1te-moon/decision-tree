from graphviz import Digraph
import numpy as np
from itertools import combinations, product
from multiprocessing import Pool, freeze_support
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.metrics import accuracy_score

np.random.seed(42)


class TreeNode:
    def __init__(self, feature_indices=None, threshold=None, value=None, left=None, right=None, depth=1):
        self.feature_indices = feature_indices
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right
        self.depth = depth

def predict_tree(node, sample):
    if node.value is not None:
        return node.value
    else:
        if np.all(sample[node.feature_indices] <= node.threshold):
            return predict_tree(node.left, sample)
        else:
            return predict_tree(node.right, sample)

def calculate_entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def find_best_split_helper(args):
    X, y, feature_combination, thresholds = args

    # Create a boolean matrix for left_mask and right_mask
    left_mask = np.all(X[:, feature_combination] <= thresholds, axis=1)
    right_mask = ~left_mask

    left_entropy = calculate_entropy(y[left_mask])
    right_entropy = calculate_entropy(y[right_mask])
    total_entropy = left_entropy + right_entropy

    return total_entropy, feature_combination, thresholds


def find_best_split(X, y, feature_indices, depth):
    best_entropy = float('inf')
    best_feature_indices = None
    best_thresholds = None

    with Pool() as pool:
        for feature_combination in combinations(feature_indices, min(depth, 2)):
            # Find the union of unique values for all features
            args_list = [(X, y, feature_combination, threshold_values)
                        for threshold_values in np.unique(X[:, feature_combination])]
            # print(feature_combination)
            # for arg in args_list:
            #     result = find_best_split_helper(arg)
            #     if result[0] < best_entropy:
            #         best_entropy = result[0]
            #         best_feature_indices = result[1]
            #         best_thresholds = result[2]
            results = pool.map(find_best_split_helper, args_list)
            for result in results:
                if result[0] < best_entropy:
                    best_entropy = result[0]
                    best_feature_indices = result[1]
                    best_thresholds = result[2]

    return best_feature_indices, best_thresholds


def build_tree(X, y, all_feature_indices, max_depth, depth=1, value=None):
    if len(np.unique(y)) == 0:
        return TreeNode(value=value, depth=depth)

    if max_depth == 1 or len(np.unique(y)) == 1:
        return TreeNode(value=np.argmax(np.bincount(y)), depth=depth)

    feature_indices, thresholds = find_best_split(
        X, y, all_feature_indices, depth)

    if feature_indices is None:
        return TreeNode(value=np.argmax(np.bincount(y)), depth=depth)

    left_mask = np.all(X[:, feature_indices] <= np.array(thresholds), axis=1)
    right_mask = ~left_mask

    left_subtree = build_tree(X[left_mask], y[left_mask], all_feature_indices, max_depth=max_depth - 1, depth=depth + 1,
                              value=np.argmax(np.bincount(y[right_mask])) if len(np.unique(y[left_mask])) == 0 else np.argmax(np.bincount(y[left_mask])))
    right_subtree = build_tree(X[right_mask], y[right_mask], all_feature_indices, max_depth=max_depth - 1, depth=depth + 1,
                               value=np.argmax(np.bincount(y[left_mask])) if len(np.unique(y[right_mask])) == 0 else np.argmax(np.bincount(y[right_mask])))

    return TreeNode(feature_indices=list(feature_indices), threshold=thresholds, left=left_subtree, right=right_subtree, depth=depth)


if __name__ == '__main__':
    freeze_support()
    # dataset = load_breast_cancer()
    dataset = load_iris()
    X = dataset.data  # type: ignore
    y = dataset.target  # type: ignore
    # Define feature indices and maximum depth for the decision tree
    feature_indices = list(range(X.shape[1]))
    max_depth = min(5, X.shape[1])
    for i in range(1,6):
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1*i, random_state=42)
        # Build the decision tree
        tree = build_tree(X_train, y_train, feature_indices, max_depth)

        y_pred = []
        for sample in X_train:
            prediction = predict_tree(tree, sample)
            y_pred.append(prediction)

        accuracy = accuracy_score(y_train, y_pred)
        print(f"Accuracy: {accuracy:.2%}")

        # 对测试集进行预测
        y_pred = []
        for sample in X_test:
            prediction = predict_tree(tree, sample)
            y_pred.append(prediction)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2%}\n")

    dot = Digraph(comment='Decision Tree')

    def visualize_tree(tree, parent_name=None, side='root'):
        global dot

        node_name = f"{parent_name}_{tree.depth}_{side}"

        if tree.value is not None:
            dot.node(node_name, label=f"Class: {tree.value}")
        else:
            dot.node(
                node_name, label=f"Feature: {tree.feature_indices}\nThreshold: {tree.threshold}")

            if tree.left is not None:
                left_name = visualize_tree(
                    tree.left, parent_name=node_name, side='left')
                dot.edge(str(node_name), str(left_name), label='True')

            if tree.right is not None:
                right_name = visualize_tree(
                    tree.right, parent_name=node_name, side='right')
                dot.edge(str(node_name), str(right_name), label='False')

        return node_name
