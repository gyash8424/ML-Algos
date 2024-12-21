from collections import Counter
import numpy as np
from Decision_Tree import DecisionTree

# Function to create a bootstrap sample from the dataset
def bootstrap_sample(X, y):
    """
    Generate a bootstrap sample of the dataset.
    Args:
        X: Features matrix.
        y: Target array.
    Returns:
        A bootstrap sample (random sample with replacement) of X and y.
    """
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    return X[idxs], y[idxs]

# Function to find the most common label in an array
def most_common_label(y):
    """
    Determine the most common label in the array.
    Args:
        y: Array of labels.
    Returns:
        The most frequently occurring label.
    """
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common

# Implementation of the Random Forest algorithm
class RandomForest:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=100, n_feats=None):
        """
        Initialize the RandomForest class.
        Args:
            n_trees: Number of decision trees in the forest.
            min_samples_split: Minimum samples required to split a node.
            max_depth: Maximum depth of each tree.
            n_feats: Number of features to consider when splitting a node.
        """
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []  # List to store all decision trees

    def fit(self, X, y):
        """
        Train the random forest on the dataset.
        Args:
            X: Features matrix.
            y: Target array.
        """
        self.trees = []  # Clear the list of trees before fitting
        for _ in range(self.n_trees):
            # Create and configure a new decision tree
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_features=self.n_feats,
            )
            # Generate a bootstrap sample of the data
            X_samp, y_samp = bootstrap_sample(X, y)
            # Train the decision tree on the bootstrap sample
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)

    def predict(self, X):
        """
        Make predictions for a given set of input samples.
        Args:
            X: Features matrix of input samples.
        Returns:
            An array of predicted labels.
        """
        # Collect predictions from all decision trees
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Transpose the predictions to group predictions for each sample
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        # Determine the most common label for each sample
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)

# Testing the Random Forest implementation
if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    # Function to calculate the accuracy of predictions
    def accuracy(y_true, y_pred):
        """
        Calculate the accuracy of predictions.
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
        Returns:
            Accuracy as a fraction of correct predictions.
        """
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    # Load the breast cancer dataset from sklearn
    data = datasets.load_breast_cancer()
    X = data.data  # Features
    y = data.target  # Labels

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    # Create a RandomForest classifier with 3 trees and a maximum depth of 10
    clf = RandomForest(n_trees=3, max_depth=10)

    # Train the RandomForest on the training data
    clf.fit(X_train, y_train)
    # Predict labels for the testing data
    y_pred = clf.predict(X_test)
    # Calculate the accuracy of the predictions
    acc = accuracy(y_test, y_pred)

    # Print the accuracy of the RandomForest model
    print("Accuracy:", acc)
