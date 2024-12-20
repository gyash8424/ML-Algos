import numpy as np
from collections import Counter

def entropy(y):
    hist = np.bincount(y) # Count occurrences of each class
    ps = hist/len(y) # Class probabilities
    return -np.sum([p*np.log2(p) for p in ps if p>0]) # Entropy formula

class Node:
    '''Any arguments specified after the * in the parameter list
     must be passed by name (using keywords) when calling the function.'''
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature # Feature index for splitting
        self.threshold = threshold # Threshold value for splitting
        self.left = left
        self.right = right
        self.value = value # Value if it's a leaf node (most common label)

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:

    def __init__(self, min_samples_split =2, max_depth = 100, n_features = None):
        self.min_samples_split = min_samples_split  # Minimum samples required to split
        self.max_depth = max_depth  # Maximum tree depth
        self.n_feats = n_features  # Number of features to consider for splitting
        self.root = None  # Root node of the tree

    def fit(self, X,y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X,y)
    

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    

    def _grow_tree(self, X,y, depth=0):
        # Recursively grow the decision tree.
        n_samples, n_features  = X.shape
        n_labels = len(np.unique(y))

        # Stopping conditions
        if (depth>= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split):

            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Select features to consider for splitting
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # Find the best split
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)

        # Split the dataset
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)

        left = self._grow_tree(X[left_idxs,:], y[left_idxs], depth+1)
        right  =self._grow_tree(X[right_idxs,:], y[right_idxs], depth+1)

        return Node(best_feat, best_thresh, left, right)
    
    def _best_criteria(self, X, y, feat_idxs):
        """
        Returns the feature index and threshold
        that provide the highest information gain.
        """
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain> best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh
    
    def _information_gain(self, y, X_column, split_thresh):
        """
        Information gain is the reduction
        in entropy after the split.
        """
        parent_entropy = entropy(y)

        # Split the data
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        # If no split occurs, return 0 gain
        if len(left_idxs)==0 or len(right_idxs)==0:
            return 0
        
        # Compute weighted average of child entropy
        n = len(y)

        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l/n)*e_l + (n_r/n)*e_r

        # Information gain
        ig = parent_entropy - child_entropy

        return ig
    
    def _split(self, X_column, split_thresh):
        """
        Partitions the dataset into left and right subsets based on the threshold.
        Points less than or equal to the threshold go to the left subset, while
        points greater than the threshold go to the right subset.
        """
        left_idxs = np.argwhere(X_column<=split_thresh).flatten()
        right_idxs = np.argwhere(X_column>split_thresh).flatten()
        return left_idxs, right_idxs
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature]<=node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    
if __name__=='__main__':
    # Imports
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    clf = DecisionTree(max_depth=10)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy(y_test, y_pred)

    print("Accuracy:", acc)