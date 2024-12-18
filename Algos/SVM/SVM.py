import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param # regularization parameter
        self.n_iters = n_iters
        self.w = None  # Weight vector
        self.b = None  # Bias term

    def fit(self, X, y):
        #Train the SVM using Stochastic Gradient Descent (SGD).
        
        n_samples, n_features = X.shape

        # Convert labels to {-1, 1} if they are not already
        y_ = np.where(y <= 0, -1, 1)

        # Initialize weights and bias to zero
        self.w = np.zeros(n_features)
        self.b = 0

        # Perform training for n_iters
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Check if the current sample satisfies the margin condition
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    # No misclassification: Only regularization term updates
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Misclassification: Update weights and bias using SVM loss
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)


# Testing the SVM implementation
if __name__ == "__main__":
    from sklearn import datasets
    import matplotlib.pyplot as plt

    # Generate a simple dataset with two classes for binary classification
    X, y = datasets.make_blobs(
        n_samples=50, n_features=2, centers=2, cluster_std=1.05
    )
    # Convert target labels to {-1, 1} for SVM
    y = np.where(y == 0, -1, 1)

    # Initialize and train the SVM model
    clf = SVM()
    clf.fit(X, y)

    # Print the learned weights and bias
    print(clf.w, clf.b)

    # Visualization function for the decision boundary and margins
    def visualize_svm():
        def get_hyperplane_value(x, w, b, offset):
            """
            Calculate the y-coordinate of the hyperplane for a given x-coordinate.
            - x: x-coordinate
            - w: weight vector
            - b: bias term
            - offset: distance from the main decision boundary (for margin lines)
            """
            return (-w[0] * x + b + offset) / w[1]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # Plot the data points with colors based on their labels
        plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

        # Define x-coordinates for plotting the hyperplane
        x0_1 = np.amin(X[:, 0])  # Minimum x-coordinate
        x0_2 = np.amax(X[:, 0])  # Maximum x-coordinate

        # Main decision boundary (offset = 0)
        x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
        x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

        # Negative margin line (offset = -1)
        x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
        x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

        # Positive margin line (offset = 1)
        x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
        x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

        # Plot the decision boundary and margins
        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")  # Decision boundary
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")  # Negative margin
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")  # Positive margin

        # Set y-axis limits for better visualization
        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])

        # Display the plot
        plt.show()

    # Visualize the trained SVM decision boundary
    visualize_svm()