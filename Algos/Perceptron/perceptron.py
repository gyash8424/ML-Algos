import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        # Initialize learning rate, number of iterations, and activation function
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # X: Training data with shape (n_samples, n_features)
        # y: Target labels for each sample
        
        n_samples, n_features = X.shape

        # Initialize weights and bias to zero
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Convert labels to binary (1 if > 0, else 0)
        y_ = np.array([1 if i > 0 else 0 for i in y])

        # Training loop for a fixed number of iterations
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Calculate the linear output: dot product of weights and input plus bias
                linear_output = np.dot(x_i, self.weights) + self.bias
                # Apply activation function to get predicted label
                y_predicted = self.activation_func(linear_output)

                # Update rule: adjust weights and bias based on prediction error
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i  # Update weights
                self.bias += update           # Update bias

    def predict(self, X):
        # Make predictions on input data X
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        # Activation function: returns 1 if x >= 0, else 0
        return np.where(x >= 0, 1, 0)


# Testing the Perceptron
if __name__ == "__main__":
    # Import required libraries
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    # Define accuracy function
    def accuracy(y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)

    # Generate a simple dataset with two clusters for binary classification
    X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.5)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Initialize and train the Perceptron model
    p = Perceptron(learning_rate=0.01, n_iters=1000)
    p.fit(X_train, y_train)

    # Predict on the test set and calculate accuracy
    predictions = p.predict(X_test)
    print("Perceptron classification accuracy", accuracy(y_test, predictions))

    # Visualization of decision boundary
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

    # Determine the decision boundary line based on weights and bias
    x0_1 = np.amin(X_train[:, 0])  # Minimum x-coordinate value
    x0_2 = np.amax(X_train[:, 0])  # Maximum x-coordinate value

    # Calculate corresponding y-coordinates for decision boundary
    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    # Plot decision boundary
    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    # Set y-axis limits for better visualization
    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])

    # Display plot
    plt.show()
