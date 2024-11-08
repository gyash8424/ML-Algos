import numpy as np

class LogisticRegression:
    def __init__(self, lr = 0.0005, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X,y):
        # size of input decides the dimentions of metrices
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias

            y_predicted = self._sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)


            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        # calculating as if the linear model
        linear_model = np.dot(X, self.weights) + self.bias
        # passing the linear calculation to the sigmoid function to calculate the probability
        y_predicted = self._sigmoid(linear_model)
        # classifing on the basis of probabilities
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)


    def _sigmoid(self, x):
        return 1 / (1+np.exp(-x))
    
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    regressor = LogisticRegression()
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    print("LR classification accuracy:", accuracy(y_test, predictions))
