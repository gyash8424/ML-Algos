import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        # X is a 2D array with shape (n_samples, n_features)
        # y is a 1D array containing class labels for each sample
        n_samples, n_features = X.shape

        # Get unique classes in the training data
        self._classes = np.unique(y)

        # Number of unique classes in the dataset
        n_classes = len(self._classes)

        # Initialize arrays to store mean, variance, and prior probability for each class
        # _mean and _var will have shape (n_classes, n_features)
        # _priorprob will have shape (n_classes,)
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priorprob = np.zeros(n_classes, dtype=np.float64)

        # Calculate mean, variance, and prior probability for each class
        for idx, c in enumerate(self._classes):
            # Select samples belonging to class 'c'
            X_c = X[y==c]
            # Compute mean of features for class 'c'
            self._mean[idx,:] = X_c.mean(axis=0)
            # Compute variance of features for class 'c'
            self._var[idx,:] = X_c.var(axis = 0)
            # Compute prior probability of class 'c'
            self._priorprob[idx] = X_c.shape[0]/float(n_samples)

    def predict(self, X):
        # Predict class labels for each sample in X
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        # Calculate the posterior probability for each class
        posteriors = []

        for idx, c in enumerate(self._classes):
            # Log of prior probability of the class
            prior = np.log(self._priorprob[idx])
            # Sum of log of likelihoods (calculated using PDF for each feature)
            posterior = np.sum(np.log(self._pdf(idx, x)))
            # Add prior and likelihood to get posterior
            posterior = prior + posterior
            posteriors.append(posterior)
        
        # Return class with the highest posterior probability
        return self._classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx, x):
        # Calculate probability density function for Gaussian distribution
        mean = self._mean[class_idx]
        var = self._var[class_idx]

        # Numerator of Gaussian PDF
        numerator = np.exp(-((x-mean)**2)/ (2*var))
        # Denominator of Gaussian PDF
        denominator = np.sqrt(2* np.pi * var)

        return numerator / denominator
    
if __name__ == "__main__":
    # Import necessary libraries for model evaluation
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    # Define accuracy metric
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    # Generate synthetic dataset with 1000 samples, 10 features, and 2 classes
    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2
    )

    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    # Initialize Naive Bayes classifier and train it on the training data
    nb = NaiveBayes()
    nb.fit(X_train, y_train)

    # Predict class labels for the test set
    predictions = nb.predict(X_test)

    # Print accuracy of the classifier on the test set
    print("Naive Bayes classification accuracy", accuracy(y_test, predictions))