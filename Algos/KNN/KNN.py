import numpy as np
from collections import Counter
class KNN:
    def __init__(self,k=3):
        # k is the number of nearest neighbours our algo will
        # consider before making the prediction
        self.k=k

    def fit(self,X,y):
        # this function takes in input data dpending on which our model will predict the class
        # Saving training data into the object
        self.X_train= X
        self.y_train=y
    
    def predict(self,X):
        # the function takes a range of inputs and makes prection on each one of them
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self,x):
        # calculate euclidian distance from each point in train dataset
        disctances = [KNN.euclidian_disctance(x,x_train) for x_train in self.X_train]
        # find k points with minimum distance
        k_indices = np.argsort(disctances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # find the maxium frequency label, and assign it as label of test data
        label = Counter(k_nearest_labels).most_common(1)
        return label[0][0]

    @staticmethod
    def euclidian_disctance(x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))

if __name__ == "__main__":
    # Imports
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    # defining our accuracy metrics
    def accuracy(y_true, y_pred):
        accuracy = float(np.sum(y_true == y_pred)) / float(len(y_true))
        return accuracy
    
    # loading pre existing dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # spliting data into train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    #initiating our model
    knn = KNN(3)
    # saving the train dataset
    knn.fit(X_train, y_train)
    # making predictions
    predictions = knn.predict(X_test)
    #prining accuracy
    print("KNN classification accuracy", accuracy(y_test, predictions))