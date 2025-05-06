import pandas as pd
import numpy as np
from numpy.linalg import pinv

class LinearRegressionClosedForm():
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, x, y):
        if isinstance(x, (pd.DataFrame, pd.Series)):
            x = x.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values

        X_with_intercept = np.c_[np.ones(x.shape[0]), x]

        # W = (X^T * X)^-1 * X^T * Y
        # using pinv instead of inv for possibility of singular matrices
        weights = pinv(X_with_intercept.T.dot(X_with_intercept)).dot(X_with_intercept.T).dot(y)

        self.coef_ = weights[1:]
        self.intercept_ = weights[0]
        
        return self
    
    def predict(self, x):
        if self.coef_ is None or self.intercept_ is None:
            raise Exception("Model is not fitted yet.")
        
        if isinstance(x, (pd.DataFrame, pd.Series)):
            x = x.values
            
        return x.dot(self.coef_) + self.intercept_
    

class LinearRegressionGradientDescent():
    def __init__(self, learning_rate=0.01, n_epochs=100, batch_size=64):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, x, y):
        if isinstance(x, (pd.DataFrame, pd.Series)):
            x = x.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values

        n_samples, n_features = x.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0
        
        for _ in range(self.n_epochs):
            indices = np.random.permutation(n_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]
            
            for i in range(0, n_samples, self.batch_size):
                x_batch = x_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                y_pred = x_batch.dot(self.coef_) + self.intercept_
                
                # derivative of the weights
                # dL/dw = 1/m * X^T * (X * W - Y)
                # where:
                # m = batch size
                # X^T = transposed X
                # X * W = Y^ = predicted data
                # Y = actual data
                dw = (1/self.batch_size) * x_batch.T.dot(y_pred - y_batch)

                # derivative of the bias
                # dL/db = 1/m * ∑(X * W - Y)
                # where:
                # m = batch size
                # ∑(X * W - Y) sum of the matrix
                # X * W = Y^ = predicted data
                # Y = actual data
                db = (1/self.batch_size) * np.sum(y_pred - y_batch)

                self.coef_ -= self.learning_rate * dw
                self.intercept_ -= self.learning_rate * db
    
    def predict(self, x):
        if self.coef_ is None or self.intercept_ is None:
            raise Exception("Model is not fitted yet.")
        
        if isinstance(x, (pd.DataFrame, pd.Series)):
            x = x.values
            
        return x.dot(self.coef_) + self.intercept_
    

def mean_squared_error(y_true, y_pred):
    m = len(y_pred)
    return np.sum((y_pred - y_true) ** 2) / m