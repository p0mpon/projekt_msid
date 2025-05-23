import pandas as pd
import numpy as np
from numpy.linalg import pinv

class LinearRegressionClosedForm():
    def __init__(self, regularization=False, alpha=1.0):
        self.coef_ = None
        self.intercept_ = None
        self.regularization = regularization
        self.alpha = alpha

    def fit(self, x, y):
        if isinstance(x, (pd.DataFrame, pd.Series)):
            x = x.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values

        X_with_intercept = np.c_[np.ones(x.shape[0]), x]

        if self.regularization:
            I = np.eye(X_with_intercept.shape[1])
            weights = pinv(X_with_intercept.T.dot(X_with_intercept) + self.alpha * I).dot(X_with_intercept.T).dot(y)            
        else:
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
    def __init__(self, learning_rate=0.01, n_epochs=100, batch_size=64,
                 regularization=None, alpha=1.0, l1_ratio=0.5):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.regularization = regularization
        self.alpha = alpha
        self.l1_ratio = l1_ratio
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
                
                dw = (2/self.batch_size) * x_batch.T.dot(y_pred - y_batch)

                db = (2/self.batch_size) * np.sum(y_pred - y_batch)

                if self.regularization == 'l1':
                    dw += self.alpha * np.sign(self.coef_)
                elif self.regularization == 'l2':
                    dw += self.alpha * 2 * self.coef_
                elif self.regularization == 'elasticnet':
                    dw += self.alpha * (self.l1_ratio * np.sign(self.coef_) + 
                                        (1 - self.l1_ratio) * 2 * self.coef_)

                self.coef_ -= self.learning_rate * dw
                self.intercept_ -= self.learning_rate * db
    
    def predict(self, x):
        if self.coef_ is None or self.intercept_ is None:
            raise Exception("Model is not fitted yet.")
        
        if isinstance(x, (pd.DataFrame, pd.Series)):
            x = x.values
            
        return x.dot(self.coef_) + self.intercept_
    
    
class LogisticRegressionGradientDescent():
    def __init__(self, learning_rate=0.01, n_epochs=100, batch_size=64):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _encode_y(self, y):
        if self.classes_ is None:
            self.classes_ = np.unique(y)
        return (y == self.classes_[1]).astype(int)

    def _decode_y(self, y_pred_binary):
        return np.where(y_pred_binary == 1, self.classes_[1], self.classes_[0])

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        y = self._encode_y(y)

        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0

        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]

                linear_output = X_batch.dot(self.coef_) + self.intercept_
                y_pred = self._sigmoid(linear_output)

                error = y_pred - y_batch
                dw = (1 / self.batch_size) * X_batch.T.dot(error)
                db = (1 / self.batch_size) * np.sum(error)

                self.coef_ -= self.learning_rate * dw
                self.intercept_ -= self.learning_rate * db

    def predict_proba(self, X):
        linear_output = np.array(X).dot(self.coef_) + self.intercept_
        return self._sigmoid(linear_output)

    def predict(self, X, threshold=0.5):
        y_binary = (self.predict_proba(X) >= threshold).astype(int)
        return self._decode_y(y_binary)
