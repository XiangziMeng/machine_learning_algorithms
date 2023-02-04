# coding: utf-8

import numpy as np

class LogisticRegression(object):
    def __init__(self):
        self.n_samples = None
        self.n_features = None
        self.w = None
        self.b = None

    def parameter_initialization(self):
        self.w = np.zeros(self.n_features)
        self.b = 0.0
        if 0:
            self.w = np.random.uniform(low=-1, high=1, size=self.n_features)
            self.b = np.random.uniform(low=-1, high=1, size=1)

    def fit(self, X, y, n_iterations=100):
        '''
        X: (n_samples, n_features)
        y: (n_samples,)
        '''
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.parameter_initialization()
        self.optimize(X, y, n_iterations)

    def predict(self, X):
        res = []
        output = X.dot(self.w) + self.b
        probs = self.sigmoid(output)
        for prob in probs:
            if prob >= 0.5:
                res.append(1)
            else:
                res.append(0)
        return np.array(res)

    def sigmoid(self, output):
        res = []
        for out in output:
            res.append(1.0 / (1.0 + np.exp(-out)))
        return np.array(res)

    def optimize(self, X, y, n_iterations):
        for _ in range(n_iterations):
            output = X.dot(self.w) + self.b
            y_pred = self.sigmoid(output)
            w_1 = -np.sum([X[k] * (y[k] - y_pred[k]) for k in range(self.n_samples)], axis=0)
            w_2 = np.sum([X[k].dot(X[k].T) * y_pred[k] * (1 - y_pred[k]) for k in range(self.n_samples)])
            self.w -= w_1 / w_2

            b_1 = np.sum([y_pred[k] - y[k] for k in range(self.n_samples)], axis=0)
            b_2 = np.sum([y_pred[k] * (1 - y_pred[k]) for k in range(self.n_samples)])
            self.b -= b_1 / b_2
            #print(b_1, b_2)
            #print(self.training_acc(X, y))

    def training_acc(self, X, y):
        res = 0
        y_pred = self.predict(X)
        for i in range(self.n_samples):
            if y_pred[i] == y[i]:
                res += 1
        res = res * 1.0 / self.n_samples
        return res

    def debug(self, X, y):
        res = 0
        y_pred = self.predict(X)
        for i in range(10):
            print(y_pred[i], y[i])  

    def print_parameters(self):
        print(self.w)
        print(self.b)    

