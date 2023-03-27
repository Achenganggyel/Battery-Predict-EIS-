'''
Period 1: modify EIS_RUL_GPR.m
model - elastic net
'''
import numpy as np
# EN
"""
Elastic Net Regression

Parameters:
X: array-like, shape (n_samples, n_features)
    Training data
y: array-like, shape (n_samples,)
    Target values
alpha: float, optional (default=0.5)
    Constant that multiplies the penalty terms
l1_ratio: float, optional (default=0.5)
    The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1.
    For l1_ratio = 0 the penalty is an L2 penalty
    For l1_ratio = 1 the penalty is an L1 penalty
    For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2
max_iter: int, optional (default=1000)
    Maximum number of iterations for the solver
tol: float, optional (default=1e-4)
    Tolerance for stopping criteria

Returns:
coef: array, shape (n_features,)
    Estimated coefficients for the linear regression problem
"""
class ElasticNet():
    def soft_threshold(self, alpha=0.8, ):
        pass

    def process(self, X_train=None, y_train=None, l1_ratio=0.5, alpha=0.8, max_iter=1000, tol=1e-4):
        n_samples, n_features = X_train.shape
        w = np.zeros(n_features)
        r = y_train - np.dot(X_train, w)
        # iteration
        for _ in range(max_iter):
            w_old = w.copy()

            for j in range(n_features):
                X_j = X_train[:, j]
                y_pred = np.dot(X_train, w) + X_j * w[j]
                r = y_train - y_pred
                if l1_ratio == 0:
                    w[j] = self.soft_threshold(alpha, r + np.dot(X_j, w))
                elif l1_ratio == 1:
                    w[j] = np.sign(np.dot(X_j, r)) * max(abs(np.dot(X_j, r)) - alpha, 0) / (np.dot(X_j, X_j) + 1e-8)
                else:
                    w[j] = self.soft_threshold(l1_ratio * alpha, r + np.dot(X_j, w)) / (1 + alpha * (1 - l1_ratio) * np.sum(np.abs(w)))
            
            if np.linalg.norm(w - w_old, ord=2) / np.linalg.norm(w_old, ord=2) < tol:
                break

        self.w = w
        self.r = r
        
    def predict(self, X_test, y_test):
        pass    

    def __init__(self, X_train=None, y_train=None, l1_ratio=0.5, alpha=0.8, max_iter=1000, tol=1e-4) -> None:
        self.w = None
        self.r = None
        self.process(X_train=X_train, y_train=y_train, l1_ratio=l1_ratio, 
                     alpha=alpha, max_iter=max_iter, tol=tol)