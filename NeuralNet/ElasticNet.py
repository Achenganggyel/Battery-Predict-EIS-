'''
Period 1: modify EIS_RUL_GPR.m
model - elastic net
'''

# TODO: 网格搜索寻找最优参数
# https://blog.csdn.net/weixin_42163563/article/details/128101784

import numpy as np
from TrainAndPredict import Process
from dataLoader import DataLoader

"""
Elastic Net Regression, which is a kind of linear regression model, 
combined with L1 regularization(L1 正则化) and L2 regularization

@model Parameters:
    model_name: the name of this model, 'ElasticNet'
    config: the value of alpha, l1_ratio, max_iter, and tol
    w: easy to guess
    r: easy to guess
    n_features: the number of features in training dataset
@Network Parameters:
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
"""

class ElasticNet():
    def soft_threshold(self, alpha=0, value=None):
        if value>0 and alpha<abs(value):
            return value-alpha
        elif value<0 and alpha<abs(value):
            return value+alpha
        else:
            return 0 
        
    def loss(self, ):
        pass

    def train(self, X_train=None, y_train=None, l1_ratio=0.5, alpha=0.8, max_iter=1000, tol=1e-4) -> None:
        n_samples, n_features = X_train.shape #393,120
        w = np.zeros(n_features)
        r = y_train - np.dot(X_train, w)
        # iteration
        for _ in range(max_iter):
            w_old = w.copy()

            # TODO：怀疑它生成的不能用，重看吧

            for j in range(n_features):
                X_j = X_train[:, j]
                y_pred = np.dot(X_train, w) + X_j * w[j]
                r = y_train - y_pred
                if l1_ratio == 0:
                    w[j] = self.soft_threshold(alpha=alpha, value = r + np.dot(X_j, w))
                elif l1_ratio == 1:
                    w[j] = np.sign(np.dot(X_j, r)) * max(abs(np.dot(X_j, r)) - alpha, 0) / (np.dot(X_j, X_j) + 1e-8)
                else:
                    w[j] = self.soft_threshold(alpha = l1_ratio * alpha, value = r + np.dot(X_j, w)) / (1 + alpha * (1 - l1_ratio) * np.sum(np.abs(w)))
            
            if np.linalg.norm(w - w_old, ord=2) / np.linalg.norm(w_old, ord=2) < tol:
                break
        
        self.w = w
        self.r = r
        self.n_features = n_features
        
    def predict(self, X_pre=None) -> np.ndarray:
        n_samples, n_features = X_pre.shape
        if(n_features!=self.n_features):
            print('error: the number of features in the test dataset doesn\'t equal to the train.')
        y_pre = np.dot(X_pre, self.w) + self.r
        print('the result of predict:\n',y_pre,'\n\n\n')
        return y_pre

    def test(self, X_test=None, y_test=None) -> None:
        process = Process()
        process.test(modelClass=self, X_test=X_test, y_test=y_test,ML_flag=False)

    def __init__(self, X_train=None, y_train=None, X_test=None, y_test=None,
                 l1_ratio=0.5, alpha=0.8, max_iter=1000, tol=1e-4) -> None:
        self.model_name='ElasticNet'
        self.config = {'l1_ratio':l1_ratio,'alpha':alpha, 'max_iter':max_iter, 'tol':tol}
        # train
        self.train(X_train=X_train, y_train=y_train, l1_ratio=l1_ratio, 
                     alpha=alpha, max_iter=max_iter, tol=tol)
        # test
        self.test(X_test=X_test, y_test=y_test)

        
if __name__=='__main__': # unit test
    # initial dataset: will be split into train and test
    calData_path = r'D:/desktop/StudyPro/Identify/code/Period_1/dataset/EIS_data.txt'
    X_path = r'D:/desktop/StudyPro/Identify/code/Period_1/dataset/EIS_data_RUL.txt'
    y_path = r'D:/desktop/StudyPro/Identify/code/Period_1/dataset/RUL.txt'
    print('---train the model---')
    thisLoader = DataLoader(calData_path=calData_path, X_path=X_path, y_path=y_path)
    # train and test
    EN = ElasticNet(X_train = thisLoader.X_train, y_train=thisLoader.y_train,
                    X_test=thisLoader.X_test, y_test=thisLoader.y_test)
    # predict
    predictData_path = r'D:/desktop/StudyPro/Identify/code/Period_1/dataset/EIS_data_35C02.txt'
    predictData = thisLoader.getData(data_path=predictData_path)
    predictData = thisLoader.normalization(x=predictData, calData=calData_path)
    EN.predict(X=predictData)
    
    print('run the ElasticNet.py successfully!!!')