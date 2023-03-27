'''
Period 1: modify EIS_RUL_GPR.m
models - Regression, SVM, RF
'''
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC

# Regression
class Regression():
    # linear
    def Linear(self, X_train=None, y_train=None):
        lr_model = LinearRegression.fit(X_train, y_train)
        return lr_model

    # polynomial with degree default 5
    def Polynomial(self, X_train=None, y_train=None, degree=5): 
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        poly_model = LinearRegression()
        poly_model.fit(X_train_poly, y_train)
        # Predict
        # X_test_poly = poly.fit_transform(X_test)
        # y_test_pred = model.predict(x_test_poly)
        return poly_model

    # lasso with alpha to control the sparsity of the model, default 0.1
    def Lasso(self, X_train=None, y_train=None,alpha=0.1):
        lasso_model = Lasso(alpha=alpha)
        lasso_model.fit(X_train, y_train)
        # Predict
        # y_test_pred = lasso.predict(x_test_scaled)
        return lasso_model

    def __init__(self,X_train=None, y_train=None, type='', ):
        self.model_name = type
        self.model = None
        if type=='Linear':
            self.model = self.linear(X_train=X_train, y_train=y_train)
        elif type=='Polynomial':
            self.model = self.Polynomial(X_train=X_train, y_train=y_train)
        elif type=='Lasso':
            self.model = self.Lasso(X_train=X_train, y_train=y_train)
        else:
            self.model_name = '' # wrong input


# SVM
class SVMRegression():
    def predict(self, X_test=None, y_test=None):
        if(self.model!=None):
            y_test_pre = self.model.predit(X_test)
            mse = mean_squared_error(y_test, y_test_pre)
        else:
            print('ERROR: get model first!')

    def __init__(self, C=1, kernel='Gaussian', X_train=None, y_train=None):
        self.model = None
        self.model = SVC(kernel='', C=1)
        self.model.fit(X_train, y_train)

# RF
class RandomForest():
    def predict(self, X_test=None, y_test=None):
        if(self.model!=None):
            y_test_pre = self.model.predit(X_test)
            mse = mean_squared_error(y_test, y_test_pre)
        else:
            print('ERROR: get model first!')
    # n_estimators - 
    # max_depth - 
    def __init__(self, n_estimators=100, max_depth=5, random_state=42, X_train=None, y_train=None):
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        rf.fit(X_train, y_train)

if __name__ == '__main__':
    pass