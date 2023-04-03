'''
Period 1: modify EIS_RUL_GPR.m
models - Regression, SVM, RF
@how to use: 
1) init the class, need have a folder 'ML' to save pictures 
   e.g., rf = randomForest(X_train='...)
2) predict a (series of) x,
   e.g., rf.predict(X_pre='...') 
'''
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from dataLoader import DataLoader
from TrainAndPredict import Process
import numpy as np

# activation function - to avoid negative value 
def ReLu(y=None):
    # print('原y值',y) y是一维向量
    length = len(y)
    for i in range(length):
        if y[i] < 0:
            y[i] = 0
    # y = y.flatten() # change the array into the 1-D 
    return y

# Regression
class Regression():
    # predict new data, a bit of rudundancy in the code
    def predict(self, X_pre=None):
        print(self.model_name,'\n')
        if(self.model_name=='Regression_Polynomial'):
            poly = PolynomialFeatures(degree=self.config['degree'])
            X_pre = poly.fit_transform(X_pre)

        y_pre = self.model.predict(X_pre)
        n_samples, n_features = X_pre.shape
        y_pre = ReLu(y=y_pre)
        print('the result of predict:\n',y_pre,'\n\n\n')
        return y_pre

    # linear - method
    def Linear(self, X_train=None, y_train=None):
        lr = LinearRegression()
        lr_model = lr.fit(X=X_train, y=y_train)
        return lr_model

    # polynomial with degree default 3 - method
    def Polynomial(self, X_train=None, y_train=None, degree=3): 
        self.config['degree'] = degree
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        poly_model = LinearRegression()
        poly_model.fit(X_train_poly, y_train)
        return poly_model

    # lasso with alpha to control the sparsity of the model, default 0.1 - method
    def Lasso(self, X_train=None, y_train=None,alpha=0.1):
        lasso_model = Lasso(alpha=alpha)
        lasso_model.fit(X_train, y_train)
        return lasso_model

    def __init__(self,X_train=None, y_train=None,
                 X_test=None, y_test=None, type=''):
        self.model_name = 'Regression_'+type
        self.config={}
        self.model = None
        # train
        if type=='Linear':
            self.model = self.Linear(X_train=X_train, y_train=y_train)
        elif type=='Polynomial':
            self.model = self.Polynomial(X_train=X_train, y_train=y_train)
        elif type=='Lasso':
            self.model = self.Lasso(X_train=X_train, y_train=y_train)
        else:
            print('this method doesn\' exit!')
            return # wrong input
        # test
        process = Process()
        process.test(modelClass=self, X_test=X_test, y_test=y_test,ML_flag=True)

# SVM
class SVMRegression():
    # predict new data
    def predict(self, X_pre=None):
        print(self.model_name,'\n')
        y_pre = self.model.predict(X_pre)
        print('the result of predict:\n',y_pre,'\n\n\n')
        return y_pre

    def __init__(self, C=1, kernel='Gaussian', 
                 X_train=None, y_train=None, X_test=None, y_test=None):
        # train
        self.model_name = 'SVM Regression'
        self.config = {'C':C, 'kernel':kernel}
        self.model = SVC(kernel='rbf', C=1)
        self.model.fit(X_train, y_train)
        # test
        process = Process()
        process.test(modelClass=self, X_test=X_test, y_test=y_test,ML_flag=True)

# RF
class RandomForest():
    # predict new data
    def predict(self, X_pre=None):
        print(self.model_name,'\n')
        y_pre = self.model.predict(X_pre)
        print('the result of predict:\n',y_pre,'\n\n\n')
        return y_pre

    # n_estimators - 
    # max_depth - 
    def __init__(self, n_estimators=100, max_depth=5, random_state=42, 
                 X_train=None, y_train=None, X_test=None, y_test = None):
        self.model_name = 'Random Forest'
        self.config = {'n_estimators':n_estimators,'max_depth':max_depth}
        # train
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        rf.fit(X_train, y_train)
        self.model = rf
        # test
        process = Process()
        process.test(modelClass=self, X_test=X_test, y_test=y_test,ML_flag=True)


if __name__ == '__main__': #单元测试
    # initial dataset: will be split into train and test
    calData_path = r'D:/desktop/StudyPro/Identify/code/Period_1/dataset/EIS_data.txt'
    X_path = r'D:/desktop/StudyPro/Identify/code/Period_1/dataset/EIS_data_RUL.txt'
    y_path = r'D:/desktop/StudyPro/Identify/code/Period_1/dataset/RUL.txt'
    print('---train the model---')
    thisLoader = DataLoader(calData_path=calData_path, X_path=X_path, y_path=y_path)

    reg_linear = Regression(X_train=thisLoader.X_train,y_train=thisLoader.y_train,
                            X_test=thisLoader.X_test, y_test=thisLoader.y_test, type='Linear')
    reg_polynomial = Regression(X_train=thisLoader.X_train,y_train=thisLoader.y_train,
                                X_test=thisLoader.X_test, y_test=thisLoader.y_test, type='Polynomial')
    reg_lasso = Regression(X_train=thisLoader.X_train,y_train=thisLoader.y_train,
                           X_test=thisLoader.X_test, y_test=thisLoader.y_test, type='Lasso')
    svm = SVMRegression(X_train=thisLoader.X_train,y_train=thisLoader.y_train,
                        X_test=thisLoader.X_test, y_test=thisLoader.y_test)
    rf = RandomForest(X_train=thisLoader.X_train,y_train=thisLoader.y_train,
                      X_test=thisLoader.X_test, y_test=thisLoader.y_test)

    # predict dataset
    print('---predict data---')
    predictData_path = r'D:/desktop/StudyPro/Identify/code/Period_1/dataset/EIS_data_35C02.txt'
    predictData = thisLoader.getData(data_path=predictData_path)
    predictData = thisLoader.normalization(x=predictData, calData=thisLoader.calData)

    reg_linear_pre = reg_linear.predict(X_pre=predictData)
    reg_poly_pre = reg_polynomial.predict(X_pre=predictData)
    reg_lasso_pre = reg_lasso.predict(X_pre=predictData)
    svm_pre = svm.predict(X_pre=predictData)
    rf_pre = rf.predict(X_pre=predictData)

    # compare the different predict values
    # set config
    _drawConfig = [['Linear Regression','Polynomial Regression','Lasso Regression',
               'SVM (Guassian Kernel)','Random Forest'],
               ['.','o','*','s','D'],
               ['#f72585','#7209b7','#3a0ca3','#4361ee','#4cc9f0'],
               ['solid','dashed','dashdot','dashed','dashdot']]
               
    y_set = (reg_linear_pre, reg_poly_pre, reg_lasso_pre, svm_pre, rf_pre)
    
    # save results
    for i, y_each in enumerate(y_set):
        txt_save_path = 'D:\\desktop\\StudyPro\\Identify\\code\\Period_1\\predict_1\\ML\\'+_drawConfig[0][i]+'.txt'
        with open(txt_save_path,'w') as f:
            for val in y_each:
                f.write(str(val) + '\n')
            f.close()

    process = Process()
    process.comparsion(_drawConfig, True, y_set, 'predicted RUL') 
    print('run the ML.py successfully!!!')