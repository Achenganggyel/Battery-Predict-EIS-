'''
Methods used in both ML and neural networks when training and predicting. 
No unit test for this .py file.
'''
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
import numpy as np
import pdb

class Process():
    # draw the visualization picture
    def visualization(self, y_test=None, y_test_pre=None, 
                      xlabel='', ylabel='', pic_save_path=''):
        # the format of y_test and y_test_pre are both np.ndarray
            
        plt.figure(1)
        # draw the plot
        plt.scatter(x=y_test, y=y_test_pre, 
                    marker='^',c='#447ED9', label='predict value') #bo：蓝色圆点
        
        plt.plot(y_test, y_test_pre, c='#539EE0', linewidth=3, alpha=0.6)
        # set the label of x and y axis
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(pic_save_path)
        plt.show()

    # HACK 修改了返回格式
    # model results evaluation 
    def validation(self, y_test=None, y_test_pre=None) -> dict:
        evalution={} # initialize the dictionary
        mse = mean_squared_error(y_test, y_test_pre)
        evalution['mse']=mse
        mae = median_absolute_error(y_test, y_test_pre)
        evalution['mae']=mae
        mape = mean_absolute_percentage_error(y_test, y_test_pre)
        evalution['mape']=mape
        r2 = r2_score(y_test, y_test_pre)
        evalution['r2']=r2

        for key, value in evalution.items():
            print(key,':',value)
        print('\n\n')
        return evalution

    '''
    for test dataset
    @params
        modelClass-class
        ML_flag: if ML_flag is True, used in ML.py; otherwise, neural networks
    '''
    def test(self, X_test=None, y_test=None, modelClass=None, ML_flag=False):
        print(modelClass.model_name,'\n',modelClass.config,'\n\n')
        if ML_flag:
            subFolder='ML/'
        else:
            subFolder='NeuralNets/'
        
        y_test_pre = modelClass.predict(X_pre = X_test)
        # method to evaluate the model's result
        self.validation(y_test=y_test, y_test_pre=y_test_pre)
        # draw the plot to visualize true and pred
        pic_save_path='./pic/'+subFolder+str(modelClass.model_name)+'.jpg'
        self.visualization(y_test=y_test, y_test_pre=y_test_pre,
                            pic_save_path=pic_save_path,
                            xlabel='Actual Rul',ylabel='Predicted RUL')
    '''
    for predict dataset
    @params
        drawConfig(4 rows): 
            1st row-the name of this series
            2nd row-the shape of dots
            3rd row-the colors of dots
            4th-the format of the line
        ML_flag
        y: the tuple of y values
    '''
    def comparsion(self, drawConfig, ML_flag, y, ylabel)->None: # 很奇怪，这个函数不能用默认值..
        print('config of comparsion fig:\n',drawConfig)
        drawConfig_rows = len(drawConfig)
        if drawConfig_rows!=4:
            print('error config for the comparsion fig!')
            return 
        
        plt.figure(1)
        x = np.arange(1, len(y[0])+1)

        for i,y_each in enumerate(y):
            plt.scatter(x=x, y=y_each,
                    marker=drawConfig[1][i],c=drawConfig[2][i], 
                    label=drawConfig[0][i]) 
            plt.plot(x, y_each,
                    c=drawConfig[2][i], alpha=0.6,
                    linestyle=drawConfig[3][i]) 
        plt.legend(loc="best")

        plt.xlabel('number')
        plt.ylabel(ylabel)

        if ML_flag==True:
            subFolder='ML/'
        else:
            subFolder='NeuralNets/'

        pic_save_path='./pic/'+subFolder+'comparsion_y_pred.jpg'
        plt.savefig(pic_save_path)
        plt.show()

if __name__ == '__main__':
    pass