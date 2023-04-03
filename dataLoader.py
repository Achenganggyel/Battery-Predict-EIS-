import numpy as np 
from sklearn.model_selection import train_test_split

'''
DataLoader
@params:
    data: the data loaded and prepocessed
'''
class DataLoader():
    def getData(self, data_path='') -> np.ndarray:
        data = np.loadtxt(data_path)
        print('The data is in np.ndarry format: ',isinstance(data, np.ndarray))
        print('get target data successfully!')
        return data

    def normalization(self, x=None, calData=None)-> np.ndarray: #归一化！不是正则化
        mean = np.mean(calData,axis=0)
        std = np.std(calData,axis=0)
        x = (x-mean)/std
        return x

    def splitDataset(self, test_size=0.2, rs_each=42):
        train_X,test_X,train_y,test_y = train_test_split(self.X,self.y)
        self.X_train = train_X
        self.X_test = test_X
        self.y_train = train_y
        self.y_test = test_y
        print('split the data successfully!\n')

    '''
    @args:
        data_path: the absolute path of data, named by '.txt'
    '''
    def __init__(self, calData_path='', X_path='',y_path='', test_size = 0.2, rs_each=42) -> None:
        # data to calculate mean and std
        calData = self.getData(data_path=calData_path)
        # X and y
        self.X = self.getData(data_path=X_path)
        self.y = self.getData(data_path=y_path)
        # normalization
        self.X = self.normalization(x=self.X,calData=calData)
        # split the dataset
        self.splitDataset(test_size=test_size, rs_each=rs_each)
        
        self.calData = calData

if __name__== '__main__':
    calData_path = r'D:/desktop/StudyPro/Identify/code/Period_1/dataset/EIS_data.txt'
    X_path = r'D:/desktop/StudyPro/Identify/code/Period_1/dataset/EIS_data_RUL.txt'
    y_path = r'D:/desktop/StudyPro/Identify/code/Period_1/dataset/RUL.txt'

    DataLoader(calData_path=calData_path, X_path=X_path, y_path=y_path)
    print('run the dataLoader.py successfully!!!')