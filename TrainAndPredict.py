'''

'''

import dataLoader
from sklearn.metrics import mean_squared_error

class Process():
    # In ML ,model is a model; In RNN and EN and so on, model is a class
    def getModel(self, model_name='', model=None): 
        self.model_name = model_name
        self.model = model

    def drawPlot(self, ):
        pass

    # TODO: find enough validation method    
    def validation(self,):
        pass

    def predict(self, X_test, y_test, y_test_pre):
        self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_test_pre)
        


if __name__ == '__main__':
    calData_path = r'D:/desktop/StudyPro/Identify/code/Period_1/dataset/EIS_data.txt'
    X_path = r'D:/desktop/StudyPro/Identify/code/Period_1/dataset/EIS_data_RUL.txt'
    y_path = r'D:/desktop/StudyPro/Identify/code/Period_1/dataset/RUL.txt'

    LoadData = dataLoader.DataLoader(calData_path=calData_path, X_path=X_path, y_path=y_path)
    X = LoadData.X