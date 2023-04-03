import torch
import torch.nn as nn
import torch.nn.functional as F
from RNN import RNN
import sys
sys.path.append('../')
from TrainAndPredict import Process

class useModel():
    # for predict dataset and train's test
    def predict(self, X=None, y=None, isTest=True, save_path = r"\\predict_NN\\my\\"):
        # NOTE 在评估模式下，batchNorm层，dropout层等用于优化训练而添加的网络层会被关闭，从而使得评估时不会发生偏移。
        self.train(False)
        y_pre = None

        if isTest:
            # get the 
            process = Process()
            self.valid = Process.validation(y_test=y, y_test_pre=y_pre)
            process.visualization()
        else:
            # save results
            txt_save_path = 'D:\\desktop\\StudyPro\\Identify\\code\\Period_1+'+save_path+'.txt'
            with open(txt_save_path,'w') as f:
                for val in y_pre:
                    f.write(str(val) + '\n')
                f.close()



    # for train dataset, including train and test
    def train(self, train_loader=None, input=None, test_loader=None):
        for epoch in range(self.num_epochs):
            self.train(True)
            for batch_idx, batch_data in enumerate(train_loader):
                # forward and back prop
                logits = self(input, len(input)) #TODO 修改此处
                cost = F.cross_entropy(logits, batch_data.classlabel.long())
                self.optimizer.zero_grad()
                
                cost.backward()
        # update model params


        self.predict(X=test_loader, )

    def __init__(self, modelClass=RNN, seed=42) -> None:
        torch.manager_seed(seed)
        # config
        model = modelClass(input_dim=368, )
        model = model.to(self.device)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = model.model_num_epochs
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.model_lr)
       
        # get the data
        # instantion of the RNN
        


if __name__ == '__main__':
    pass