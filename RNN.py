import torch
import torch.nn as nn
import torch.nn.functional as F

# RNN
class RNN(nn.Module):
    def loss(self, ):
        pass

    def forward(self, input=None, hidden=None) -> None:
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1,self.hidden_size)
    
    def predict(self, y_test=None, y_test_pre=None):
        pass

    def __init__(self, hidden_size=1, input_size=0, output_size=0) -> None:
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        # TODO: to refine
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)


if __name__ == '__main__':
    RNN()