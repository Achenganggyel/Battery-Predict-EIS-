'''
reference: rnn_bi_multilayer_lstm_own_csn_agnews.ipynb
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# RNN
class RNN(nn.Module):
    def loss(self, ):
        return 

    def forward(self, input=None, hidden=None) -> None:
        embedded = self.dropout(self.embedding(input))
        # packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_length)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        hidden = self.fc1(hidden)
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)
        return hidden

    '''
    @params
        input_dim 数据集的行数
    '''
    def __init__(self, hidden_dim=64, input_dim=0, 
                 output_dim=1, embedding_dim=0,
                num_layers=2, bidirectional=True,
                dropout=0.5, pad_idx=0) -> None:
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
        self.hidden_size = hidden_dim
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=num_layers,
                           bidirectional=bidirectional, 
                           dropout=dropout)
        
        self.fc1 = nn.Linear(hidden_dim * num_layers, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU(dim=1) # elu activate function

        self.model_lr = 1e-3
        self.model_batch_size = input_dim/4 # XXX if there is problem
        self.model_num_epochs = 100

        self.save_hyperparameters()