import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, dmodel = 512, d_ff = 2048, active = 'relu') -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dmodel = dmodel
        self.d_ff = d_ff
        self.linear1 = nn.Linear(in_features=self.dmodel, out_features= self.d_ff, device= self.device, dtype= torch.float32)
        self.linear2 = nn.Linear(in_features=self.d_ff, out_features= self.dmodel, device= self.device, dtype= torch.float32)
        if active == 'relu':
            self.active = nn.ReLU()
        elif active == 'gelu' :
            self.active = nn.GELU()
        else:
            self.active = nn.Sigmoid()       
    
    def forward(self, x):
        out = self.active(self.linear1(x))        
        out = self.linear2(out)
        return out