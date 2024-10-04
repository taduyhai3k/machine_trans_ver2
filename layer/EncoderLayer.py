import torch
from layer import MultiHeadAtten as MAH
from layer import FeedForward as FF

class EncoderLayer(torch.nn.Module):
    def __init__(self, dembed = 512, dmodel = 512, d_ff = 2048, head = 8, active = 'relu', dropout = 0.1, eps = 1e-5) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dmodel = dmodel
        self.dembed = dembed
        self.d_ff = d_ff
        self.head = head
        self.active = active
        self.dropout = torch.nn.Dropout(p = dropout)
        self.norm1 = torch.nn.LayerNorm(normalized_shape= self.dembed, eps= eps, device= self.device)
        self.norm2 = torch.nn.LayerNorm(normalized_shape= self.dembed, eps= eps, device= self.device)
        self.eps = eps
        self.ff = FF.FeedForward(self.dmodel, self.d_ff, self.active)
        self.mha = MAH.MultiHeadAtten(self.dembed, self.dmodel, self.head) 
        
    def forward(self, x, padding_mask = None, look_ahead_mask = None, padding_global_mask = None ):
        mha_out,_,_ = self.mha(x,x,x, padding_mask)  
        out = self.norm1(x + self.dropout(mha_out))         
        ffo = self.ff(out)
        ffo = self.norm2(out + self.dropout(ffo))
        return ffo