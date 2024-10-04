import torch
import torch.nn as nn
class InpEmbed(nn.Module):
    def __init__(self, vocab_size, dembed = 512) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vocab_size = vocab_size
        self.dembed = dembed
        self.map = nn.Embedding(num_embeddings= self.vocab_size, embedding_dim= self.dembed, padding_idx= 0, dtype= torch.float32, device= self.device)
    def forward(self, x, start = 0):
        # x.shape is [batch, sequence_length index]   
        #one_hot = torch.zeros(size = [x.shape[0], x.shape[1], self.vocab_size], dtype= torch.float32, device= self.device).scatter_(2, x.unsqueeze(2), 1.0)
        return self.map(x) + embed_position(dembed= self.dembed, sequence_length = x.shape[1] ,device= self.device, start= start)
    
def embed_position(dembed = 512, sequence_length = 300, device = 'cpu', start = 0):
    #dembed is 2n
    ep = torch.arange(start= 0 ,end= dembed , step= 1, dtype= torch.float32, device = device).reshape([1, -1]).repeat([sequence_length, 1])    
    ep[:,1::2] = ep[:, 0::2]   
    ep = 1/ (10000**(ep/dembed)) 
    pos = torch.arange(start= 0 + start,end= sequence_length + start, step= 1, dtype= torch.float32, device = device).reshape([1, -1]).transpose(0,1).repeat([1, dembed])
    pos = pos * ep
    pos[:, 1::2] = torch.cos(pos[:, 1::2])
    pos[:, 0::2] = torch.sin(pos[:, 0::2])
    return pos