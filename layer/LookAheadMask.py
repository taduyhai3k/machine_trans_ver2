import torch
def look_ahead_mask(x):
    #x is tensor with shape [batch_size, sequence_length]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return (1 - torch.tril(torch.ones(size = [x.shape[0] , x.shape[1], x.shape[1]], device= device)))
def padding_mask(x):
    #x is tensor with shape [batch_size, sequence_length]
    #output is [batch_size, sequence_length, sequence_length]
    return (x.unsqueeze(1).repeat([1, x.shape[1], 1]) == 0) * 1 

def padding_mask_global(x, sequence_length):
    return (x.unsqueeze(1).repeat([1, sequence_length, 1]) == 0) * 1