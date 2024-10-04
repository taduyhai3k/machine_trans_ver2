import torch
import torch.nn as nn
from layer import DecoderLayer

class Decoder(nn.Module):
    def __init__(self, dmodel = 512, dembed = 512,d_ff = 2048,head = 8, active = 'relu', layer = 6, dropout = 0.1, eps = 1e-5) -> None:
        super().__init__()
        self.dembed = dembed
        self.dmodel = dmodel
        self.active = active
        self.layer = layer
        self.drop_rate = dropout
        self.eps = eps
        self.d_ff = d_ff
        self.head = head
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.decoder_layer = nn.ModuleList([DecoderLayer.DecoderLayer(self.dembed, self.dmodel,self.d_ff, self.head, self.active, self.drop_rate, self.eps ) for i in range(self.layer)])
    
    def forward(self, x, encoder_out, padding_mask = None, look_ahead_mask = None, padding_global_mask = None, inference_mode = False, k_cache = None, v_cache = None):
        #cache is [batch_size, layer, head, sequence_length, dmodel // head]
        out = x
        is_First = False
        if inference_mode:
            if k_cache is None:
                is_First = True
                k_cache = torch.rand(x.shape[0], self.layer, self.head, 1, self.dmodel // self.head, device=self.device)
            else:
                k_cache = torch.cat([k_cache, torch.rand(size = [x.shape[0], self.layer, self.head, 1, self.dmodel // self.head], 
                                                         dtype = torch.float32, device = self.device)], dim = 3)    
            if v_cache is None:
                is_First = True 
                v_cache = torch.rand(x.shape[0], self.layer, self.head, 1, self.dmodel // self.head, device=self.device)  
            else:
                v_cache = torch.cat([v_cache, torch.rand(size = [x.shape[0], self.layer, self.head, 1, self.dmodel // self.head], 
                                                         dtype = torch.float32, device = self.device)], dim = 3)                     
                
        for i in range(self.layer):
            if inference_mode:
                if is_First:
                    out, k_cache[:, i, :, :, :], v_cache[:, i,:, :, :] = self.decoder_layer[i](out, encoder_out, padding_mask, look_ahead_mask, padding_global_mask, 
                                                           inference_mode, None, None)
                else:
                    out, k_cache[:, i, :, :, :], v_cache[:, i,:, :, :] = self.decoder_layer[i](out, encoder_out, padding_mask, look_ahead_mask, padding_global_mask, 
                                                           inference_mode, k_cache[:, i, :, :, :], v_cache[:, i, :, :, :])                        
            else:    
                out,_,_  = self.decoder_layer[i](out,encoder_out,padding_mask, look_ahead_mask, padding_global_mask)        
        return out, k_cache, v_cache    