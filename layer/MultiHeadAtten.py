import torch
import torch.nn as nn
class MultiHeadAtten(nn.Module):
    def __init__(self, dembed, dmodel = 512, head = 8) -> None:
        super().__init__()
        self.dmodel = dmodel
        self.dembed = dembed
        self.head = head
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.q = nn.Linear(in_features= self.dembed, out_features= self.dmodel, dtype= torch.float32, device= self.device)
        self.k = nn.Linear(in_features= self.dembed, out_features= self.dmodel, dtype= torch.float32, device= self.device)
        self.v = nn.Linear(in_features= self.dembed, out_features= self.dmodel, dtype= torch.float32, device= self.device)
        self.o = nn.Linear(in_features= self.dmodel, out_features= self.dembed, dtype= torch.float32, device= self.device)
    
    def splitting_head(self,x):
        #shape x is [batch_size,length, dmodel]
        #output shape is [batch_size, head, length, dmodel // head]
        batch_size = x.shape[0]
        length = x.shape[1]
        dmodel = x.shape[2]
        h_dv = dmodel // self.head 
        return x.transpose(1,2).reshape(batch_size, self.head, h_dv, length).transpose(2,3) 
    
    def scaled_dot_product(self, q, k, v, mask = None):
        dk = k.shape[-1]
        head = q.shape[1]
        attention_score = torch.matmul(q, k.transpose(2,3)) / torch.math.sqrt(dk)   
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1,head,1,1)
            attention_score += mask * -1e30
        attention_weight = attention_score.softmax(dim = -1)
        out = torch.matmul(attention_weight, v)
        return out, attention_weight    
    
    def forward(self, q, k, v, mask = None, inference_mode = False, k_cache = None, v_cache = None):
        # q, k, v are same dimension if inference_mode is True
        qw = self.q(q)
        kw = self.k(k)
        vw = self.v(v)
        head_q = self.splitting_head(qw)
        head_k = self.splitting_head(kw)
        head_v = self.splitting_head(vw)
        if inference_mode:
            if k_cache is not None:
                k_cache[:, :, -1, :] = head_k.squeeze(dim = 2)
                head_k = k_cache
            if v_cache is not None:    
                v_cache[:, :, -1, :] = head_v.squeeze(dim = 2)
                head_v = v_cache
        out, attention_weight = self.scaled_dot_product(head_q, head_k,head_v, mask)
        out = out.transpose(2,3).reshape([q.shape[0], self.dmodel, q.shape[1]]).transpose(1,2)
        final = self.o(out)
        if inference_mode:
            return final, head_k, head_v
        else:
            return final, attention_weight, None
         