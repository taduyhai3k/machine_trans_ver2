import torch
import torch.nn as nn
import utils
from layer import Encoder, Decoder, InputEmbed, LookAheadMask, Spe


class Transformer(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, dmodel = 512, dembed = 512,d_ff = 2048,head = 8, active = 'relu', layer = 6, dropout = 0.1, eps = 1e-5, tying = False) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = Encoder.Encoder(dmodel = dmodel, dembed = dembed,d_ff = d_ff,head = head, active = 'relu', layer = layer, dropout = dropout, eps = eps)
        self.decoder = Decoder.Decoder(dmodel = dmodel, dembed = dembed,d_ff = d_ff,head = head, active = 'relu', layer = layer, dropout = dropout, eps = eps)
        self.inp_embed = InputEmbed.InpEmbed(input_vocab_size, dembed)
        self.out_embed = InputEmbed.InpEmbed(output_vocab_size, dembed)
        self.layer = layer
        self.tying = tying
        if not self.tying:
            self.linear = nn.Linear(in_features= dmodel, out_features= output_vocab_size, device = self.device, dtype = torch.float32)
        self.dropout = nn.Dropout(p = dropout)
        self.dmodel = dmodel
    
    def forward(self, x, y):
        inp_embed = self.dropout(self.inp_embed(x))
        padding_mask_enc = LookAheadMask.padding_mask(x)
        encoder_out = self.encoder(inp_embed, padding_mask_enc)
        out_embed = self.dropout(self.out_embed(y))
        look_ahead_mask = LookAheadMask.look_ahead_mask(y)
        padding_global_mask = LookAheadMask.padding_mask_global(x, y.shape[1]) 
        decoder_out, _, _ = self.decoder(out_embed, encoder_out, None, look_ahead_mask, padding_global_mask)
        if self.tying:
            return torch.matmul(decoder_out ,self.out_embed.map.weight.t())
        else:
            return self.linear(decoder_out)   
    def inference(self, x, id_token_end, max_length = 300):
        #x is [batch, sequence_length]
        is_complete = torch.tensor([0 for i in range(x.shape[0])])
        indices = torch.tensor([i for i in range(x.shape[0])], dtype= torch.int64)
        indices_cache = torch.tensor([i for i in range(x.shape[0])])
        results = [torch.tensor([1]) for i in range(x.shape[0])]
        with torch.no_grad():
            inp_embed = self.dropout(self.inp_embed(x))
            padding_mask_enc = LookAheadMask.padding_mask(x)
            encoder_out = self.encoder(inp_embed, padding_mask_enc)
            out_embed = self.dropout(self.out_embed(torch.tensor([1], dtype= torch.int64).repeat([x.shape[0], 1])))                  
            k_caches = None
            v_caches = None
            for j in range(max_length):  
                padding_global_mask = LookAheadMask.padding_mask_global(x, 1)
                decoder_out, k_caches, v_caches = self.decoder(out_embed, encoder_out, None, None, padding_global_mask, True, k_caches, v_caches)
                if self.tying:
                    out = torch.matmul(decoder_out ,self.out_embed.map.weight.t())
                else:
                    out = self.linear(decoder_out)
                out = torch.argmax(out, dim = -1, keepdim= False).type(torch.int64)
                indices_tmp = torch.tensor([], dtype= torch.int64)
                indices_out = torch.tensor([], dtype= torch.int64)
                k_caches_tmp = torch.tensor([], dtype= torch.float32 )
                v_caches_tmp = torch.tensor([], dtype= torch.float32 )
                encoder_out_tmp = torch.tensor([], dtype= torch.float32)
                x_tmp = torch.tensor([], dtype = torch.int64)
                for i in range(out.shape[0]):
                    if is_complete[indices[i]] == 0:
                        results[indices[i]] = torch.cat([results[indices[i]], out[i]], dim = 0)
                    if out[i] == id_token_end:
                        is_complete[indices[i]] = 1    
                    else:
                        indices_tmp = torch.cat([indices_tmp, torch.tensor([indices[i]], dtype= torch.int64)], dim = 0)    
                        indices_out = torch.cat([indices_out, torch.tensor([i], dtype = torch.int64)], dim = 0)
                        k_caches_tmp = torch.cat([k_caches_tmp, k_caches[i,:,:,:,:].unsqueeze(0)], dim = 0)
                        v_caches_tmp = torch.cat([v_caches_tmp, v_caches[i,:,:,:,:].unsqueeze(0)], dim = 0)
                        encoder_out_tmp = torch.cat([encoder_out_tmp, encoder_out[i,:,:].unsqueeze(0)], dim = 0)
                        x_tmp = torch.cat([x_tmp, x[i].unsqueeze(0)], dim = 0)
                indices = indices_tmp
                x = x_tmp
                encoder_out = encoder_out_tmp
                k_caches = k_caches_tmp
                v_caches = v_caches_tmp      
                if len(indices_out) == 0:
                    break  
                out = torch.cat([out[indices_out[i]] for i in range(len(indices_out))], dim = 0).unsqueeze(-1)    
                if out.shape[0] == 0:
                    break        
                out_embed = self.dropout(self.out_embed(out, start = j + 1))
            
            return results                                
        
class TransformerParallel(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, dmodel = 512, dembed = 512,d_ff = 2048,head = 8, active = 'relu', layer = 6, dropout = 0.1, eps = 1e-5) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.SpeE = Spe.Spe(dmodel = dmodel, dembed = dembed,d_ff = d_ff,head = head, active = 'relu', layer = layer, dropout = dropout, eps = eps)
        self.SpeV = Spe.Spe(dmodel = dmodel, dembed = dembed,d_ff = d_ff,head = head, active = 'relu', layer = layer, dropout = dropout, eps = eps)
        self.E_embed = InputEmbed.InpEmbed(input_vocab_size, dembed)
        self.V_embed = InputEmbed.InpEmbed(output_vocab_size, dembed)
        self.dropout = nn.Dropout(p = dropout)
        self.dmodel = dmodel            
        
    def forward(self, x, y, inp = 'E'):
        if inp == 'E':
            inp_embed = self.dropout(self.E_embed(x))
            padding_mask_enc = LookAheadMask.padding_mask(x)
            encoder_out,_,_ = self.SpeE(inp_embed, is_encode = True, padding_mask = padding_mask_enc)
            out_embed = self.dropout(self.V_embed(y))
            look_ahead_mask = LookAheadMask.look_ahead_mask(y)
            padding_global_mask = LookAheadMask.padding_mask_global(x, y.shape[1]) 
            decoder_out,_,_ = self.SpeV(out_embed, False ,encoder_out, None, look_ahead_mask, padding_global_mask)
            return torch.matmul(decoder_out ,self.V_embed.map.weight.t())
        else:
            inp_embed = self.dropout(self.V_embed(x))
            padding_mask_enc = LookAheadMask.padding_mask(x)
            encoder_out,_,_ = self.SpeV(inp_embed, is_encode = True, padding_mask = padding_mask_enc)
            out_embed = self.dropout(self.E_embed(y))
            look_ahead_mask = LookAheadMask.look_ahead_mask(y)
            padding_global_mask = LookAheadMask.padding_mask_global(x, y.shape[1]) 
            decoder_out,_,_ = self.SpeE(out_embed, False ,encoder_out, None, look_ahead_mask, padding_global_mask)
            return torch.matmul(decoder_out ,self.E_embed.map.weight.t())                  
    
    def inference(self, x, id_token_end, inp = 'E', max_length = 300):
        is_complete = torch.tensor([0 for i in range(x.shape[0])])
        indices = torch.tensor([i for i in range(x.shape[0])], dtype= torch.int64)
        results = [torch.tensor([1]) for i in range(x.shape[0])]             
        with torch.no_grad():
            if inp == 'E':
                inp_embed = self.dropout(self.E_embed(x))
                padding_mask_enc = LookAheadMask.padding_mask(x)
                encoder_out,_,_ = self.SpeE(inp_embed, is_encode = True, padding_mask = padding_mask_enc)
                out_embed = self.dropout(self.V_embed(torch.tensor([1], dtype= torch.int64).repeat([x.shape[0], 1])))                  
                k_caches = None
                v_caches = None
                for j in range(max_length):  
                    padding_global_mask = LookAheadMask.padding_mask_global(x, 1)
                    decoder_out, k_caches, v_caches = self.SpeV(out_embed, False, encoder_out, None, None, padding_global_mask, True, k_caches, v_caches)
                    out = torch.matmul(decoder_out ,self.V_embed.map.weight.t())
                    out = torch.argmax(out, dim = -1, keepdim= False).type(torch.int64)
                    indices_tmp = torch.tensor([], dtype= torch.int64)
                    indices_out = torch.tensor([], dtype= torch.int64)
                    k_caches_tmp = torch.tensor([], dtype= torch.float32 )
                    v_caches_tmp = torch.tensor([], dtype= torch.float32 )
                    encoder_out_tmp = torch.tensor([], dtype= torch.float32)
                    x_tmp = torch.tensor([], dtype = torch.int64)
                    for i in range(out.shape[0]):
                        if is_complete[indices[i]] == 0:
                            results[indices[i]] = torch.cat([results[indices[i]], out[i]], dim = 0)
                        if out[i] == id_token_end:
                            is_complete[indices[i]] = 1    
                        else:
                            indices_tmp = torch.cat([indices_tmp, torch.tensor([indices[i]], dtype= torch.int64)], dim = 0)    
                            indices_out = torch.cat([indices_out, torch.tensor([i], dtype = torch.int64)], dim = 0)
                            k_caches_tmp = torch.cat([k_caches_tmp, k_caches[i,:,:,:,:].unsqueeze(0)], dim = 0)
                            v_caches_tmp = torch.cat([v_caches_tmp, v_caches[i,:,:,:,:].unsqueeze(0)], dim = 0)
                            encoder_out_tmp = torch.cat([encoder_out_tmp, encoder_out[i,:,:].unsqueeze(0)], dim = 0)
                            x_tmp = torch.cat([x_tmp, x[i].unsqueeze(0)], dim = 0)
                    indices = indices_tmp
                    x = x_tmp
                    encoder_out = encoder_out_tmp
                    k_caches = k_caches_tmp
                    v_caches = v_caches_tmp      
                    if len(indices_out) == 0:
                        break  
                    out = torch.cat([out[indices_out[i]] for i in range(len(indices_out))], dim = 0).unsqueeze(-1)    
                    if out.shape[0] == 0:
                        break        
                    out_embed = self.dropout(self.V_embed(out, start = j + 1))       
            else:
                inp_embed = self.dropout(self.V_embed(x))
                padding_mask_enc = LookAheadMask.padding_mask(x)
                encoder_out,_,_ = self.SpeV(inp_embed, is_encode = True, padding_mask = padding_mask_enc)
                out_embed = self.dropout(self.E_embed(torch.tensor([1], dtype= torch.int64).repeat([x.shape[0], 1])))                  
                k_caches = None
                v_caches = None
                for j in range(max_length):  
                    padding_global_mask = LookAheadMask.padding_mask_global(x, 1)
                    decoder_out, k_caches, v_caches = self.SpeE(out_embed, False, encoder_out, None, None, padding_global_mask, True, k_caches, v_caches)
                    out = torch.matmul(decoder_out ,self.E_embed.map.weight.t())
                    out = torch.argmax(out, dim = -1, keepdim= False).type(torch.int64)
                    indices_tmp = torch.tensor([], dtype= torch.int64)
                    indices_out = torch.tensor([], dtype= torch.int64)
                    k_caches_tmp = torch.tensor([], dtype= torch.float32 )
                    v_caches_tmp = torch.tensor([], dtype= torch.float32 )
                    encoder_out_tmp = torch.tensor([], dtype= torch.float32)
                    x_tmp = torch.tensor([], dtype = torch.int64)
                    for i in range(out.shape[0]):
                        if is_complete[indices[i]] == 0:
                            results[indices[i]] = torch.cat([results[indices[i]], out[i]], dim = 0)
                        if out[i] == id_token_end:
                            is_complete[indices[i]] = 1    
                        else:
                            indices_tmp = torch.cat([indices_tmp, torch.tensor([indices[i]], dtype= torch.int64)], dim = 0)    
                            indices_out = torch.cat([indices_out, torch.tensor([i], dtype = torch.int64)], dim = 0)
                            k_caches_tmp = torch.cat([k_caches_tmp, k_caches[i,:,:,:,:].unsqueeze(0)], dim = 0)
                            v_caches_tmp = torch.cat([v_caches_tmp, v_caches[i,:,:,:,:].unsqueeze(0)], dim = 0)
                            encoder_out_tmp = torch.cat([encoder_out_tmp, encoder_out[i,:,:].unsqueeze(0)], dim = 0)
                            x_tmp = torch.cat([x_tmp, x[i].unsqueeze(0)], dim = 0)
                    indices = indices_tmp
                    x = x_tmp
                    encoder_out = encoder_out_tmp
                    k_caches = k_caches_tmp
                    v_caches = v_caches_tmp      
                    if len(indices_out) == 0:
                        break  
                    out = torch.cat([out[indices_out[i]] for i in range(len(indices_out))], dim = 0).unsqueeze(-1)   
                    if out.shape[0] == 0:
                        break        
                    out_embed = self.dropout(self.E_embed(out, start = j + 1))                   
        return results                         
    
