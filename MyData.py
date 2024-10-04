import pandas as pd
import re
import json
import torch 
from torch.utils.data import Dataset, DataLoader 

# class MyTokenizer():
#     def __init__(self, vocab = None) -> None:
#         self.vocab = vocab
#         if self.vocab is not None:
#             self.invers_vocab = {value: key for key, value in self.vocab.items()}
#         self.pattern = r'([^a-zA-Z0-9])'    
#         self.BOS = '<start>'
#         self.EOS = '<end>'
#         self.prefixes = ['un', 're', 'in', 'im', 'dis', 'en', 'em', 'non', 'over', 'mis', 'sub', 'pre', 'inter', 'fore', 'de', 'trans', 'super', 'semi', 'anti', 'mid', 'under']
#         self.suffixes = ['able', 'ible', 'al', 'ial', 'ed', 'er', 'est', 'ful', 'ic', 'ing', 'ion', 'tion', 'ation', 'ition', 'ity', 'ty', 'ive', 'ative', 'itive', 'less', 'ly', 'ment', 'ness', 'ous', 'eous', 'ious', 'es', 's', 'y']
#     def pre_tokenize(self,w):
#         w = re.sub(self.pattern, r' \1 ', w)         
#         w = w.lower().strip()
#         w = re.sub(r"([?.!,¿])", r" \1 ", w)
#         w = re.sub(r'(\d)', r' \1 ', w)
#         w = re.sub(r'\\n', '', w)
#         w = re.sub(r'\s*:\s*', ' : ', w)
#         w = re.sub(r'(["\'])', r' \1 ', w)    
#         w = re.sub(r'\\', ' ', w)
#         w = re.sub(r'-{2,}', '-', w)  # Thay thế 2 hoặc nhiều dấu gạch ngang liên tiếp bằng 1 dấu gạch ngang
#         w = re.sub(r'-+', ' - ', w)  # Thêm khoảng trắng xung quanh dấu gạch ngang hoặc chuỗi dấu gạch ngang
#         w = re.sub(r'[" "]+', " ", w)
#         w = re.sub(r'\s+', ' ', w)  # Loại bỏ khoảng trắng thừa
#         w = w.strip()
#         # Add start and end token 
#         w = '{} {} {}'.format(self.BOS, w, self.EOS)
#         return w    
    
#     def stemmed_word(self, text):
#         original_word = text
#         prefix = 'none'
#         suffix = 'none'
            
#             # Kiểm tra và tách tiền tố
#         for p in self.prefixes:
#             if text.startswith(p):
#                 prefix = p + '###'
#                 text = text[len(p):]
#                 break
            
#             # Kiểm tra và tách hậu tố
#         for s in self.suffixes:
#             if text.endswith(s):
#                 suffix = '###' + s
#                 text = text[:-len(s)]
#                 break
            
#             # Nếu từ có thể đứng độc lập
#         if text in self.prefixes or text in self.suffixes:
#             text = original_word
#             prefix = 'none'
#             suffix = 'none'
#         return prefix, text, suffix        
    
#     def tokenize(self,w):
#         results = []
#         words = self.pre_tokenize(w)
#         for word in words.split():
#             pre, text, suff = self.stemmed_word(word)
#             if pre != 'none':
#                 results.append(pre)
#             results.append(text)
#             if suff != 'none':
#                 results.append(suff)
#         return ' '.join(results)            
    
#     def encode(self, w):
#         results = []
#         x = self.tokenize(w)
#         for word in x.split(sep = ' '):
#             results.append(self.vocab[word])
#         return torch.tensor(results, dtype = torch.int64, requires_grad= False)
#     def decode(self,tokens):
#         sentence = []
#         i = 0
        
#         while i < len(tokens):
#             token = self.invers_vocab[tokens[i].item()]
            
#             if '###' in token:
#                 if token.startswith('###'):
#                     sentence[-1] = sentence[-1] + token.replace('###', '')
#                 elif token.endswith('###'):
#                     sentence.append(token.replace('###', '') + tokens[i + 1])
#                 else:
#                     sentence.append(token)
#             else:
#                 sentence.append(token)
#             i += 1
#         return ' '.join(sentence)
class MyTokenizer():
    def __init__(self, vocab) -> None:
        self.vocab = vocab
        self.invers_vocab = {value: key for key, value in self.vocab.items()}
        self.BOS = '<start>'
        self.EOS = '<end>'
    def tokenize(self,w):
        w = w.lower().strip()
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        w = re.sub(r'-{2,}', '-', w)  # Thay thế 2 hoặc nhiều dấu gạch ngang liên tiếp bằng 1 dấu gạch ngang
        w = re.sub(r'-+', ' - ', w)  # Thêm khoảng trắng xung quanh dấu gạch ngang hoặc chuỗi dấu gạch ngang
        w = re.sub(r'\s+', ' ', w)  # Loại bỏ khoảng trắng thừa
        w = w.strip()
        # Add start and end token 
        w = '{} {} {}'.format(self.BOS, w, self.EOS)
        return w    
    def encode(self, w):
        results = []
        x = self.tokenize(w)
        for word in x.split(sep = ' '):
            results.append(self.vocab[word])
        return torch.tensor(results, dtype = torch.int64, requires_grad= False)
    def decode(self,w):
        results = []   
        for i in range(len(w)):
            results.append(self.invers_vocab[w[i].item()])
        return ' '.join(results)         
 
    
class EV_Data(Dataset):
    def __init__(self, data_path, E_vocab_path = 'vocab/vocab_E.json' , V_vocab_path = 'vocab/vocab_V.json', inp = 'E', out = 'V', max_length = 300 ) -> None:
        super().__init__()
        self.inp = inp
        self.out = out
        self.data_path = data_path
        self.E_vocab_path = E_vocab_path
        self.V_vocab_path = V_vocab_path
        self.max_length = max_length 
        self.__read_data__()
        self.__read_vocab__()
    
    def __read_data__(self):
        self.data = pd.read_json(self.data_path)
    def __read_vocab__(self):
        if self.inp == 'E':
            with open(self.E_vocab_path, 'r', encoding='utf-8') as f:
                self.inp_vocab = json.load(f)    
            self.inp_tokenizer = MyTokenizer(self.inp_vocab)
            with open(self.V_vocab_path, 'r', encoding='utf-8') as f:
                self.out_vocab = json.load(f)    
            self.out_tokenizer = MyTokenizer(self.out_vocab)        
        else:
            with open(self.V_vocab_path, 'r', encoding='utf-8') as f:
                self.inp_vocab = json.load(f)    
            self.inp_tokenizer = MyTokenizer(self.inp_vocab)
            with open(self.E_vocab_path, 'r', encoding='utf-8') as f:
                self.out_vocab = json.load(f)    
            self.out_tokenizer = MyTokenizer(self.out_vocab)                 
    def __len__(self):
        return len(self.data)        
    def __getitem__(self, index):
        if self.inp == 'E':
            input = self.inp_tokenizer.encode(self.data.iloc[index]['E']) 
            output = self.out_tokenizer.encode(self.data.iloc[index]['V'])
        else:
            input = self.inp_tokenizer.encode(self.data.iloc[index]['V']) 
            output = self.out_tokenizer.encode(self.data.iloc[index]['E'])       
        if len(input)> self.max_length:
            input = input[:self.max_length]
        if len(output)> self.max_length:
            output = output[:self.max_length]                 
        return torch.nn.functional.pad(input, pad = (0, self.max_length - input.shape[0]), value = 0), torch.nn.functional.pad(output, pad = (0, self.max_length - output.shape[0]), value = 0)
    