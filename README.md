# Machine Translation using Transformer architecture
This repository contains code for constructing a translation model based on the Transformer architecture using PyTorch, focusing on translation between Vietnamese and English.
## Overview
The Transformer architecture has been highly successful in the field of machine translation and natural language processing in general. This sparked my curiosity about its inner workings. Therefore, I reimplemented this architecture using PyTorch for the task of English-to-Vietnamese and Vietnamese-to-English translation. Additionally, I made some modifications to the architecture to allow the model to handle both languages and reduce the number of parameters without significantly impacting the model's performance.
## Data
The dataset I used for this project was collected from [here](https://huggingface.co/datasets/harouzie/vi_en-translation). For convenience in the training process, the data is stored in a .json format as a list of dictionaries:
```
[
    {
        "id": "HAI_0",
        "E": "I heard that since Tom isn't feeling well he won't come to school today",
        "V": "tôi nghe nói rằng vì tom không khỏe nên hôm nay anh ấy sẽ không đến trường"
    },
    {
        "id": "HAI_1",
        "E": "The pharmacy is on Fresno Street",
        "V": "hiệu thuốc nằm trên đường fresno"
    },
    ...
]    
```
## Model
### Standard Architecture
The Transformer architecture is widely known, and you can refer to the paper [Attention Is All You Need](https://arxiv.org/pdf/1706.03762). Numerous code implementations of this architecture can be found on GitHub, for example, at [here](https://github.com/bangoc123/transformer).
### My Model
First, in the decoder, the word embedding matrix can be reused to map the output. See the following example:
```
class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, src, trg_mask, src_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)
        return output
```
Essentially, the `self.emb` component is a matrix of size `dec_voc_size` x `dmodel`, and `self.linear` is a matrix of size `dmodel` x `dec_voc_size`. Thus, instead of using two separate matrices, we can use a single matrix of size `dec_voc_size` x `dmodel`. When performing `self.linear`, we can simply reuse this matrix by transposing it.

Second, the decoder is essentially the encoder. The difference lies in the decoder's use of cross-attention between the input and output sentences. By removing cross-attention, it functions as an encoder. Thus, two decoders can be employed in the model—one for Vietnamese and one for English. Depending on the translation task, the input sentence will be passed through one of these decoders, which will act as the encoder. In this case, the encoder will skip the cross-attention mechanism
