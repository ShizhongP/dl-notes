import torch 

import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.vocab_size = vocab_size
        self._norma_factor = d_model **  0.5
    def forward(self, x):
        return self.embed(x) * self._norma_factor

class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len = 512, dropout = 0.1):
        super(PositionEmbedding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        self.dropout = nn.Dropout(dropout)
        
        # maxlen, d_model
        pe = torch.zeros(max_len, d_model)
        # [1,max_len]
        position = torch.arange(0, max_len).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self,x):
        # x: batch_size, sequence_length, d_model
        # pe : 1,  max_length , d_model
        # broadcast 
        x = x + self.pe[:,:x.shape[1]]
        return self.dropout(x)
    

class TransFormerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len = 512, dropout = 0.1):
        super(TransFormerEmbedding, self).__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.position_embedding = PositionEmbedding(d_model, max_len, dropout)
        
    def forward(self, x):
        return self.position_embedding(self.embedding(x))
    
def test():
    batch_size = 3
    dim_embedding = 16
    sequence_length =4
    vocab_size = 200
    x =torch.ones(batch_size,sequence_length,dtype=torch.long)
    
    transformerembedding = TransFormerEmbedding(vocab_size=200,d_model=dim_embedding)
    
    output = transformerembedding(x)
    
    print(output.shape)

if __name__ == "__main__":
    test()

        
        