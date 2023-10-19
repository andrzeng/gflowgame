'''

**********************************************

**********************************************

**********************************************

**********************************************
**********************************************
**********************************************
**********************************************
**********************************************


ADD RELU!!!! DONT MAKE THIS SILLY MISTAKE AGAIN

**********************************************
**********************************************
********************************************** YOU SILLY
**********************************************GOOSE
**********************************************
**********************************************
**********************************************


Fix the heads issue as well

add dropout

compare with https://github.com/TranQuocTrinh/transformer/blob/main/models.py line by line


worst case scenario: just copy someone elses code


'''
















import torch
import torch.nn as nn
import math

class Norm(nn.Module):
    def __init__(self, d_embed):
        super(Norm, self).__init__()
        self.norm = nn.LayerNorm(d_embed)

    def forward(self, x):
        return self.norm(x)

class SelfAttention(nn.Module):
    def __init__(self, d_embed):
        super().__init__()
        self.Wq = nn.Linear(d_embed, d_embed)
        self.Wk = nn.Linear(d_embed, d_embed)
        self.Wv = nn.Linear(d_embed, d_embed)
        self.d_embed = d_embed

    def forward(self, embeddings, mask=None):
        q = self.Wq(embeddings)
        k = self.Wk(embeddings)
        v = self.Wv(embeddings)
        attention = torch.matmul(q, k.transpose(2,1)) # transpose the embedding and sequence dims
        
        if(mask != None):
            attention = attention + mask


        attention = attention / math.sqrt(self.d_embed)
        attention = attention.softmax(dim=-1)
        
        return attention @ v
    
class MultiheadAttention(nn.Module):
    def __init__(self, d_embed, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(d_embed) for _ in range(n_heads)])
        self.combine = nn.Linear(n_heads * d_embed, d_embed)

    def forward(self, embeddings, mask=None):
        head_outputs = torch.concat([head(embeddings, mask) for head in self.heads], dim=2)
        return self.combine(head_outputs)
    
class CrossAttention(nn.Module):
    def __init__(self, d_embed):
        super().__init__()
        self.Wq = nn.Linear(d_embed, d_embed)
        self.Wk = nn.Linear(d_embed, d_embed)
        self.Wv = nn.Linear(d_embed, d_embed)
        self.d_embed = d_embed

    def forward(self, encoder_embs, decoder_embs, mask=None):
        q = self.Wq(decoder_embs)
        k = self.Wk(encoder_embs)
        v = self.Wv(encoder_embs)
        
        attention = torch.matmul(q, k.transpose(2,1)) # transpose the embedding and sequence dims
        if(mask != None):
            attention = attention + mask
        attention = attention / math.sqrt(self.d_embed)
        attention = attention.softmax(dim=-1)

        return attention @ v
    
class MultiheadCrossAttention(nn.Module):
    def __init__(self, d_embed, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([CrossAttention(d_embed) for _ in range(n_heads)])
        self.combine = nn.Linear(d_embed * n_heads, d_embed)
    
    def forward(self, encoder_embs, decoder_embs, mask=None):
        head_outputs = torch.concat([head(encoder_embs, decoder_embs, mask) for head in self.heads], dim=2)
        return self.combine(head_outputs)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class EncoderLayer(nn.Module):
    def __init__(self, d_embed, d_ff, n_heads):
        super().__init__()
        self.mha = MultiheadAttention(d_embed, n_heads)
        self.attention_norm = Norm(d_embed) # RMSNorm(d_embed)
        self.feedforward_norm = Norm(d_embed) # RMSNorm(d_embed)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_embed, d_ff),
            nn.ELU(),
            nn.Linear(d_ff, d_embed)
        )

    def forward(self, embeddings):
        attention_output = self.mha(embeddings)
        ln_output = self.attention_norm(embeddings + attention_output)
        ff_out = self.feed_forward(ln_output)
        return self.feedforward_norm(ln_output + ff_out)
    
class Encoder(nn.Module):
    def __init__(self, d_embed, d_ff, n_heads, n_encoders):
        super().__init__()
        self.encoders = nn.ModuleList([EncoderLayer(d_embed, d_ff, n_heads) for _ in range(n_encoders)])
    
    def forward(self, embeddings):
        for encoder in self.encoders:
            embeddings = encoder(embeddings)
        
        return embeddings
    
class DecoderLayer(nn.Module):
    def __init__(self, d_embed, d_ff, n_heads):
        super().__init__()
        self.masked_mha = MultiheadAttention(d_embed, n_heads)
        self.attention_norm = Norm(d_embed)# RMSNorm(d_embed)
        self.feedforward_norm1 = Norm(d_embed)# RMSNorm(d_embed)
        self.feedforward_norm2 = Norm(d_embed)# RMSNorm(d_embed)
        
        self.cross_mha = MultiheadCrossAttention(d_embed, n_heads)
        self.feed_forward = nn.Sequential(
                    nn.Linear(d_embed, d_ff),
                    nn.ELU(),
                    nn.Linear(d_ff, d_embed)
                )
    def forward(self, decoder_embeddings, encoder_embeddings):
        B, L, D = decoder_embeddings.shape

        mask = torch.tril(torch.ones(L,L))
        mask[mask == 0] = -1e20
        mask[mask == 1] = 0
        mask=None
        masked_mha_out = self.masked_mha(decoder_embeddings, mask=mask)
        ln_out = self.feedforward_norm1(masked_mha_out + decoder_embeddings)
        cross_mha_out = self.cross_mha(encoder_embeddings, ln_out)
        ln_out = self.attention_norm(cross_mha_out + ln_out)
        ff_out = self.feed_forward(ln_out)
        return self.feedforward_norm2(ff_out + ln_out)
    
class Decoder(nn.Module):
    def __init__(self, d_embed, d_ff, n_heads, n_decoders):
        super().__init__()
        self.decoders = nn.ModuleList([DecoderLayer(d_embed, d_ff, n_heads) for _ in range(n_decoders)])
    
    def forward(self, decoder_embeddings, encoder_embeddings):
        for decoder in self.decoders:
            decoder_embeddings = decoder(decoder_embeddings, encoder_embeddings)
        return decoder_embeddings
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        return x + self.pe[:,:x.size(1)]
    
class BoardTransformer(nn.Module):
    def __init__(self, side_len, d_embed, d_ff, n_heads, n_encoders, n_decoders, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.board_embedding = nn.Embedding(side_len**2, d_embed)
        self.pe = PositionalEncoding(d_embed)
        self.encoder = Encoder(d_embed, d_ff, n_heads, n_encoders)
        self.decoder = Decoder(d_embed, d_ff, n_heads, n_decoders)
        self.linear = nn.Linear(d_embed, vocab_size)

    def forward(self, board, move_seq):
        board = board.flatten(start_dim=1)
        board_embs = self.board_embedding(board)
        print('Board embs:\n', board_embs)
        encoder_out = self.encoder(board_embs)
        print('Encoder out:\n', encoder_out)
        decoder_emb = self.pe(self.embedding(move_seq))
        decoder_out = self.decoder(decoder_emb, encoder_out)
        print('Decoder out:\n', decoder_out)
        prelogits = self.linear(decoder_out)
        return encoder_out, prelogits #.softmax(dim=2)     

class BoardGFLowNet(nn.Module):
    def __init__(self, side_len, d_embed, d_ff, n_heads, encoder_layers, decoder_layers, vocab_size):
        super().__init__()
        self.transformer = BoardTransformer(side_len, d_embed, d_ff, n_heads, encoder_layers, decoder_layers, vocab_size)
        self.logZ_predictor = nn.Sequential(
            nn.Linear(d_embed * side_len ** 2, int(d_embed * side_len ** 2)//2),
            nn.ELU(),
            nn.Linear(int(d_embed * side_len ** 2)//2, int(d_embed * side_len ** 2)//4),
            nn.ELU(),
            nn.Linear(int(d_embed * side_len ** 2)//4,1),
        )
        
        self.logz = nn.Parameter(torch.Tensor([1]), requires_grad=True)

    def forward(self, boards, moves):
        board_embs, logits = self.transformer(boards, moves)
        
        board_embs = board_embs.flatten(start_dim=1)
        
        #print(encoder_out)
        predicted_logZ = self.logZ_predictor(board_embs)
        # predicted_logZ = self.logz
        
        return predicted_logZ, logits