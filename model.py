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
        attention = torch.matmul(q, k.transpose(2,1)) 
        
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
        
        attention = torch.matmul(q, k.transpose(2,1)) 
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

class EncoderLayer(nn.Module):
    def __init__(self, d_embed, d_ff, n_heads, dropout=0.1):
        super().__init__()
        self.mha = MultiheadAttention(d_embed, n_heads)
        self.attention_norm = Norm(d_embed)
        self.feedforward_norm = Norm(d_embed) 
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_embed, d_ff),
            nn.ELU(),
            nn.Linear(d_ff, d_embed)
        )

    def forward(self, embeddings):
        normed_embs = self.attention_norm(embeddings)
        embeddings = embeddings + self.dropout(self.mha(normed_embs))
        
        normed_embs = self.feedforward_norm(embeddings)
        embeddings = embeddings + self.dropout(self.feed_forward(normed_embs))

        return embeddings
    
class Encoder(nn.Module):
    def __init__(self, d_embed, d_ff, n_heads, n_encoders):
        super().__init__()
        self.encoders = nn.ModuleList([EncoderLayer(d_embed, d_ff, n_heads) for _ in range(n_encoders)])
        self.norm = Norm(d_embed)
    def forward(self, embeddings):
        for encoder in self.encoders:
            embeddings = encoder(embeddings)
        
        return self.norm(embeddings)

class DecoderLayer(nn.Module):
    def __init__(self, d_embed, d_ff, n_heads, dropout=0.1):
        super().__init__()
        self.masked_mha = MultiheadAttention(d_embed, n_heads)
        self.attention_norm = Norm(d_embed)
        self.feedforward_norm1 = Norm(d_embed)
        self.feedforward_norm2 = Norm(d_embed)
        self.dropout = nn.Dropout(dropout)
        self.cross_mha = MultiheadCrossAttention(d_embed, n_heads)
        self.feed_forward = nn.Sequential(
                    nn.Linear(d_embed, d_ff),
                    nn.ELU(),
                    nn.Linear(d_ff, d_embed)
                )
    def forward(self, decoder_embeddings, encoder_embeddings):
        B, L, D = decoder_embeddings.shape

        mask = torch.tril(torch.ones(L,L)).to(decoder_embeddings.device)
        mask[mask == 0] = -1e20
        mask[mask == 1] = 0
        
        normed_embs = self.attention_norm(decoder_embeddings)
        decoder_embeddings = decoder_embeddings + self.dropout(self.masked_mha(normed_embs, mask=mask))
        
        normed_embs = self.feedforward_norm1(decoder_embeddings)
        decoder_embeddings = decoder_embeddings + self.dropout(self.cross_mha(encoder_embeddings, normed_embs))
        
        normed_embs = self.feedforward_norm2(decoder_embeddings)
        decoder_embeddings = decoder_embeddings + self.dropout(self.feed_forward(normed_embs))
                
        return decoder_embeddings
    
class Decoder(nn.Module):
    def __init__(self, d_embed, d_ff, n_heads, n_decoders):
        super().__init__()
        self.decoders = nn.ModuleList([DecoderLayer(d_embed, d_ff, n_heads) for _ in range(n_decoders)])
        self.norm = Norm(d_embed)

    def forward(self, decoder_embeddings, encoder_embeddings):
        for decoder in self.decoders:
            decoder_embeddings = decoder(decoder_embeddings, encoder_embeddings)
        return self.norm(decoder_embeddings)
    
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
        board_embs = self.pe(self.board_embedding(board))
        encoder_out = self.encoder(board_embs)
        decoder_emb = self.pe(self.embedding(move_seq))
        decoder_out = self.decoder(decoder_emb, encoder_out)
        prelogits = self.linear(decoder_out)
        return encoder_out, prelogits    

class BoardGFLowNet(nn.Module):
    def __init__(self, side_len, d_embed, d_ff, n_heads, encoder_layers, decoder_layers, vocab_size, logz_layers=10, dropout=0.1):
        super().__init__()
        self.transformer = BoardTransformer(side_len, d_embed, d_ff, n_heads, encoder_layers, decoder_layers, vocab_size)

        self.logZ_predictor = nn.ModuleList([nn.Sequential(nn.Linear(d_embed * side_len ** 2, d_embed * side_len ** 2),
                                             nn.ELU(),
                                             nn.Dropout(dropout)
                                             ) for _ in range(logz_layers)])  
        self.logZ_predictor_proj = nn.Linear(d_embed * side_len ** 2, 1)

    def forward(self, boards, moves):
        board_embs, logits = self.transformer(boards, moves)
        board_embs = board_embs.flatten(start_dim=1)
        
        for layer in self.logZ_predictor:
            board_embs = layer(board_embs)
        predicted_logZ = self.logZ_predictor_proj(board_embs)
        
        return predicted_logZ, logits
    


