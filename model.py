import torch
import torch.nn as nn
import math

class MLPLayer(nn.Module):
    def __init__(self, entry_size, dropout):
        super().__init__()
        self.linear = nn.Linear(entry_size, entry_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()
    
    def forward(self, embeddings):
        x = self.linear(embeddings)
        x = self.dropout(x)
        x = self.activation(x)
        return x + embeddings

class BoardMLP(nn.Module):
    def __init__(self, side_len, d_embed, n_layers=10, max_steps=20, dropout=0.1):
        super().__init__()
        self.board_embedding = nn.Embedding(side_len**2, d_embed)
        self.move_embedding = nn.Embedding(max_steps, d_embed)
        entry_size = (side_len ** 2 + 1) * d_embed   
        self.MLP = nn.ModuleList([MLPLayer(entry_size, dropout) for _ in range(n_layers)])
        self.MLP_proj = nn.Linear(entry_size, 6)
        
    def forward(self, 
                board: torch.Tensor, 
                move_num: int
                ): 
        batch_size, _, _ = board.shape
        board = board.flatten(start_dim=1)
        board_embs = self.board_embedding(board)
        board_embs = board_embs.flatten(start_dim=1)
        move_num_embed = self.move_embedding(torch.LongTensor([move_num]))
        move_num_embed = move_num_embed.expand((batch_size, -1)) # might lead to issues; if in doubt, try repeat() instead
        
        embs = torch.cat([board_embs, move_num_embed], dim=1)
        for layer in self.MLP:
            embs = layer(embs)
        embs = self.MLP_proj(embs)
        return board_embs, embs

class BoardGFLowNet2(nn.Module):
    def __init__(self, side_len, d_embed, n_layers=10, max_steps=20, dropout=0.1):
        super().__init__()
        self.MLP = BoardMLP(side_len, d_embed, n_layers, max_steps, dropout)
        self.logz_predictor = nn.ModuleList([MLPLayer(side_len ** 2 * d_embed, dropout) for _ in range(n_layers)])
        self.logz_predictor_proj = nn.Linear(side_len ** 2 * d_embed, 1)

    def forward(self, boards, move_num):
        board_embs, output_embs = self.MLP(boards, move_num)
        for layer in self.logz_predictor:
            board_embs = layer(board_embs)
        logz = self.logz_predictor_proj(board_embs)

        return logz, output_embs.unsqueeze(0)
