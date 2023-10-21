import torch
from torch.distributions.categorical import Categorical
from model import BoardGFLowNet
from board import random_board, get_reward, move, create_action_mask
import wandb

def loss_fn(predicted_logZ: torch.Tensor, 
            reward: torch.Tensor, 
            forward_probabilities: list):
    
    log_Pf = sum(list(map(torch.log, forward_probabilities)))
    inner = predicted_logZ + log_Pf - torch.log(reward) 
    return inner ** 2

def sample_move(boards: torch.Tensor, 
                logits: torch.Tensor, 
                at_step_limit: bool):
    
    last_logits = logits[0, -1, :]
    mask = create_action_mask(boards[0])
    if(at_step_limit):
        mask = torch.ones_like(mask) * -1e20
        mask[1] = 0
    last_logits = torch.softmax(mask + last_logits, dim=0)
    new_moves = Categorical(probs=last_logits.squeeze()).sample()
    new_moves = torch.Tensor([new_moves]).type(torch.LongTensor)
    return new_moves, last_logits[new_moves]

def train(
    lr=1e-4,
    decoder_layers=3,
    encoder_layers=3,
    embed_dim=32,
    d_ff=32,
    n_heads=8,
    batch_size=16,
    side_len=3,
    max_steps=20,
    total_batches=1000,
    checkpoint_freq=10,
    ):
    
    wandb.init(
        project="Gflowgame",
        config={
            'lr': lr,
            'batch_size': batch_size,
            'decoder_layers': decoder_layers,
            'encoder_layers': encoder_layers,
            'embed_dim': embed_dim,
            'n_heads': n_heads,
            'd_ff': d_ff,
            'side_len': side_len,
            'max_steps': max_steps,
        }
    )

    gfn = BoardGFLowNet(side_len, embed_dim, d_ff, n_heads, encoder_layers, decoder_layers, 6)
    optimizer = torch.optim.Adam(gfn.parameters(), lr=lr)

    for batch in range(total_batches):
        batch_loss = 0
        batch_reward = 0
        batch_matching = 0
        for _ in range(batch_size):
            boards = random_board(side_len=side_len).unsqueeze(0)
            moves = torch.zeros(1,1).type(torch.LongTensor)
            forward_probabilities = []
            predicted_logZ, _ = gfn(boards, moves)

            for i in range(max_steps):
                _, logits = gfn(boards, moves)
                new_move, move_prob = sample_move(boards, logits, i == max_steps-1)
                forward_probabilities.append(move_prob)
                moves = torch.cat([moves, new_move.unsqueeze(0)], dim=1)
                boards = boards.clone()
                boards[0] = move(boards[0], new_move)
                
            reward, matching = get_reward(boards)
            loss = loss_fn(predicted_logZ, reward, forward_probabilities)
            loss.backward(retain_graph=False)

            batch_reward += reward
            batch_matching += matching
            batch_loss += loss
        
        optimizer.step()
        optimizer.zero_grad()
        batch_reward = batch_reward.item() / batch_size
        batch_matching = batch_matching.item() / batch_size
        batch_loss = batch_loss.item() / batch_size
        print(f'Batch {batch}, loss: {batch_loss}, reward: {batch_reward}, Matching: {batch_matching}')
        wandb.log({'batch': batch, 'loss': batch_loss, 'reward': batch_reward, 'num_matching': batch_matching})

        if((batch+1) % checkpoint_freq == 0):
            torch.save(gfn.state_dict(), f'checkpoints/model_step_{batch}.pt')

        