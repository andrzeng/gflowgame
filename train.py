import torch
from torch.distributions.categorical import Categorical
from model import BoardGFLowNet
from board import random_board, get_reward, move, create_action_mask
import wandb


def sample_move(boards: torch.Tensor, 
                logits: torch.Tensor, 
                at_step_limit: bool):
    
    batch_size, _, _ = boards.shape
    last_logits = logits[:, -1, :]
    
    if(at_step_limit):
        mask = torch.ones(6) * -1e20
        mask[1] = 0
        mask = mask.expand((batch_size, 6))
    else:
        mask = create_action_mask(boards)
    
    last_logits = torch.softmax(mask + last_logits, dim=1)
    new_moves = Categorical(probs=last_logits).sample()
    new_moves = torch.Tensor(new_moves).type(torch.LongTensor)
    return new_moves, last_logits[torch.arange(batch_size), new_moves]

def loss_fn(predicted_logZ: torch.Tensor, 
            reward: torch.Tensor, 
            forward_probabilities: torch.Tensor):
    
    log_Pf = torch.log(forward_probabilities).sum(dim=1)
    inner = predicted_logZ.squeeze() + log_Pf - torch.log(reward) 
    return inner ** 2

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
    
    gfn = BoardGFLowNet(side_len, embed_dim, d_ff, n_heads, encoder_layers, decoder_layers, 6)
    optimizer = torch.optim.Adam(gfn.parameters(), lr=lr)

    for batch in range(total_batches):
    
        boards = random_board(batch_size, side_len) 
        finished = torch.zeros((batch_size, 1))
        moves = torch.zeros(batch_size, 1).type(torch.LongTensor)
        forward_probabilities = torch.ones(batch_size, 1)

        predicted_logZ, _ = gfn(boards, moves)
        batch_loss = 0
        batch_reward = 0
        batch_matching = 0
        
        for i in range(max_steps):
            _, logits = gfn(boards, moves)
            new_move, move_prob = sample_move(boards, logits, i == max_steps-1)

            for index in range(len(move_prob)):
                if(finished[index] == 1):
                    move_prob[index] = 1
            for index, _move in enumerate(new_move):
                if(_move == 1):
                    finished[index] = 1

            forward_probabilities = torch.cat([forward_probabilities, move_prob.unsqueeze(1)], dim=1)
            moves = torch.cat([moves, new_move.unsqueeze(1)], dim=1)
            boards = boards.clone()
            boards = move(boards, new_move, finished_mask=finished)
        
        reward, matching = get_reward(boards)
        loss = loss_fn(predicted_logZ, reward, forward_probabilities)
        loss = torch.sum(loss)
        reward = torch.sum(reward)
        matching = torch.sum(matching)
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
