import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from model import BoardGFLowNet2
from board import random_board, get_reward, move, create_action_mask
import wandb

def sample_move(boards: torch.Tensor, 
                logits: torch.Tensor,
                temperature: float, 
                at_step_limit: bool,
                forbid_stop: bool,
                device='cpu'):
    
    batch_size, _, _ = boards.shape
    last_logits = logits[-1, :, :]
    if(at_step_limit):
        mask = torch.ones(6) * -1e20
        mask[1] = 0
        mask = mask.expand((batch_size, 6)).to(device)
        last_logits_with_temp = torch.softmax((last_logits + mask) / temperature, dim=1)
        print('last logits with temp:\n', last_logits_with_temp[0])
        last_logits = torch.softmax(last_logits + mask, dim=1)
        new_moves = Categorical(probs=last_logits_with_temp).sample()
        new_moves = torch.Tensor(new_moves).type(torch.LongTensor).to(device)

    else:
        mask = create_action_mask(boards, device)
        tempered_logits = (last_logits + mask) / temperature
        if(forbid_stop):
            tempered_logits[:, 1] = -1e20
        last_logits_with_temp = torch.softmax(tempered_logits, dim=1)
        last_logits = torch.softmax(last_logits + mask, dim=1)
        new_moves = Categorical(probs=last_logits_with_temp).sample()
        new_moves = torch.Tensor(new_moves).type(torch.LongTensor).to(device)
        
    return new_moves, last_logits[torch.arange(batch_size), new_moves]

def sample_move_offline(
                offline_moves: torch.Tensor,
                boards: torch.Tensor, 
                logits: torch.Tensor,
                ):
    batch_size, _, _ = boards.shape
    last_logits = logits[:, -1, :]
    mask = create_action_mask(boards)
    last_logits = torch.softmax(mask + last_logits, dim=1)
    new_moves = offline_moves
    new_moves = torch.Tensor(new_moves).type(torch.LongTensor)

    return new_moves, last_logits[torch.arange(batch_size), new_moves]


def loss_fn(predicted_logZ: torch.Tensor, 
            reward: torch.Tensor, 
            forward_probabilities: torch.Tensor):
    #print(f'predicted logZ: {predicted_logZ}\nreward: {reward}\nforward probs: {forward_probabilities}')
    log_Pf = torch.log(forward_probabilities).sum(dim=1)
    inner = predicted_logZ.squeeze() + log_Pf - reward
    return inner ** 2

def get_total_params(model):
    total = 0
    for param in model.parameters():
        total += param.numel()
    return total

def train(
    lr=1e-4,
    embed_dim=32,
    layers=10,
    dropout=0.0,
    batch_size=16,
    side_len=3,
    max_steps=20,
    total_batches=1000,
    checkpoint_freq=10,
    beta=1,
    temperature=1,
    logz_factor=10,
    hiddenexpansion=3,
    name=None,
    device='cpu',
    ):

    wandb.init(
        project="none",
        name=name,
        config={
            'lr': lr,
            'batch_size': batch_size,
            'embed_dim': embed_dim,
            'n_layers': layers,
            'dropout': dropout,
            'side_len': side_len,
            'max_steps': max_steps,
            'beta': beta,
            'logz_factor': logz_factor,
            'hidden_expansion': hiddenexpansion,
        }
    )
    hiddenexpansion = int(hiddenexpansion)
    gfn = BoardGFLowNet2(side_len, embed_dim, layers, max_steps, dropout, hiddenexpansion)
    print(f'There are {get_total_params(gfn)} parameters')
    optimizer = torch.optim.Adam(gfn.parameters(), lr=lr, weight_decay=1e-4)
    
    
    for batch in range(total_batches):
        boards = random_board(batch_size, side_len, num_random_moves=200, device=device) 
        # boards = starting_board.repeat([batch_size, 1, 1])
        finished = torch.zeros((batch_size, 1)) # Keep track of which boards in the batch have sampled a terminating state
        move_num = 0
        moves = torch.zeros(batch_size, 1).type(torch.LongTensor).to(device)
        forward_probabilities = torch.ones(batch_size, 1).to(device) # Keep track of the forward probabilities along each board's trajectory
        predicted_logZ, _ = gfn(boards, 0) 
        predicted_logZ = predicted_logZ * logz_factor


        for i in range(max_steps):
            _, logits= gfn(boards, move_num)

            move_num += 1
            #print('last logits:\n', logits[0, -1, :])
            #print('board before move:\n', boards[0])
            new_move, move_prob = sample_move(boards, logits, temperature, i == max_steps-1, (i < 10) and (batch < 20), device)
            move_prob[torch.where(finished == 1)[0]] = 1
            finished[torch.where(new_move == 1)] = 1
            forward_probabilities = torch.cat([forward_probabilities, move_prob.unsqueeze(1)], dim=1)
            moves = torch.cat([moves, new_move.unsqueeze(1)], dim=1)
            boards = boards.clone()
            boards = move(boards, new_move, finished_mask=finished)
            #print('boards after move:\n', boards[0])
        #print('\n\n\n')
        log_reward, matching = get_reward(boards, beta)
        loss = loss_fn(predicted_logZ, log_reward, forward_probabilities).sum() 
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        log_reward = log_reward.sum() / batch_size
        matching = matching.sum() / batch_size
        loss = loss.item() / batch_size
        average_trajectory_len = torch.where(moves[:, 1:] == 1, 1, 0).argmax(dim=1).type(torch.Tensor).mean()
        temperature *= 0.99
        if(temperature < 1):
            temperature = 1
        
        print(f'Batch {batch}, loss: {loss}, log reward: {log_reward}, Matching: {matching}, avg trajectory len: {average_trajectory_len}')
        wandb.log({'batch': batch, 'loss': loss, 'log reward': log_reward, 'num_matching': matching, 'avg_trajectory_len': average_trajectory_len})
        if((batch+1) % checkpoint_freq == 0):
            torch.save(gfn.state_dict(), f'checkpoints/model_step_{batch}.pt')
