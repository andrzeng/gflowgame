import torch
from torch.distributions.categorical import Categorical
from model import BoardGFLowNet
from board import random_board, get_reward, move, create_action_mask
import wandb

def sample_move(boards: torch.Tensor, 
                logits: torch.Tensor,
                temperature: float, 
                at_step_limit: bool):
    
    batch_size, _, _ = boards.shape
    last_logits = logits[:, -1, :]
    
    if(at_step_limit):
        mask = torch.ones(6) * -1e20
        mask[1] = 0
        mask = mask.expand((batch_size, 6))
    else:
        mask = create_action_mask(boards)
    
    last_logits_with_temp = torch.softmax((mask + last_logits) * temperature, dim=1)
    last_logits = torch.softmax(mask + last_logits, dim=1)
    new_moves = Categorical(probs=last_logits_with_temp).sample()
    new_moves = torch.Tensor(new_moves).type(torch.LongTensor)
    return new_moves, last_logits[torch.arange(batch_size), new_moves]

def loss_fn(predicted_logZ: torch.Tensor, 
            log_reward: torch.Tensor, 
            forward_probabilities: torch.Tensor):
    #print('predicted logZ:\n', predicted_logZ)
    #print('log reward: \n', log_reward)
    log_Pf = torch.log(forward_probabilities).sum(dim=1)

    #print('sum log of forward probabilities:\n', log_Pf)
    inner = predicted_logZ.squeeze() + log_Pf - log_reward
    #print('loss:\n', inner ** 2)
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
    beta=1,
    temperature=1,
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
            'beta': beta,
        }
    )
    
    gfn = BoardGFLowNet(side_len, embed_dim, d_ff, n_heads, encoder_layers, decoder_layers, 6)
    optimizer = torch.optim.Adam(gfn.parameters(), lr=lr)
    starting_board = torch.Tensor([[3,1,2],
                                   [6,4,5],
                                   [7,8,0]]).unsqueeze(0).type(torch.LongTensor)

    for batch in range(total_batches):
    
        # boards = random_board(batch_size, side_len) 
        boards = starting_board.repeat(batch_size, 1, 1)
        finished = torch.zeros((batch_size, 1)) # Keep track of which boards in the batch have sampled a terminating state
        moves = torch.zeros(batch_size, 1).type(torch.LongTensor)
        forward_probabilities = torch.ones(batch_size, 1) # Keep track of the forward probabilities along each board's trajectory
        predicted_logZ, _ = gfn(boards, moves) 

        for i in range(max_steps):
            _, logits = gfn(boards, moves)
            new_move, move_prob = sample_move(boards, logits, temperature, i == max_steps-1)
            #print('last logits:\n', logits[:, -1, :])
            #print('new move:\n', new_move)

            move_prob[torch.where(finished == 1)[0]] = 1
            finished[torch.where(new_move == 1)] = 1
            forward_probabilities = torch.cat([forward_probabilities, move_prob.unsqueeze(1)], dim=1)
            moves = torch.cat([moves, new_move.unsqueeze(1)], dim=1)
            #print('boards before:\n', boards)
            boards = boards.clone()
            boards = move(boards, new_move, finished_mask=finished)
            #print('boards after:\n', boards)

        log_reward, matching = get_reward(boards, beta)
        
        loss = loss_fn(predicted_logZ, log_reward, forward_probabilities).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        log_reward = log_reward.sum() / batch_size
        matching = matching.sum() / batch_size
        loss = loss.item() / batch_size
        print(f'Batch {batch}, loss: {loss}, log reward: {log_reward}, Matching: {matching}')
        wandb.log({'batch': batch, 'loss': loss, 'log reward': log_reward, 'num_matching': matching})
        if((batch+1) % checkpoint_freq == 0):
            torch.save(gfn.state_dict(), f'checkpoints/model_step_{batch}.pt')
