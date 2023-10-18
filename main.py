'''

TRY WITH 2X2 BOARD
TRY WITH 3X3 BOARD



'''

import torch
from model import BoardGFLowNet
from board import random_board, get_reward, move, create_action_mask
import wandb
from torch.distributions.categorical import Categorical
import argparse
import random

def loss_fn(predicted_logZ, reward, forward_probabilities):
    
    log_Pf = sum(list(map(torch.log, forward_probabilities)))
    inner = predicted_logZ + log_Pf - torch.log(reward) 
    print(f'Forward probs: {forward_probabilities}, log Pf: {log_Pf}, log reward: {torch.log(reward)}, predicted logZ: {predicted_logZ}, loss: {inner ** 2}')
    return inner ** 2

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--modelsize', type=str, default='1.3b', metavar='s',
                        help='OPT model size to use')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--gamma', type=float, default=0.97724, metavar='M',
                        help='Learning rate step gamma (default: 0.97724)')
    parser.add_argument('--batch_size', type=int, default=32, metavar='S',
                        help='batch size to use')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # torch.set_printoptions(precision=1)

    lr = 1e-2
    decoder_layers = 3
    encoder_layers = 3
    embed_dim = 32
    d_ff = 32
    n_heads = 8
    batch_size = 16
    side_len = 2
    max_steps = 8
    wandb.login()
    wandb.init(
        # set the wandb project where this run will be logged
        project="Gflowgame",
        config={
            'lr': lr,
            'batch_size': batch_size,
            'decoder_layers': decoder_layers,
            'encoder_layers': encoder_layers,
            'embed_dim': embed_dim,
            'n_heads': n_heads
        }
    )
    gfn = BoardGFLowNet(side_len, embed_dim, d_ff, n_heads, encoder_layers, decoder_layers, 6)
    optimizer = torch.optim.Adam(gfn.parameters(), lr=lr)
    s_boards = random_board(side_len=2).unsqueeze(0)
    s_boards = torch.Tensor([[2,1],
                             [0,3]]).unsqueeze(0).type(torch.LongTensor)
    total_parameters = 0
    for param in gfn.parameters():
        total_parameters += param.numel()
    print(f"There are {total_parameters} parameters.")
    loss_history = []
    reward_history = []

    for batch in range(10000):
        total_loss = 0
        total_reward = 0
        total_matching = 0
        for sample in range(batch_size):
            #boards = s_boards.clone()
            boards = random_board(side_len=side_len).unsqueeze(0)
            moves = torch.zeros(1,1).type(torch.LongTensor)
            forward_probabilities = []
            for i in range(max_steps):
                print(boards)
                logz, logits = gfn(boards, moves)
                if(i == 0):
                    starting_logz = logz

                logits = logits[0, -1, :]
                mask = create_action_mask(boards[0])
                if(i == max_steps-1):
                    mask = torch.ones_like(mask) * -1e20
                    mask[1] = 0
                    
                # print('Logits before mask: ', logits)
                logits = torch.softmax(mask + logits, dim=0)
                print('Masked logits:',logits)
                # print('Mask', mask)
                
                if(random.randint(0, 5) >= 0):               
                    new_moves = Categorical(probs=logits.squeeze()).sample()
                    new_moves = torch.Tensor([new_moves]).type(torch.LongTensor)
                else:
                    new_moves = torch.Tensor([logits.squeeze().argmax()]).type(torch.LongTensor)
                    print(logits.squeeze())
                    # print(new_moves)
                
                
                forward_probabilities.append(logits[new_moves])
                moves = torch.cat([moves, new_moves.unsqueeze(0)], dim=1)
                
                boards = boards.clone()
                for index, board in enumerate(boards):
                    boards[index] = move(board, new_moves[index])
                if(new_moves[0] == 1):
                    break
            
            
            reward, matching = get_reward(boards)
            print(f'Reward: {reward}, matching: {matching}')
            total_reward += reward
            total_matching += matching
            loss = loss_fn(starting_logz, reward, forward_probabilities)
            loss.backward(retain_graph=False)
            total_loss += loss
            print('\n\n\n')
        for name, param in gfn.logZ_predictor.named_parameters():
            param.grad *= 10
            print(name, param.grad)

        # gfn.logz.grad *= 10
        optimizer.step()
        # print(gfn.logz)
        optimizer.zero_grad()
        wandb.log({'batch': batch, 'loss': (total_loss/batch_size).item(), 'reward': (total_reward/batch_size).item(), 'matching': {(total_matching/batch_size).item()}, 'LogZ': gfn.logz.item()})
        print(f'Batch {batch}, loss: {total_loss/batch_size}, reward: {total_reward/batch_size}, Matching: {(total_matching/batch_size).item()}')
        loss_history.append((total_loss/batch_size).item())
        reward_history.append((total_reward/batch_size).item())