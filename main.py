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

    lr = 1e-4
    decoder_layers = 4
    encoder_layers = 4
    embed_dim = 8
    d_ff = 8
    n_heads = 4
    batch_size = 32
    '''wandb.login()
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
    '''
    
    gfn = BoardGFLowNet(embed_dim, d_ff, n_heads, encoder_layers, decoder_layers, 6)
    optimizer = torch.optim.AdamW(gfn.parameters(), lr=lr)

    loss_history = []
    reward_history = []

    for batch in range(10000):
        total_loss = 0
        total_reward = 0
        for sample in range(batch_size):

            boards = random_board().unsqueeze(0)
            moves = torch.zeros(1,1).type(torch.LongTensor)
            forward_probabilities = []
            for i in range(100):
                logz, logits = gfn(boards, moves)
                if(i == 0):
                    starting_logz = logz

                logits = logits[0, -1, :]
                mask = create_action_mask(boards[0])
                logits = torch.softmax(mask * logits, dim=0)

                if(random.randint(0, 9) >= 0):               
                    new_moves = Categorical(probs=logits.squeeze()).sample()
                    new_moves = torch.Tensor([new_moves]).type(torch.LongTensor)
                else:
                    new_moves = torch.Tensor([logits.squeeze().argmax()]).type(torch.LongTensor)
                    # print(logits.squeeze())
                    # print(new_moves)
                
                
                forward_probabilities.append(logits[new_moves])
                moves = torch.cat([moves, new_moves.unsqueeze(0)], dim=1)
                
                boards = boards.clone()
                for index, board in enumerate(boards):
                    boards[index] = move(board, new_moves[index])
                if(new_moves[0] == 1):
                    break
            
            reward = get_reward(boards)
            total_reward += reward
            loss = loss_fn(starting_logz, reward, forward_probabilities)
            loss.backward(retain_graph=False)
            total_loss += loss

        for param in gfn.logZ_predictor.parameters():
            param.grad *= 10
        optimizer.step()
        optimizer.zero_grad()
        # wandb.log({'batch': batch, 'loss': (total_loss/batch_size).item(), 'reward': (total_reward/batch_size).item()})
        print(f'Batch {batch}, loss: {total_loss/batch_size}, reward: {total_reward/batch_size}')
        loss_history.append((total_loss/batch_size).item())
        reward_history.append((total_reward/batch_size).item())