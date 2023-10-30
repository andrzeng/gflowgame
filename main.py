import wandb
import argparse
from train import train

def main():
    parser = argparse.ArgumentParser(description='GFlowGame')
    parser.add_argument('--batches', type=int, default=1000, metavar='B',
                        help='Number of batches to train for')
    parser.add_argument('--batchsize', type=int, default=16, metavar='S',
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--encoders', type=int, default=3, metavar='E',
                        help='Number of encoder layers')
    parser.add_argument('--decoders', type=int, default=3, metavar='D',
                        help='Number of decoder layers')
    parser.add_argument('--embedding', type=int, default=32, metavar='M',
                        help='Embedding dimension')
    parser.add_argument('--heads', type=int, default=8, metavar='H',
                        help='Number of heads')
    parser.add_argument('--ff', type=int, default=32, metavar='F',
                        help='Feedforward dimension')
    parser.add_argument('--maxsteps', type=int, default=20, metavar='P',
                        help='Maximum steps')
    parser.add_argument('--boardwidth', type=int, default=3, metavar='L',
                        help='Side length of the square board')
    parser.add_argument('--checkpointfreq', type=int, default=10, metavar='C',
                        help='How often to save checkpoints (in terms of batches)')
    parser.add_argument('--beta', type=float, default=1, metavar='RT',
                        help='reward temperature')
    parser.add_argument('--temperature', type=float, default=1, metavar='EE',
                        help='sampling temperature')
    parser.add_argument('--logz_factor', type=float, default=10, metavar='LZ',
                        help='Factor to multiply the predicted logZ before feeding it into the loss function')
    parser.add_argument('--name', type=str, default=None, metavar='NM',
                        help='Name of run (for Wandb logging purposes)')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    wandb.login()
    args = main()
    train(args.lr, 
          args.decoders, 
          args.encoders, 
          args.embedding, 
          args.ff, 
          args.heads, 
          args.batchsize, 
          args.boardwidth, 
          args.maxsteps, 
          args.batches, 
          args.checkpointfreq, 
          args.beta, 
          args.temperature,
          args.logz_factor,
          args.name)