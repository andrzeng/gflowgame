import wandb
import argparse
from train import train, train_alternate

def main():
    parser = argparse.ArgumentParser(description='GFlowGame')
    parser.add_argument('--batches', type=int, default=1000, metavar='B',
                        help='Number of batches to train for')
    parser.add_argument('--batchsize', type=int, default=16, metavar='S',
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--embedding', type=int, default=32, metavar='M',
                        help='Embedding dimension')
    parser.add_argument('--layers', type=int, default=10, metavar='LY',
                        help='Number of layers')
    parser.add_argument('--dropout', type=int, default=0.0, metavar='D',
                        help='Dropout')
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
    parser.add_argument('--device', type=str, default='cpu', metavar='DV',
                        help='Device (cuda:n or cpu)')
    parser.add_argument('--alternate', action='store_true')
    parser.add_argument('--datafile', type=str, default='cpu', metavar='DF',
                        help='Dataset file for offline training')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    wandb.login()
    args = main()
    if(args.alternate):
        train_alternate(
            args.datafile,
            args.lr, 
            args.embedding, 
            args.layers,
            args.dropout,
            args.batchsize, 
            args.boardwidth, 
            args.maxsteps, 
            args.batches, 
            args.checkpointfreq, 
            args.beta, 
            args.temperature,
            args.logz_factor,
            args.name,
            args.device)
    else:
        train(args.lr, 
            args.embedding, 
            args.layers,
            args.dropout,
            args.batchsize, 
            args.boardwidth, 
            args.maxsteps, 
            args.batches, 
            args.checkpointfreq, 
            args.beta, 
            args.temperature,
            args.logz_factor,
            args.name,
            args.device)