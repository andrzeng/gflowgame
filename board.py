import torch
import random

DIR_UP = 2
DIR_DOWN = 3
DIR_RIGHT = 4
DIR_LEFT = 5

def create_action_mask(board: torch.Tensor):
    batch_size, _, side_len = board.shape
    gap_coord = torch.where(board == 0)[1:]
    gap_coord = torch.stack(gap_coord).transpose(0,1)
    mask = torch.zeros(batch_size, 6)

    for i in range(batch_size):
        mask[i,0] = -1e20
        if(gap_coord[i,0] == 0):
            mask[i,DIR_UP] = -1e20
        if(gap_coord[i,0] == side_len-1):
            mask[i,DIR_DOWN] = -1e20
        if(gap_coord[i,1] == 0):
            mask[i,DIR_LEFT] = -1e20
        if(gap_coord[i,1] == side_len-1):
            mask[i,DIR_RIGHT] = -1e20
    return mask

def move(boards: torch.Tensor, move_dirs: torch.Tensor):
    _, _, side_length = boards.shape
    gap_coord = torch.where(boards == 0)[1:]
    gap_coord = torch.stack(gap_coord).transpose(0,1)
    for i, board in enumerate(boards):
        dir = move_dirs[i]
        if(dir == DIR_UP and gap_coord[i,0] > 0):
            board[gap_coord[i,0], gap_coord[i,1]] = board[gap_coord[i,0]-1, gap_coord[i,1]] 
            board[gap_coord[i,0]-1, gap_coord[i,1]] = 0
        elif(dir == DIR_RIGHT and gap_coord[i,1] < side_length-1):
            board[gap_coord[i,0], gap_coord[i,1]] = board[gap_coord[i,0], gap_coord[i,1]+1]
            board[gap_coord[i,0], gap_coord[i,1]+1] = 0
        elif(dir == DIR_DOWN and gap_coord[i,0] < side_length-1):
            board[gap_coord[i,0], gap_coord[i,1]] = board[gap_coord[i,0]+1, gap_coord[i,1]] 
            board[gap_coord[i,0]+1, gap_coord[i,1]] = 0
        elif(dir == DIR_LEFT and gap_coord[i,1] > 0):
            board[gap_coord[i,0], gap_coord[i,1]] = board[gap_coord[i,0], gap_coord[i,1]-1]
            board[gap_coord[i,0], gap_coord[i,1]-1] = 0

    return boards

def random_board(num, side_len):
    boards = torch.arange(0,side_len**2).reshape((side_len,side_len)).repeat(num, 1,1)
    for _ in range(100):
        moves = torch.randint(2,6, (num,))
        boards = move(boards, moves)
    
    return boards

def get_reward(boards: torch.Tensor):
    side_len = boards.shape[-1]
    ground_truth = torch.arange(0, side_len**2).reshape(side_len,side_len)
    mismatch = boards - ground_truth
    match = mismatch == 0
    mismatch = mismatch != 0
    num_mismatch = mismatch.flatten(1).count_nonzero(1)
    num_match = match.flatten(1).count_nonzero(1)
    reward = torch.exp(-num_mismatch) 
    return torch.Tensor([reward]), num_match