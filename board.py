import torch
import random

DIR_UP = 2
DIR_DOWN = 3
DIR_RIGHT = 4
DIR_LEFT = 5

def create_action_mask(board: torch.Tensor):
    side_len = board.shape[-1]
    gap_coord = torch.where(board == 0)
    mask = torch.zeros(6)
    mask[0] = -1e20
    if(gap_coord[0] == 0):
        mask[DIR_UP] = -1e20
    if(gap_coord[0] == side_len-1):
        mask[DIR_DOWN] = -1e20
    if(gap_coord[1] == 0):
        mask[DIR_LEFT] = -1e20
    if(gap_coord[1] == side_len-1):
        mask[DIR_RIGHT] = -1e20
    return mask

def move(board: torch.Tensor, dir: int):
    length = board.shape[0]
    gap_coord = torch.where(board == 0)
    if(dir == DIR_UP and gap_coord[0] > 0):
        board[gap_coord[0], gap_coord[1]] = board[gap_coord[0]-1, gap_coord[1]] 
        board[gap_coord[0]-1, gap_coord[1]] = 0
    elif(dir == DIR_RIGHT and gap_coord[1] < length-1):
        board[gap_coord[0], gap_coord[1]] = board[gap_coord[0], gap_coord[1]+1]
        board[gap_coord[0], gap_coord[1]+1] = 0
    if(dir == DIR_DOWN and gap_coord[0] < length-1):
        board[gap_coord[0], gap_coord[1]] = board[gap_coord[0]+1, gap_coord[1]] 
        board[gap_coord[0]+1, gap_coord[1]] = 0
    elif(dir == DIR_LEFT and gap_coord[1] > 0):
        board[gap_coord[0], gap_coord[1]] = board[gap_coord[0], gap_coord[1]-1]
        board[gap_coord[0], gap_coord[1]-1] = 0
    return board

def random_board(side_len=4):
    board = torch.arange(0,side_len**2).reshape((side_len,side_len))
    for _ in range(100):
        board = move(board, random.randint(2,5))
    return board

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