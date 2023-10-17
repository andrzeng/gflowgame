import torch
import random

DIR_UP = 2
DIR_DOWN = 3
DIR_RIGHT = 4
DIR_LEFT = 5

def create_action_mask(board):
    gap_coord = torch.where(board == 0)
    mask = torch.ones(6)
    mask[0] = -1e20
    if(gap_coord[0] == 0):
        mask[DIR_UP] = -1e20
    if(gap_coord[0] == 3):
        mask[DIR_DOWN] = -1e20
    if(gap_coord[1] == 0):
        mask[DIR_LEFT] = -1e20
    if(gap_coord[1] == 3):
        mask[DIR_RIGHT] = -1e20
    return mask

def move(board, dir):
    new_board = board
    gap_coord = torch.where(new_board == 0)
    if(dir == DIR_UP and gap_coord[0] > 0):
        new_board[gap_coord[0], gap_coord[1]] = new_board[gap_coord[0]-1, gap_coord[1]] 
        new_board[gap_coord[0]-1, gap_coord[1]] = 0
    elif(dir == DIR_RIGHT and gap_coord[1] < 3):
        new_board[gap_coord[0], gap_coord[1]] = new_board[gap_coord[0], gap_coord[1]+1]
        new_board[gap_coord[0], gap_coord[1]+1] = 0
    if(dir == DIR_DOWN and gap_coord[0] < 3):
        new_board[gap_coord[0], gap_coord[1]] = new_board[gap_coord[0]+1, gap_coord[1]] 
        new_board[gap_coord[0]+1, gap_coord[1]] = 0
    elif(dir == DIR_LEFT and gap_coord[1] > 0):
        new_board[gap_coord[0], gap_coord[1]] = new_board[gap_coord[0], gap_coord[1]-1]
        new_board[gap_coord[0], gap_coord[1]-1] = 0
    return new_board

def random_board():
    board = torch.arange(0,16).reshape((4,4))
    for i in range(100):
        board = move(board, random.randint(2,5))
    return board

def get_reward(boards, eps=1e-6):
    mismatch = boards - torch.arange(0, 16).reshape(4,4).expand_as(boards)
    mismatch = mismatch == 0

    matching = (mismatch.flatten(1).count_nonzero(1))
    reward = max(0, matching - 6)
    reward = reward ** 2
    reward = reward + 1
    return torch.Tensor([reward])
    # return (mismatch.flatten(1).count_nonzero(1)) ** 2 + 1
    return (mismatch.flatten(1).count_nonzero(1) + eps)/(16 + eps)