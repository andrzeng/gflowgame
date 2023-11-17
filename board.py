import torch

DIR_UP = 2
DIR_DOWN = 3
DIR_RIGHT = 4
DIR_LEFT = 5

def create_action_mask(board: torch.Tensor, device='cpu'):
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
    
    mask = mask.to(device)
    return mask

def move(boards: torch.Tensor, move_dirs: torch.Tensor, finished_mask=None):
    batch_size, _, side_length = boards.shape
    gap_coord = torch.where(boards == 0)[1:]
    gap_coord = torch.stack(gap_coord).transpose(0,1)
    if(finished_mask == None):
        finished_mask = torch.zeros(batch_size,1)
    
    for i, board in enumerate(boards):
        if(finished_mask[i] == 1):
            continue

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

def random_board(num, side_len, num_random_moves=100, device='cpu'):
    boards = torch.arange(0,side_len**2).reshape((side_len,side_len)).repeat(num, 1,1)
    for _ in range(num_random_moves):
        moves = torch.randint(2,6,(num,))
        boards = move(boards, moves)
    
    boards = boards.to(device)
    return boards

def random_trajectories(num, side_len, num_random_moves=100, device='cpu'):
    boards = torch.arange(0,side_len**2).reshape((side_len,side_len)).repeat(num, 1,1)
    for _ in range(num_random_moves):
        moves = torch.randint(2,6,(num,))
        boards = move(boards, moves)
    
    boards = boards.to(device)
    return boards

def get_reward(boards: torch.Tensor, beta=1.0):
    batch_size, _, side_len = boards.shape
    ground_truth = torch.arange(0, side_len**2).reshape(side_len,side_len).expand_as(boards)
    ground_truth = ground_truth.to(boards.device)
    mismatch = boards - ground_truth
    match = mismatch == 0
    mismatch = mismatch != 0
    num_mismatch = mismatch.flatten(1).count_nonzero(1)
    num_match = match.flatten(1).count_nonzero(1)
    log_reward = -num_mismatch * beta
    return log_reward, num_match