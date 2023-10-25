import unittest
import torch
from board import create_action_mask, get_reward, move

class TestBoardMethods(unittest.TestCase):

    def test_create_action_mask(self):
        board1 = torch.arange(4).reshape(1,2,2)
        mask1 = create_action_mask(board1)
        self.assertTrue(mask1.equal(torch.Tensor([[-1e20, 0, -1e20, 0, 0, -1e20]])))

        board2 = torch.arange(16).reshape(1,4,4)
        mask = create_action_mask(board2)
        self.assertTrue(mask.equal(torch.Tensor([[-1e20, 0, -1e20, 0, 0, -1e20]])))

        board3 = torch.Tensor([[1,2,3],
                               [4,0,5],
                               [6,7,8]]).unsqueeze(0)
        mask = create_action_mask(board3)
        self.assertTrue(mask.equal(torch.Tensor([[-1e20, 0, 0, 0, 0, 0]])))

        board4 = torch.Tensor([[1,2,3,4],
                               [5,6,7,8],
                               [9,10,11,12],
                               [13,14,15,0]]).unsqueeze(0)
        mask = create_action_mask(board4)
        self.assertTrue(mask.equal(torch.Tensor([[-1e20, 0, 0, -1e20, -1e20, 0]])))

    def test_get_reward(self):
        board1 = torch.arange(16).reshape(1,4,4)
        log_reward, num_match = get_reward(board1, beta=1.0)
        self.assertEqual(log_reward, torch.Tensor([0]))

        board2 = torch.arange(100).reshape(1,10,10)
        log_reward, num_match = get_reward(board2, beta=1.0)
        self.assertEqual(log_reward, torch.Tensor([0]))

        board3 = torch.Tensor([[1,2],
                               [3,0]]).unsqueeze(0)
        log_reward, num_match = get_reward(board3, beta=1.0)
        self.assertEqual(log_reward, torch.Tensor([-4]))


        board3 = torch.Tensor([[1,2],
                               [3,0]]).unsqueeze(0)
        log_reward, num_match = get_reward(board3, beta=2.0)
        self.assertEqual(log_reward, torch.Tensor([-8]))

    def test_move(self):
        board1 = torch.arange(4).reshape(1,2,2)
        board1 = move(board1, torch.Tensor([[2]]))
        board1 = move(board1, torch.Tensor([[2]]))
        self.assertTrue(board1.equal(torch.Tensor([[0,1],
                                                   [2,3]]).unsqueeze(0)))
        
        board2 = torch.arange(4).reshape(1,2,2)
        board2 = move(board2, torch.Tensor([[3]]))
        board2 = move(board2, torch.Tensor([[3]]))
        board2 = move(board2, torch.Tensor([[3]]))
        self.assertTrue(board2.equal(torch.Tensor([[2,1],
                                                   [0,3]]).unsqueeze(0)))
        
        board3 = torch.arange(9).reshape(1,3,3)
        board3 = move(board3, torch.Tensor([[3]]))
        board3 = move(board3, torch.Tensor([[3]]))
        board3 = move(board3, torch.Tensor([[3]]))
        board3 = move(board3, torch.Tensor([[3]]))
        self.assertTrue(board3.equal(torch.Tensor([[3,1,2],
                                                   [6,4,5],
                                                   [0,7,8]]).unsqueeze(0)))
        
        board4 = torch.arange(9).reshape(1,3,3)
        board4 = move(board4, torch.Tensor([[3]]))
        board4 = move(board4, torch.Tensor([[2]]))
        board4 = move(board4, torch.Tensor([[4]]))
        board4 = move(board4, torch.Tensor([[4]]))
        board4 = move(board4, torch.Tensor([[4]]))
        board4 = move(board4, torch.Tensor([[3]]))
        board4 = move(board4, torch.Tensor([[3]]))
        self.assertTrue(board4.equal(torch.Tensor([[1,2,5],
                                                   [3,4,8],
                                                   [6,7,0]]).unsqueeze(0)))
        
        board5 = torch.arange(16).reshape(1,4,4)
        board5 = move(board5, torch.Tensor([[3]]))
        board5 = move(board5, torch.Tensor([[2]]))
        board5 = move(board5, torch.Tensor([[4]]))
        board5 = move(board5, torch.Tensor([[4]]))
        board5 = move(board5, torch.Tensor([[4]]))
        board5 = move(board5, torch.Tensor([[3]]))
        board5 = move(board5, torch.Tensor([[3]]))
        board5 = move(board5, torch.Tensor([[5]]))
        board5 = move(board5, torch.Tensor([[5]]))
        board5 = move(board5, torch.Tensor([[5]]))
        board5 = move(board5, torch.Tensor([[2]]))
        board5 = move(board5, torch.Tensor([[2]]))
        board5 = move(board5, torch.Tensor([[2]]))
        self.assertTrue(board5.equal(torch.Tensor([[0, 2, 3, 7],
                                                   [1, 5, 6, 11],
                                                   [4, 8, 9,10],
                                                   [12,13,14,15]]).unsqueeze(0)))
        
if __name__ == '__main__':
    unittest.main()