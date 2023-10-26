
This repository contains code for training a GFlowNet encoder-decoder transformer to sample trajectories from the [sliding puzzle](https://en.wikipedia.org/wiki/Sliding_puzzle) game. 
To start training, run 

``python main.py``.
The board size can be modified using the ``--boardwidth`` argument. Training hyperparameters can also be modified. Their descriptions are found in ``main.py``.

The reward function is $e^{-n}$, where $n$ is the number of mismatching squares (Hamming distance) between the final board and the solved board.

