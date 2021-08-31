import numpy as np

import sys
sys.path.append('../src')
from hmm import HiddenMarkovModel


if __name__ == '__main__':
    # case 1
    hmm = HiddenMarkovModel(nstate=2, nobs=3)
    
    hmm.ini_prob = np.array([0.8, 0.2])
    hmm.A = np.array([
        [0.6, 0.4],
        [0.5, 0.5]
    ])
    hmm.B = np.array([
        [0.2, 0.4, 0.4],
        [0.5, 0.4, 0.1]
    ])

    seq_obs = np.array([2, 0, 2])
    best_prob, best_seq = hmm.decoding(seq_obs)
    
    print('Case 1:')
    print(f'    max log-probability = {best_prob}')
    print(f'    corresponding hidden state sequence :\n        {best_seq}')
    
    
    # case 2
    hmm = HiddenMarkovModel(2, 3)

    hmm.ini_prob = np.array([0.6, 0.4])
    hmm.A = np.array([
        [0.7, 0.3],
        [0.4, 0.6]
    ])
    hmm.B = np.array([
        [0.5, 0.4, 0.1],
        [0.1, 0.3, 0.6]
    ])

    seq_obs = np.array([0, 1, 2])
    best_prob, best_seq = hmm.decoding(seq_obs)
    
    print('Case 2:')
    print(f'    max log-probability = {best_prob}')
    print(f'    corresponding hidden state sequence :\n        {best_seq}')