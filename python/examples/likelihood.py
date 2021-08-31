import numpy as np

import sys
sys.path.append('../src')
from hmm import HiddenMarkovModel


if __name__ == '__main__':
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
    ans = hmm.likelihood(seq_obs)
    
    print(f'log-likelihood = {ans}')