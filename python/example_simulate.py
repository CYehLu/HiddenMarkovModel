import numpy as np
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

    seq_obs, seq_state = hmm.simulate(nseq=30)
    
    print('Simulated sequence of observations:')
    print(seq_obs)
    print('Simulated sequence of hidden states:')
    print(seq_state)