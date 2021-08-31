import numpy as np

import sys
sys.path.append('../src')
from hmm import HiddenMarkovModel


def show_params(hmm, truehmm):
    A = hmm.A
    B = hmm.B
    ini_prob = hmm.ini_prob
    
    tA = truehmm.A
    tB = truehmm.B
    tini_prob = truehmm.ini_prob
    
    nstate, nobs = hmm.B.shape
    
    for i in range(nstate):
        for j in range(nstate):
            print(f'     A[{i},{j}] = {A[i,j]:.4f} ({tA[i,j]})  ', end='')
        print()
    print()
    
    for i in range(nstate):
        for v in range(nobs):
            print(f'     B[{i},{v}] = {B[i,v]:.4f} ({tB[i,v]})  ', end='')
        print()
    print()
    
    for i in range(nstate):
        print(f'     ini_prob[{i}] = {ini_prob[i]:.4f} ({tini_prob[i]})  ', end='')
    print()


if __name__ == '__main__':
    # create true HMM and generate observations
    truehmm = HiddenMarkovModel(nstate=2, nobs=3)
    truehmm.ini_prob = np.array([0.8, 0.2])
    truehmm.A = np.array([
        [0.65, 0.35],
        [0.25, 0.75]
    ])
    truehmm.B = np.array([
        [0.2, 0.4, 0.4],
        [0.5, 0.4, 0.1]
    ])

    nseq = 1000
    obs, _ = truehmm.simulate(nseq)

    # fit HMM
    hmm = HiddenMarkovModel(nstate=2, nobs=3)
    
    print(' Parameter = initial value (true value)')
    show_params(hmm, truehmm)
    
    iterinfo = hmm.learning(obs, maxiter=10, tol=0.01, verbose=False)
    
    print()
    print()
    print(' Iteration information:')
    for k, v in iterinfo.items():
        print(f'     {k} : {v}')
    
    print()
    print()
    print(' Parameter = estimated value (true value)')
    show_params(hmm, truehmm)
    
    

