import numpy as np
from base import Prob


class HiddenMarkovModel:
    def __init__(self, nstate, nobs):
        self.nstate = nstate
        self.nobs = nobs
        
        self.ini_prob = np.full((nstate,), 1/nstate)
        self.A = np.full((nstate, nstate), 1/nstate)
        self.B = np.random.rand(nstate, nobs)
        self.B = self.B / self.B.sum(axis=1, keepdims=True)
        
    def simulate(self, nseq):
        nstate = self.nstate
        nobs = self.nobs
        ini_prob = self.ini_prob
        A = self.A
        B = self.B
        
        seq_state = np.empty((nseq,), dtype=int)
        seq_obs = np.empty((nseq,), dtype=int)
        
        seq_state[0] = np.random.choice(nstate, p=ini_prob)
        seq_obs[0] = np.random.choice(nobs, p=B[seq_state[0],:])
        
        for t in range(1, nseq):
            seq_state[t] = np.random.choice(nstate, p=A[seq_state[t-1],:])
            seq_obs[t] = np.random.choice(nobs, p=B[seq_state[t],:])
            
        return seq_obs, seq_state
    
    def likelihood(self, seq_obs):
        A = self._convert_probobj(self.A, '`A` (transition matrix)')
        B = self._convert_probobj(self.B, '`B` (emission matrix)')
        ini_prob = self._convert_probobj(self.ini_prob, '`ini_prob` (initial probability)')
        
        alpha = self._calc_alpha(seq_obs, A, B, ini_prob)   # alpha.logval.shape = (nseq, nstate)
        return sum(Prob(logval=alpha.logval[-1,:])).logval
    
    def decoding(self):
        pass
    
    def learning(self):
        pass
    
    def _convert_probobj(self, x, name):
        if x.ndim == 1:
            if not np.isclose(x.sum(), 1):
                raise ValueError(f"{name} must sum to 1.")
        else:
            if not np.allclose(x.sum(axis=1), 1):
                raise ValueError(f"each of {name} must sum to 1.")
            
        return Prob(val=x)
    
    def _calc_alpha(self, seq_obs, A, B, ini_prob):
        """
        seq_obs : 1d array
        A, B, ini_prob : instance of Prob
        """
        nstate = self.nstate
        nobs = self.nobs
        nseq = seq_obs.size
        
        alpha = Prob(val=np.ones((nseq, nstate)))
        
        # initialization step
        alpha[0,:] = ini_prob * B[:,seq_obs[0]]
        
        # recursion step
        for t in range(1, nseq):
            for s in range(nstate):
                alpha[t,s] = sum(alpha[t-1,:] * A[:,s] * B[s,seq_obs[t]])
                
        return alpha