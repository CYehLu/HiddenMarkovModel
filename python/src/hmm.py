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
        return sum(alpha[-1,:]).logval
    
    def decoding(self, seq_obs):
        A = self._convert_probobj(self.A, '`A` (transition matrix)')
        B = self._convert_probobj(self.B, '`B` (emission matrix)')
        ini_prob = self._convert_probobj(self.ini_prob, '`ini_prob` (initial probability)')
        
        best_prob, best_seq = self._viterbi(seq_obs, A, B, ini_prob)
        return best_prob, best_seq
    
    def learning(self, seq_obs, maxiter=100, tol=0.001, verbose=False):
        nstate = self.nstate
        nobs = self.nobs
        nseq = len(seq_obs)
        
        # initial parameters
        A = self._convert_probobj(self.A, '`A` (transition matrix)')
        B = self._convert_probobj(self.B, '`B` (emission matrix)')
        ini_prob = self._convert_probobj(self.ini_prob, '`ini_prob` (initial probability)')
        
        gamma = Prob(val=np.ones((nseq, nstate)))
        xi = Prob(val=np.ones((nseq-1, nstate, nstate)))
        
        alpha = self._calc_alpha(seq_obs, A, B, ini_prob)
        prev_score = sum(alpha[-1,:]).logval
        
        if verbose:
            print('   ================================================================  ')
        
        for it in range(maxiter):
            gamma, xi = self._do_e_step(seq_obs, A, B, ini_prob)
            A, B, ini_prob = self._do_m_step(seq_obs, gamma, xi)
            
            alpha = self._calc_alpha(seq_obs, A, B, ini_prob)
            now_score = sum(alpha[-1,:]).logval
            diff = now_score - prev_score
            
            if verbose:
                print(f'   iter = {it} --- previous_score = {prev_score}  new_score = {now_score}')
                print(f'   {" " * (11+len(str(it)))} diff = {diff}  tol = {tol}  isconverge = {diff <= tol}')
                
            if diff <= tol:
                break
            else:
                prev_score = now_score
                
        if verbose:
            print('   ================================================================  ')
            
        self.A = np.exp(A.logval)
        self.B = np.exp(B.logval)
        self.ini_prob = np.exp(ini_prob.logval)
            
        iterinfo = {
            'iter': it + 1,
            'maxiter': maxiter,
            'score': now_score,
            'prev_score': prev_score,
            'score_diff': diff,
            'tol': tol,
            'isconverge': diff <= tol
        }
        return iterinfo
    
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
        A, B, ini_prob : instance of `Prob`
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
    
    def _calc_beta(self, seq_obs, A, B):
        """
        seq_obs : 1d array
        A, B : instance of `Prob`
        """
        nstate = self.nstate
        nobs = self.nobs
        nseq = seq_obs.size
        
        beta = Prob(val=np.ones((nseq, nstate)))
        
        # initialization step
        beta[-1,:] = np.log(1)
        
        # recursion step
        for t in reversed(range(nseq-1)):
            for s in range(nstate):
                beta[t,s] = sum(A[s,:] * B[:,seq_obs[t+1]] * beta[t+1,:])
                
        return beta
    
    def _viterbi(self, seq_obs, A, B, ini_prob):
        """
        seq_obs : 1d array
        A, B, ini_prob : instance of `Prob`
        """
        nstate = self.nstate
        nobs = self.nobs
        nseq = seq_obs.size
        
        v = Prob(val=np.ones((nseq, nstate)))
        backptr = np.empty((nseq, nstate), dtype=int)
        
        # initialization step
        v[0,:] = ini_prob * B[:,seq_obs[0]]
        backptr[0,:] = 0
        
        # recursion step
        for t in range(1, nseq):
            for s in range(nstate):
                candidate = v[t-1,:] * A[:,s] * B[s,seq_obs[t]]
                v[t,s] = candidate.max()
                backptr[t,s] = candidate.argmax()
                
        # termination step
        best_prob = v[-1,:].max()
        best_seq = np.empty((nseq,), dtype=int)
        best_seq[-1] = v[-1,:].argmax()
        
        for t in reversed(range(nseq-1)):
            slc = (t+1, best_seq[t+1])
            best_seq[t] = backptr[slc]
            
        return best_prob, best_seq
    
    def _do_e_step(self, seq_obs, A, B, ini_prob):
        """
        seq_obs : 1d array
        A, B, ini_prob : instance of `Prob`
        """
        nstate = self.nstate
        nobs = self.nobs
        nseq = seq_obs.size
        
        alpha = self._calc_alpha(seq_obs, A, B, ini_prob)
        beta = self._calc_beta(seq_obs, A, B)
        
        norm = Prob(val=np.ones((nseq,)))
        for t in range(nseq):
            norm[t] = sum(alpha[t,:] * beta[t,:])
        
        gamma = Prob(val=np.ones((nseq, nstate)))
        for t in range(nseq):
            numerator = alpha[t,:] * beta[t,:]
            gamma[t,:] = numerator / norm[t]
            
        xi = Prob(val=np.ones((nseq-1, nstate, nstate)))
        for t in range(nseq-1):
            alpha_i = alpha[[t],:].reshape(nstate, 1)
            a_ij = A
            b_j = B[:,seq_obs[t+1]]
            beta_j = beta[t+1,:]
            xi[t,:,:] = alpha_i * a_ij * b_j * beta_j
            
        return gamma, xi
    
    def _do_m_step(self, seq_obs, gamma, xi):
        nstate = self.nstate
        nobs = self.nobs
        nseq = seq_obs.size
        
        # 1. update `A`
        xi_sum_t = sum(xi)                       # xi_sum_t[i,j] = sum_t(xi[t,i,j])
        xi_sum_tj = sum(xi_sum_t.transpose())    # xi_sum_tj[i] = sum_t(sum_j(xi[t,i,j]))
        A = xi_sum_t / xi_sum_tj.reshape(nstate, 1)
        
        # 2. update `B`
        denominator = sum(gamma)                         # denominator[j] = sum_t(gamma[t,j])
        
        numerator = Prob(val=np.ones((nstate, nobs)))    # numerator[j,v] = sum_t(gamma[t,j]) s.t. seqObs[t] == v
        for iobs in range(nobs):
            idx = (seq_obs == iobs)
            numerator[:,iobs] = sum(gamma[idx,:])
            
        B = numerator / denominator.reshape(nstate, 1)
        
        # 3. update `ini_prob`
        ini_prob = gamma[0,:]
        
        return A, B, ini_prob