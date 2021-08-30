import numpy as np


class Prob:
    def __init__(self, val=None, logval=None):
        if logval is None:
            self.logval = np.log(val)
        else:
            self.logval = logval
            
    def __add__(self, other):
        logprob1 = np.where(self.logval > other.logval, self.logval, other.logval)
        logprob2 = np.where(self.logval < other.logval, self.logval, other.logval)
        res = logprob1 + np.log1p(np.exp(logprob2 - logprob1))
        return Prob(logval=res)
    
    def __mul__(self, other):
        return Prob(logval=self.logval+other.logval)
    
    def __truediv__(self, other):
        return Prob(logval=self.logval-other.logval)
    
    def __getitem__(self, key):
        return Prob(logval=self.logval[key])
    
    def __setitem__(self, key, item):
        """
        `item` is the value of log-probability or an instance of `Prob`
        """
        if isinstance(item, Prob):
            self.logval[key] = item.logval
        else:
            self.logval[key] = item
    
    def __radd__(self, other):
        """
        self.__radd__(), self.__iter__(), self.__next__() are defined for `sum()`
        And only consider 1-d case
        """
        if other == 0:
            return self
        else:
            return self.__add__(other)
    
    def __iter__(self):
        if isinstance(self.logval, (int, float)):
            self._isscalar = True
            self._index = -1
            self._n = 1
        else:
            self._isscalar = False
            self._index = -1
            self._n = len(self.logval)
        return self
    
    def __next__(self):
        self._index += 1
            
        if self._index < self._n:
            if self._isscalar:
                return Prob(logval=self.logval)
            else:
                return Prob(logval=self.logval[self._index])
        else:
            raise StopIteration