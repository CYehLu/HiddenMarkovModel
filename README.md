# HiddenMarkovModel

An implement of hidden Markov model (HMM) by C and Python.  
  
## C version
 - tested on Linux with gcc compiler
 - run example programs:   
   1. enter directory and compile files  
       `$ cd c/examples`  
       `$ make`
   2. execute programs for different kinds of HMM problems:  
       `$ ./simulate` : given HMM, simulate observations  
       `$ ./likelihood`: given HMM, find the likelihood of sequence of observations  
       `$ ./decoding`: given HMM, find the most probable sequence of states  
       `$ ./learning`: given sequence of observations, learn the HMM parameters
 
## Python version
 - tested on Linux
 - dependency: numpy
 - run example programs:  
   `$ cd python/examples`  
   `$ python simulate.py` for simulation problem  
   `$ python likelihood.py` for likelihood problem  
   `$ python decoding.py` for decoding problem  
   `$ python learning.py` for learning problem
   
## Algorithm
 - likelihood problem: Forward algorithm  
 - decoding problem: Viterbi algorithm  
 - learning problem: Forward-backward algorithm / Baum-Welch algorithm
 
## Reference
Speech and Language Processing. Daniel Jurafsky & James H. Martin.  
https://web.stanford.edu/~jurafsky/slp3/A.pdf