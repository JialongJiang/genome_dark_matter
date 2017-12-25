# -*- coding: utf-8 -*-
"""
We decide to fix the length of environment.
"""


import numpy as np
import scipy.signal as sg
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm
np.set_printoptions(threshold=np.nan)

class VarArray:
    """The class for the matrix of array with different length."""
    def __init__(self, num, leng, leng_max):
        arr_ini = np.random.randint(0, 2, size=(num, leng))
        # Convert binary to +1-1
        self.array = zero2minus(arr_ini)
        self.num, self.leng = self.array.shape
        self.leng_max = leng_max
        self.all_leng = get_length(self.array)
    
    def update(self, arr):
        self.num, self.leng = arr.shape
        self.array = arr
    
    def mutation(self, mut_rate):
        """Rondomly change some elements by given mutation rate."""
        mut_rand = 1 - 2 * (np.random.rand(*self.array.shape) < mut_rate)
        self.array = self.array * mut_rand
        
    def renew_array(self, new_arr, new_ind):
        """New array longer than max length would be deprecated."""
        num_arr_n, leng_n = new_arr.shape
        if leng_n > self.leng_max:
            warnings.warn(
                    'Maximum length reached, and the outliers would be deprecated')
            new_arr = new_arr[:, : self.leng_max]
        new_leng = min(max(self.leng, leng_n), self.leng_max)
        # No need to allocate memory when array have enough space
        if new_leng == self.leng:
            rep_arr = self.array
        else:            
            rep_arr = np.zeros((self.num, new_leng))
            rep_arr[:, : self.leng] = self.array
        _, leng_n = new_arr.shape
        for iter_new in range(num_arr_n):
            rep_arr[new_ind[iter_new], : leng_n] = new_arr[iter_new, :]
        self.all_leng = get_length(rep_arr)
        # Deletion may cause extra 0 in the end
        if np.max(self.all_leng) < new_leng:
            rep_arr = rep_arr[:, : np.max(self.all_leng)]
        self.update(rep_arr)
    
class EnviList(VarArray):
    """The class of environment."""
    def __init__(self, num, leng):
        # The length of environment is fixed 
        VarArray.__init__(self, num, leng, leng)  
        #print(self.array)
     
class PopList(VarArray):
    """The class for population."""
    def __init__(self, num, leng_ini, leng_max):
        VarArray.__init__(self, num, leng_ini, leng_max)
        
    def trans(self, freq, tr_leng):
        """A random sequence can be inserted to random position of array, or 
        deleted from random positon. The average insertion and deletion should 
        be the same.
        """
        self.tr_leng = tr_leng
        num, _ = self.array.shape
        num_add = np.random.binomial(num, freq)
        if num_add:
            sel_add = np.random.choice(np.arange(num), size=num_add)
            new_arr = self.pop_change(self.array[sel_add, :].reshape(-1, self.leng),
                                      mode='add')
            self.renew_array(new_arr, np.array(sel_add))
            
            sel_del = np.random.choice(np.arange(num), size=num_add)
            new_arr = self.pop_change(self.array[sel_del, :].reshape(-1, self.leng), 
                                      mode='del')
            self.renew_array(new_arr, np.array(sel_del))
    
    def pop_change(self, array, mode='add'):
        """Implementation of insertion or deletion of random sequences."""
        num, leng = array.shape
        all_leng = get_length(array)
        if mode == 'add':
            tran = zero2minus(np.random.randint(0, 2, self.tr_leng))
            leng = np.max(all_leng) + self.tr_leng
            new_arr = np.zeros((num, leng))
            for iter_arr in range(num):
                nrow = np.insert(array[iter_arr, :], 
                                 np.random.randint(0, all_leng[iter_arr]), tran)    
                new_arr[iter_arr, : all_leng[iter_arr] + self.tr_leng] = np.trim_zeros(nrow)
        elif mode == 'del':
            new_arr = np.zeros((num, leng))  
            for iter_arr in range(num): 
                if all_leng[iter_arr] <= self.tr_leng:
                    new_arr[iter_arr, : leng] = array[iter_arr, :]
                else:
                    nrow = np.delete(array[iter_arr, :], 
                                     np.random.randint(0, all_leng[iter_arr] - self.tr_leng)
                                     + np.arange(self.tr_leng))     
                    new_arr[iter_arr, : all_leng[iter_arr] - self.tr_leng] = np.trim_zeros(nrow)
        else:
            raise ValueError('Illegal input operation.')
        return new_arr
    
                   
def zero2minus(seq):
    return seq * 2 - 1
        
def score_fun(seq, env):
    """Compute the score for every sequence. """
    num_env = env.shape[0]
    num_seq = seq.shape[0]
    score_mat = np.zeros((num_env, num_seq))
    for iter_env in range(num_env):
        cur_env = env[iter_env, :].reshape(1, -1)  
        raw_score = sg.convolve(np.fliplr(cur_env), seq)
        cur_score = np.max(raw_score, axis=1)
        score_mat[iter_env, :] = cur_score
    return 0.5 * score_mat

def get_length(arr):
    """Get the length of nonzero elements for each row."""
    leng = arr.shape[1]
    aux = np.arange(leng)
    eff = np.multiply(aux, np.abs(arr))
    return (np.max(eff, axis=1) + 1).astype(int)

def reproduce(pop, envir, fun, num, first='sum', alpha=1):
    """Using the score function to get the offspring for every generation."""
    num_pop, leng_m = pop.shape
    leng = get_length(pop)
    if first == 'exp':
        all_sc = np.sum(np.exp(fun(pop, envir)), axis=0) / leng ** alpha
    elif first == 'sum':
        all_sc = np.exp(np.sum(fun(pop, envir), axis=0)) / leng ** alpha
    else:
        raise ValueError('Illegal input operation.')        
    para = num * all_sc / np.sum(all_sc)
    num_off = np.random.poisson(para)
    if np.sum(num_off) == 0:
        raise ValueError('Population died out.')
    ind = np.repeat(np.arange(num_pop), num_off)
    return pop[ind, :]
    
   
mut_rate = 0.1
tran_rate = 0.05
tran_leng = 3
leng_max = 200
leng_ini = 40
num_pop = 400

env_mut_rate = 0.01
env_leng_max = 10
env_num = 5
    
max_gener = 600

popu = PopList(num_pop, leng_ini, leng_max)
envir = EnviList(env_num, env_leng_max)
env_leng = get_length(envir.array)

rec_mean = np.zeros(max_gener)
rec_var = np.zeros(max_gener)

for iter_evolu in tqdm(range(max_gener), ascii=True):
    new_arr = reproduce(popu.array, envir.array, score_fun, num_pop)
    popu.update(new_arr)
    popu.mutation(mut_rate)
    popu.trans(tran_rate, tran_leng)
    envir.mutation(env_mut_rate)
    see_leng = get_length(popu.array)
    rec_mean[iter_evolu] = np.mean(see_leng)
    rec_var[iter_evolu] = np.std(see_leng)
    #envir.enlong(env_add_rate)

plt.plot(rec_mean)
plt.show()
plt.plot(rec_var)
plt.show()