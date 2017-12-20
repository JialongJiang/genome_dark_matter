# -*- coding: utf-8 -*-
"""
This is a minimal example of our idea. 
"""


import numpy as np
import scipy.signal as sg
import warnings


class VarArray:
    def __init__(self, num, leng, leng_max):
        arr_ini = np.random.randint(0, 2, size=(num, leng))
        self.array = zero2minus(arr_ini)
        self.num, self.leng = self.array.shape
        self.leng_max = leng_max
    
    def update(self, arr):
        self.num, self.leng = arr.shape
        self.array = arr
    
    def mutation(self, mut_rate):
        mut_rand = 1 - 2 * (np.random.rand(*self.array.shape) < mut_rate)
        self.array = self.array * mut_rand
        
    def renew_array(self, new_arr, new_ind):
        """New array longer than max length would be deprecated."""
        num_arr_n, leng_n = new_arr.shape
        num_arr, leng = self.array.shape
        if leng_n > self.leng_max:
            warnings.warn(
                    'Maximum length reached, and the outliers would be deprecated')
            new_arr = new_arr[: self.leng_max]
        new_leng = min(max(leng, leng_n), leng_max)
        rep_arr = np.zeros((num_arr, new_leng))
        fit_leng = np.lib.pad(new_arr, (0, new_leng - leng_n), 
                              'constant', constant_values=0)
        rep_arr[:, : leng] = self.array
        for iter_new in range(num_arr_n):
            rep_arr[new_ind[iter_new], :] = fit_leng[iter_new, :]
        self.update(rep_arr)
    
class EnviList(VarArray):
    def __init__(self, num, leng_ini, leng_max):
        VarArray.__init__(self, num, leng_ini, leng_max)
    
    def enlong(self, rate):
        num, _ = self.array.shape
        num_sel = np.random.binomial(num, rate)
        if num_sel:
            sel = np.random.choice(np.arange(num), size=num_sel)
            new_arr = self.env_enlong(self.array[sel, :])
            self.renew_array(new_arr, sel)
    
    @staticmethod        
    def env_enlong(env):
        num, _ = env.shape
        all_leng = get_length(env)
        leng = np.max(all_leng) + 1
        new_arr = np.zeros((num, leng))
        for iter_env in range(num):   
            cenv =  np.trim_zeros(env[iter_env, :])
            app = 2 * np.random.randint(0, 2) - 1
            if np.random.random() < 0.5:
                narr = np.append(cenv, app)
            else:
                narr = np.insert(cenv, 0, app)
            new_arr[iter_env, : narr.size] = narr
        return new_arr
                       
class PopList(VarArray):
    def __init__(self, num, leng_ini, leng_max):
        VarArray.__init__(self, num, leng_ini, leng_max)
        
    def trans(self, freq, tr_leng):
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
        num, _ = array.shape
        all_leng = get_length(array)
        if mode == 'add':
            tran = zero2minus(np.random.randint(0, 2, self.tr_leng))
            leng = np.max(all_leng) + self.tr_leng
            new_arr = np.zeros((num, leng))
            for iter_arr in range(num):
                nrow = np.insert(array[iter_arr, :], 
                                 np.random.randint(0, all_leng[iter_arr]), tran)    
                try:
                    new_arr[iter_arr, : all_leng[iter_arr] + self.tr_leng] = np.trim_zeros(nrow)
                except:
                    a = 1
        elif mode == 'del':
            leng = np.max(all_leng) - self.tr_leng
            new_arr = np.zeros((num, int(leng)))  
            for iter_arr in range(num): 
                if all_leng[iter_arr] <= self.tr_leng:
                    new_arr[iter_arr, : int(all_leng[iter_arr])] = np.trim_zeros(array[iter_arr, :])
                else:
                    nrow = np.delete(array[iter_arr, :], 
                                     np.random.randint(0, all_leng[iter_arr])
                                     + np.arange(self.tr_leng))     
                    try:
                        new_arr[iter_arr, : all_leng[iter_arr] - self.tr_leng] = np.trim_zeros(nrow)
                    except:
                        a = 1
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

def reproduce(pop, envir, fun):
    num_pop, leng_m = pop.shape
    leng = get_length(pop)
    score = np.sum(fun(pop, envir), axis=0)
    all_sc = np.exp(score) / leng
    para = num_pop * all_sc / np.sum(all_sc)
    num_off = np.random.poisson(para)
    print(np.sum(num_off))
    if np.sum(num_off) == 0:
        raise ValueError('Population died out.')
    ind = np.repeat(np.arange(num_pop), num_off)
    return pop[ind, :]
    
   
mut_rate = 1e-2
tran_rate = 0.1
tran_leng = 3
leng_max = 100
leng_ini = 5
num_pop = 500

env_mut_rate = 1e-1
env_add_rate = 1e-1
env_leng_max = 20
env_leng_ini = 5
env_num = 4
    
max_gener = 100

popu = PopList(num_pop, leng_ini, leng_max)
envir = EnviList(env_num, env_leng_ini, env_leng_max)

for iter_evolu in range(max_gener):
    new_arr = reproduce(popu.array, envir.array, score_fun)
    popu.update(new_arr)
    popu.mutation(mut_rate)
    popu.trans(tran_rate, tran_leng)
    envir.mutation(env_mut_rate)
    envir.enlong(env_add_rate)
