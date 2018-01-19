# -*- coding: utf-8 -*-
"""
This version makes more abstract and change the mechanism of environmental 
changes. 
"""

from tqdm import tqdm 
import pickle
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.signal as sg
import time 
import warnings
from scipy.special import expit

class VarArray:
    """The class for the matrix of array with different length."""
    def __init__(self, num, leng, leng_max):
        arr_ini = np.random.randint(0, 2, size=(num, leng))
        # Convert binary to +1-1
        self.array = self.zero2minus(arr_ini)
        self.num, self.leng = self.array.shape
        self.leng_max = leng_max
        self.all_leng = self.get_length(self.array)
    
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
        self.all_leng = self.get_length(rep_arr)
        # Deletion may cause extra 0 in the end
        if np.max(self.all_leng) < new_leng:
            rep_arr = rep_arr[:, : np.max(self.all_leng)]
        self.update(rep_arr)
    
    @staticmethod
    def get_length(arr):
        """Get the length of nonzero elements for each row."""
        leng = arr.shape[1]
        aux = np.arange(leng)
        eff = np.multiply(aux, np.abs(arr))
        return (np.max(eff, axis=1) + 1).astype(int)
    
    @staticmethod
    def zero2minus(seq):
        return seq * 2 - 1

class Population(VarArray):
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
        all_leng = self.get_length(array)
        if mode == 'add':
            tran = self.zero2minus(np.random.randint(0, 2, self.tr_leng))
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

class Environment:
    def __init__(self, num_env, leng, mut_rate, env_m, env_j):
        self.m = env_m
        self.j = env_j
        self.num = num_env
        self.pool = np.random.randint(0, 2, size=(num_env, leng))
        self.pool = 2 * self.pool - 1
        self.list = np.random.randint(0, 2, size=num_env)
        self.equilibrate()
        self.update_array()
        self.rate = mut_rate
               
    def mutation(self):
        if np.random.random() < self.rate:
            self.update()
            self.update_array()            
            return self.list
    
    def equilibrate(self):
        for iter_eq in range(500):
            self.update()
        print(self.list)
                            
    def update(self):
        prop = np.random.randint(0, self.num)
        ind_minus = 2 * self.list - 1
        diff_e = 2 * ind_minus[prop] * (
                self.m[prop] + np.sum(self.j[prop, :] * ind_minus))
        accept_prob = min(1, np.exp(- diff_e))
        if np.random.random() < accept_prob:
            self.list[prop] = 1 - self.list[prop]
        while np.sum(self.list) == 0:
            self.update()
    
    def update_array(self):
        ind = np.nonzero(self.list)[0]
        self.array = self.pool[ind, :]
    
class Recorder:
    def __init__(self, num_record, *args):
        self.create_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        for attr_name in args:
            setattr(self, attr_name, np.zeros(num_record))
    
    def vector_init(self, num_record, **kwargs):
        """Create record for matrix property."""
        for mat_name in kwargs:
            setattr(self, mat_name, np.zeros((num_record, *kwargs[mat_name])))   
            
    def plot_process(self, data_name):
        """Plot changing of data during training process."""
        try:
            plot_y = getattr(self, data_name)
        except AttributeError:
            print(data_name + ' does not exist')
        else:
            plot_x = np.arange(plot_y.shape[0])
            plt.plot(plot_x, plot_y)
            plt.suptitle(data_name)
            plt.show()  
    
    def update(self, count, attr_name, value):
        """Feed data into the record. Multiple properties can be combined into 
        tuples for simplicity.
        """
        for iter_attr, tr_name in enumerate(attr_name):
            attr_value = getattr(self, tr_name)
            attr_value[count] = value[iter_attr]
            
    def save_self(self, filena='rec.pkl'):
        """Save evolution record."""
        try:
            filename = self.filename
        except:            
            filename = self.create_time + '_' + filena
            self.filename = filename
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

class Evolution:
    def __init__(self, num_population=100, num_environment=10,
                                 evolution_time=100,
                                 initial_length=20, max_length=200,
                                 environment_length=10,
                                 length_cost=0.01,
                                 mutation_rate=0.01, transpose_rate=0.005,
                                 transpose_length=3,
                                 environment_change_rate=0.01, 
                                 environment_m = 0,
                                 environment_j = 0):
        
        popu = Population(num_population, initial_length, max_length)
        envir = Environment(num_environment, environment_length, 
                            environment_change_rate,
                            environment_m, environment_j)
        # The value can be saved to a time-ordered vector by add their name to 
        # the list 
        save_values = ('mean', 'std', 'mean_score', 'logit')
        self.record = Recorder(evolution_time, *save_values)
        
        rec_list = envir.list
        rec_num = [0]
        
        for iter_evolu in tqdm(range(evolution_time), ascii=True):
            raw_score = self.score_fun(popu.array, envir.array)
            sig_score = np.sum(expit(1.6 * raw_score - 8), axis=0)
            reproduce_ratio = self.get_fitness(popu, raw_score, length_cost)
            new_arr = self.reproduce(popu.array, reproduce_ratio, 
                                     reproduce_ratio, num_population)
            popu.update(new_arr)
            popu.mutation(mutation_rate)
            popu.trans(transpose_rate, transpose_length)
            n_list = envir.mutation()
            if n_list is not None:
                rec_list = np.vstack((rec_list, n_list))
                rec_num.append(iter_evolu)
            see_leng = popu.get_length(popu.array)
            self.record.update(iter_evolu, save_values, 
                               (np.mean(see_leng), np.std(see_leng),
                                np.mean(np.sum(raw_score, axis=0)),
                                np.mean(sig_score, axis=0)))
            
        self.record.save_self()
        print(rec_num)
        print(rec_list)
        
    @staticmethod
    def get_fitness(popu, score, cost):
        num_env, _ = score.shape
        raw = np.sum(expit(1.6 * score - 8), axis=0)
        prob = 0.95 ** (num_env - raw)
        leng = popu.get_length(popu.array)
        cost = 1 / (1 + cost * leng)
        return prob * cost
        
    @staticmethod
    def reproduce(pop, envir, score, num):
        """Using the score function to get the offspring for every generation."""
        num_pop, leng_m = pop.shape
        para = num * score / np.sum(score)
        num_off = np.random.poisson(para)
        if np.sum(num_off) == 0:
            raise ValueError('Population died out.')
        ind = np.repeat(np.arange(num_pop), num_off)
        return pop[ind, :]
        
    @staticmethod
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
        return score_mat

if __name__ == '__main__':
    num_env = 10
    #env_m = 0.5 * np.random.randint(0, 2, size=(num_env, 1))
    env_m = 0.6 * np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    env_j = np.zeros((num_env, num_env))
    evolution = Evolution(num_population=100, num_environment=num_env,
                                 evolution_time=5000,
                                 initial_length=20, max_length=250,
                                 environment_length=10,
                                 length_cost=0.0005,
                                 mutation_rate=0.01, transpose_rate=0.05,
                                 transpose_length=3,
                                 environment_change_rate=0.002, 
                                 environment_m = env_m,
                                 environment_j = env_j)
    record = evolution.record
    record.plot_process('mean')
    record.plot_process('mean_score')
    record.plot_process('logit')
