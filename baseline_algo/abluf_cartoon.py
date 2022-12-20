# -*- coding: utf-8 -*-
"""
ABLUF algortihm
"""

import numpy as np
import scipy as sp
import scipy.integrate
import random
from collections import deque
import matplotlib.pyplot as plt
import cv2
import time
import os
from util import show_image, gau, get_action, get_p #environment

os.makedirs(r'data',exist_ok=True)

def gau_sig(x,mu,sigma):
    g_div = ((x-mu)**2) * np.exp(-(x-mu)**2/(2*sigma**2)) 
    return g_div

def EMupdate(h,lamda,sigma,h_n):
    temp = np.zeros(N_ACT)
    a = np.zeros(N_S)
    for o in range(N_S):
        for i in range(N_ACT):
            temp[i] = sp.integrate.dblquad(fun1, 0, 1, lambda x: 0.01, lambda x: 1, args=(h,lamda,i,o,sigma,h_n), epsrel = 1e-2, epsabs = 1e-2)[0]
        try:
            tem = [i for i, j in enumerate(temp) if j == max(temp)] #  
            a[o] = tem[np.random.randint(len(tem))]
        except:
            a[o] = np.argmax(temp)
    return a

def fun1(x,y,h,lamda,a_n,o,sigma,h_n):
    # x: mu_plus y: mu_minis
    record_h = 1
    record_a = 0
    for i in h:
        if i[2] == 0:
            record_h = record_h*fun_po(i[1],lamda[i[0]],x,sigma)
        if i[2] == 1:
            record_h = record_h*fun_ne(i[1],lamda[i[0]],y,sigma)
        else:
            record_h = record_h*fun_0(i[1],lamda[i[0]],x,y,sigma)
        if i[0] == o:
            if i[2] == 0:
                record_a = record_a+fun_po_log(i[1],a_n,x,sigma)
            if i[2] == 1:
                record_a = record_a+fun_ne_log(i[1],a_n,y,sigma)
            else:
                record_a = record_a+fun_0_log(i[1],a_n,x,y,sigma)
    return record_h*record_a

def fun_po(a_o,a,x,sigma):        
    return gau(a_o,a,sigma)*(1-x)
    
def fun_ne(a_o,a,y,sigma):
    return (1-gau(a_o,a,sigma)*0.99)*(1-y)

def fun_0(a_o,a,x,y,sigma):
    return 1-fun_po(a_o,a,x,sigma)-fun_ne(a_o,a,y,sigma)

def fun_po_log(a_o,a,x,sigma):        
    return np.log(gau(a_o,a,sigma))#+np.log(1-x)
    
def fun_ne_log(a_o,a,y,sigma):
    return np.log(1-gau(a_o,a,sigma)*0.99)#+np.log(1-y)

def fun_0_log(a_o,a,x,y,sigma):
    return np.log(1-fun_po(a_o,a,x,sigma)-fun_ne(a_o,a,y,sigma))

def get_grad(h_n,lamda,weight=1):
    grad = np.zeros(2)
    for s_t in range(N_S): # compute the gradient
        a_ba = np.argmax(abs([i for i in range(N_ACT)] - lamda[s_t]))
        a_la = int(lamda[s_t])
        for a_t in range(N_ACT):            
            if a_t != lamda[s_t]:
                if sum(h_n[s_t,a_t,:]) != 0 and h_n[s_t,a_la,0] !=0:
                    re0 = (h_n[s_t,a_t,0]/sum(h_n[s_t,a_t,:]))/(h_n[s_t,a_la,0]/sum(h_n[s_t,a_la,:]))                   
                    if re0<1:
#                        grad[0] += -weight[s_t,a_t] * (gau(a_t,a_la,sigma)-re0) * gau_sig(a_t,a_la,sigma)
                        grad[0] += -(gau(a_t,a_la,sigma)-re0) * gau_sig(a_t,a_la,sigma)
            if a_t != a_ba:
                if sum(h_n[s_t,a_t,:]) != 0 and h_n[s_t,a_ba,1] !=0:
                    re1 = (h_n[s_t,a_t,1]/sum(h_n[s_t,a_t,:]))/(h_n[s_t,a_ba,1]/sum(h_n[s_t,a_ba,:]))                    
                    if re1<1:
#                        grad[1] += weight[s_t,a_t] * (1 - 0.99*gau(a_t,a_la,sigma) - re1) * gau_sig(a_t,a_la,sigma)
                        grad[1] += (1 - 0.99*gau(a_t,a_la,sigma) - re1) * gau_sig(a_t,a_la,sigma)
    return sum(grad)/(N_ACT*N_S)

# MAIN

#Constantes :
E = 0
STEP = 75 
ALPHA = 0.4 #learning rate
N_ACT = 6 # number of actions
N_S = 4 # number of states
dog_size = 100 # size of figures
rat_size = 50
length = 500 # side length of background

PI = [np.mod(i,N_ACT) for i in range(N_S)] # PI = [0, 1, 2, 3]
p_act = get_p(PI,N_S,N_ACT)
print("p_act :", p_act)
jus = np.zeros(5)
num = 0
correct = 0
num_act = 0

sigma = 3 #np.ones(2)*3 # parameter .5 or 1
lamda = np.random.randint(N_ACT,size=N_S)
hist = deque() # historical records
h_n = np.zeros([N_S,N_ACT,3]) # counting the feedback number respect to o,a
t = 0
done = 0
s = 0 # np.random.randint(0,N_S)
step_t = 0
step_record = []
catch_record = []
catch_states = 0
distance_optimal = []
dog_action_list = []
optimal_policy = []

automatic_feed_back = []

while done != 1:
    start = time.perf_counter()
    a = int(lamda[s])
    dog_action = a
    rat_action = get_action(p_act[s],N_ACT)

    print("dog_action :", dog_action)
    print("rat_action :", rat_action)
    print("optimal_policy :", PI[s])

    dog_action_list.append(dog_action)
    optimal_policy.append(PI[s])

    f, result = show_image(s,rat_action,dog_action,dog_size,rat_size,length, N_ACT, PI)
    step_t += 1

    if dog_action == rat_action :
        catch_states += 1

    if f == 3 or step_t >= 16 : #Finish one state
        s += 1
        step_record.append(step_t)
        catch_record.append(catch_states)
        catch_states = 0
        step_t = 0
        distance_optimal.append(np.linalg.norm(np.array(dog_action_list)-np.array(optimal_policy)))
        dog_action_list = []
        optimal_policy = []
        automatic_feed_back = []
        if s >= N_S :
            print('finish experiment')
            break
        else:
            continue

    hist.append((s,a,f))
    if len(hist) > STEP: # limitation of memory
        hist.popleft()
    h = hist.copy()
    h_n[s,a,f] += 1
    lamda = np.random.randint(N_ACT,size=N_S)
    lamda_t = [num+1 if num<N_ACT-1 else 0 for num in lamda]
    n_l = 0

# update of abluf --------------------------------------------------
    while (lamda != lamda_t).any() and n_l < 3: # EM algorithm for \lambda and \mu
        lamda_t = lamda.copy()
        lamda = EMupdate(h,lamda,sigma,h_n)
        n_l += 1
    if t > 0:
        grad = get_grad(h_n,lamda)
        sigma += ALPHA * grad # update of \sigma

    # temps d'execution
    end = time.perf_counter() 
    print(t,':',end-start)

    t = t + 1


#--------------------------------------------------#

# record :
step_record = np.array(step_record)
catch_record = np.array(catch_record)
distance_optimal = np.array(distance_optimal)
i = 2
# np.save(r'data\step_record_no_feedback_' + str(i) + '.npy', step_record)
# np.save(r'data\catch_times_no_feedback_' + str(i) + '.npy', catch_record) 
# np.save(r'data\distance_optimal_no_feedback_' + str(i) + '.npy', distance_optimal)
np.save(r'data\step_record_' + str(i) + '.npy', step_record)
np.save(r'data\catch_times_' + str(i) + '.npy', catch_record) 
np.save(r'data\distance_optimal_' + str(i) + '.npy', distance_optimal)

#--------------------------------------------------#
