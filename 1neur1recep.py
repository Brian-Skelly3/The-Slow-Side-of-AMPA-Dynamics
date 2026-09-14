# -*- coding: utf-8 -*-
"""
Created on Monday Jan 5th 14:23:40 2026

A single neuron with certain Hz of stimulation

numba is a librarry that runs the code in a c compiler somehow, so it's much faster



@author: brian
"""

# Hodgkin-Huxey network

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import joblib
import time
from numba import njit
time_start = time.time()


np.random.seed(123)

# Neuron parameters
# I_ext = 0.7 #Applied Current
vsyn = 0


# Excitatory parameters
Cm = 1
E_l = -70
E_na = 55
E_k = -80
g_l = 0.025
g_na = 60
g_k = 3
vt = -45  # Will definitely spike past this param
vth = 0
vr = -48.9
vspike = 54





@njit
def alpha_n(v):
    return (-0.032*(v-vt-15))/(np.exp(-(v-vt-15)/5)-1)
@njit
def alpha_m(v):
    return (-0.32*(v-vt-13))/(np.exp(-(v-vt-13)/4)-1)
@njit
def alpha_h(v):
    return 0.128*np.exp(-(v-vt-17)/18)
@njit
def beta_n(v):
    return 0.5*np.exp(-(v-vt-10)/40)
@njit
def beta_m(v):
    return (0.28*(v-vt-40))/(np.exp((v-vt-40)/5)-1)
@njit
def beta_h(v):
    return 4/(1+np.exp(-(v-vt-40)/5))


@njit
def RHS(x,glutamate):
    
    m = x[0]
    n = x[1]
    h = x[2]
    v = x[3]
    X = x[4]
    U = x[5]
    K = x[6]
    g = x[7]
    
    # eta=0.45 #-52mV
    # eta=0.2 #-60mV
    eta=0.08 #-67mV
    # eta=0

    # The different eta values essentially set the resting potentials without noise differentls
    # Pampaloni had it set to -67mV for their voltage clamped experiments

      
    dm = alpha_m(v)*(1-m)-beta_m(v)*m
    dn = alpha_n(v)*(1-n)-beta_n(v)*n
    dh = alpha_h(v)*(1-h)-beta_h(v)*h

    dv = (-(g_na*(m**3)*h*(v-E_na) + g_k*(n**4)*(v-E_k) + g_l*(v-E_l)) + eta + k*(K+g)*(vsyn-v))/Cm


    # U_plus = U #+ (U_0 * (1.0 - U))*glutamate
    dU = (U_0 - U)/tauu + (U_0 * (1.0 - U))*glutamate
    
    dX = (1-X)/taux - (alpha * U * X)*glutamate
        
    dK = -K/tauK + (A1 * U * X)*glutamate
    dg = -g/taug + (A2 * U * X)*glutamate


    return np.array([dm, dn, dh, dv, dX, dU, dK, dg])






def HHkin(T=300, dt=0.001, freq=20):
    
    M = int(T / dt)                                 # number of time steps
    t = np.linspace(0, T, M + 1)    
    
    np.random.seed(123)
    

    # Initial conditions
    m = 0
    n = 0
    h = 1
    v = np.zeros((M + 1))
    np.random.seed(123)
    v[0] = -70
    
    X = np.ones((M + 1))
    U = np.zeros((M + 1))+U_0
    K = np.zeros((M + 1))
    g = np.zeros((M + 1))
    

    

    x = np.array([m, n, h, v[0], X[0], U[0], K[0], g[0]])

    # Loop over time steps
    # Simple Euler method

    for i in range(M):
        
        # Set the spike at certain frequencies between 100 and 2100ms
        if t[i]%(1000.0/freq)<=1 and t[i]>100.0 and t[i]<2100:
              glutamate = 1
        else:
              glutamate = 0
              
        # r=1000 # Using dt instad of r so that even if the dt changes, it is the same amount of stimulatinos per second
        x = x + dt*RHS(x,glutamate)



        # Adding noise to v, K, and g
        # These are
        x[3] += 0.1*np.random.poisson(3*(dt))   #np.sqrt(dt)*np.random.normal(0.0, 0.1) # v
        x[6] += 0.02*np.random.poisson(0.05*dt)   #np.sqrt(dt)*(np.random.normal(0.0, 0.006)) #  K
        x[7] += 0.0001*np.random.poisson(0.01*dt)   #np.sqrt(dt)*(np.random.normal(0.0, 0.0006)) #  g

        
        # Update variables we wish to keep track of

        v[i+1] = x[3]
        X[i+1] = x[4]
        U[i+1] = x[5]
        K[i+1] = x[6]
        g[i+1] = x[7]
       
    return t, v, X, U, K, g



def saving(tf=300, dt=0.001, freq=20):
    t, v, X, U, K, g = HHkin(tf, dt, freq)
    print('starting')
    np.savez(f'1Neur_{str(freq).replace(".","_")}Hz_{str(tf).replace(".","_")}ms_TARPed_minus67mV_x0_9noise', t=t, V=v)
    print('figure')
    plt.figure()
    plt.plot(t[:], v[:])
    plt.xlabel('Time')
    plt.ylabel('Voltages')
    # plt.ylim(-60, -40)
    plt.savefig(f'1Neur_{str(freq).replace(".","_")}Hz_{str(tf).replace(".","_")}ms_TARPed_V_minus67mV_x0_9noise')

    
    plt.figure()
    plt.plot(t, K+g)
    plt.xlabel('Time')
    plt.ylabel('Conductances')
    plt.savefig(f'1Neur_{str(freq).replace(".","_")}Hz_{str(tf).replace(".","_")}ms_TARPed_Kg_minus67mV_x0_9noise')

    plt.figure()
    plt.plot(t, X, label='X')
    plt.plot(t, U, label='U')
    plt.xlabel('Time')
    plt.ylabel('X and U')
    plt.legend()
    plt.savefig(f'1Neur_{str(freq).replace(".","_")}Hz_{str(tf).replace(".","_")}ms_TARPed_XU_minus67mV_x0_9noise')

    print(f'Finished {freq}Hz')



# TARPless parameters
# tauK, taug, A1, A2, alpha = [9.28782606e-01, 4.02802597e+02, 1.53943493e+00, 1.51940434e-03, 1]
# taux, tauu, U_0 = [426.05575527,   0.66710861,   0.62727836]
# k=0.3 # TARPless coupling


# TARPed parameters
tauK, taug, A1, A2, alpha = [9.02833091, 64.51534201,  1.23175149,  0.23848164, 1]
taux, tauu, U_0 = [58.2495233,   0.56018434,  0.93773379]
k=0.03 #TARPed coupling

tf = 2100
dt = 0.001
freq = [5, 10, 20]


joblib.Parallel(n_jobs=3)(joblib.delayed(saving)(tf, dt, value) for value in freq)

time_end = time.time()

print((time_end-time_start)//60, 'm ', 60*(((time_end-time_start)/60)-((time_end-time_start)//60)), 's')