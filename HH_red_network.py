"""
This is currently for either a 100% TARPless or TARPed regime depending on parameters chosen

Network of Hodgkin-Huxley neurons that are randomely coupled with the Reduced model

Each connection from one neuron to another has its own set of receptors dynamics.

when indexing, [i, j, k] is the ith time, jth neuron connected to kth neuron.

@njit converts functions to optimised machine code so that it runs faster

To get run times, HHkin does not need to return anything.

For run times or firing rates, comment out the relavant, labelled lines in the final two for loops and the corresponding save functions.
Marked as "For firing rates," and "For run times."
"""

# Hodgkin-Huxey network

import numpy as np
import time
from numba import njit

# Neuron parameters
I_ext = 0.2
vsyn = 0
tauS = 0.3
amp = 2.5


# Excitatory parameters
Cm = 1
E_l = -70
E_na = 55
E_k = -80
g_l = 0.025
g_na = 60
g_k = 3
vt = -45
vth = 0
vr = -48.9
vspike = 54



# Ion channel dynamics
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



# Coupled Hodgkin-Huxely
@njit
def RHS(x, N, k):
    
    m = x[0:N]
    n = x[1*N:2*N]
    h = x[2*N:3*N]
    v = x[3*N:4*N]                  # v
    s = x[4*N:4*N + N*N].reshape(N,N)
    X = x[4*N + N*N : 4*N + 2*N*N].reshape(N,N)
    U = x[4*N + 2*N*N : 4*N + 3*N*N].reshape(N,N)
    K = x[4*N + 3*N*N : 4*N + 4*N*N].reshape(N,N)
    g = x[4*N + 4*N*N : 4*N + 5*N*N].reshape(N,N)
    
      
    dm = alpha_m(v)*(1-m)-beta_m(v)*m
    dn = alpha_n(v)*(1-n)-beta_n(v)*n
    dh = alpha_h(v)*(1-h)-beta_h(v)*h

    dv = (-(g_na*(m**3)*h*(v-E_na) + g_k*(n**4)*(v-E_k) + g_l*(v-E_l)) + I_ext + 2*k/N*np.sum(K+g, axis=0)*(vsyn-v))/Cm

    ds = (-s/tauS)

    dU = (U_0 - U)/tauu + (U_0 * (1.0 - U))*s
    
    dX = (1-X)/taux - (alpha * U * X)*s
        
    dK = -K/tauK + (A1 * U * X)*s
    dg = -g/taug + (A2 * U * X)*s
    
    
    return np.concatenate((
                dm.ravel(),
                dn.ravel(),
                dh.ravel(),
                dv.ravel(),
                ds.ravel(),
                dX.ravel(),
                dU.ravel(),
                dK.ravel(),
                dg.ravel()
    ))


@njit
def HHkin(N=2, T=300, dt=0.001, k=np.array([0.01, 0.005]),j=1):
    
    M = int(T / dt)              # number of time steps
    
    np.random.seed(j)

    # Connectivity matrix
    W = np.random.uniform(0.0, 1.0, size=(N, N))
    for i in range(N):
        for j in range(N):
            if W[i, j] > 0.5:
                W[i, j] = 0.0
            else:
                W[i, j] = 1.0
            if i == j:
                W[i, j] = 0.0

    v = np.random.uniform(-45, -85, N)
    m = np.zeros((1, N))
    n = np.zeros((1, N))
    h = np.ones((1, N))
    s = np.zeros((N, N))
    X = np.ones((N, N))
    U = np.zeros((N, N))+U_0
    K = np.zeros((N, N))
    g = np.zeros((N, N))
    
    
    x = np.concatenate((
        m.ravel(),
        n.ravel(),
        h.ravel(),
        v.ravel(),    # flatten 1xN to N
        s.ravel(),    # NxN
        X.ravel(),    # NxN
        U.ravel(),    # NxN
        K.ravel(),    # NxN
        g.ravel()     # NxN
    ))
    
    

    # Loop over time steps
    # Simple Euler method


    # 4*N used a lot. This makes it neater
    jump = 4*N
    # 4*N is the length of m,n,h, and v combined

    for i in range(M):

        x_new = x + dt*RHS(x, N, k)

        x_new[3*N:4*N] += 0.5*np.random.poisson(dt, size=N)   
       
        # Determine which, if any, neurons has reached threshold
        spike = np.where((x_new[3*N:4*N] >= vth) & (x[3*N:4*N] < vth)) 

        
        # if at least one neuron reached threshold
        if len(spike[0]) > 0:
            # 1. Freeze the baseline start and end states for the step
            x_start = x.copy()
            x_end = x_new.copy()

            # loop over those neurons that spike
            for spikeInd in spike[0]:
                # # Use x_end and x_start for interpolation
                denom = x_end[3*N + spikeInd] - x_start[3*N + spikeInd]
                # No division by 0
                if abs(denom) < 1e-12:
                    denom = 1e-12
                
                # add spike pulse
                s_temp = x_new[jump:jump+N*N].reshape(N, N)
                if np.sum(W[spikeInd, :]) != 0.0:
                    s_temp[spikeInd, :] += amp*W[spikeInd, :] #Multipled by amp as stated earlier
                x_new[jump:jump+N*N] = s_temp.ravel()
                
            
        # Update all variables for next time step
        x = x_new
        
    return #spike_times[:spikeCounter, :]



# Uncomment whichever regime is being used. COmment out the other
# TARPless
tauK, taug, A1, A2, alpha = [9.28782606e-01, 4.02802597e+02, 1.53943493e+00, 1.51940434e-03, 1]
taux, tauu, U_0 = [426.05575527,   0.66710861,   0.62727836]
k=0.08

# #TARPed
# tauK, taug, A1, A2, alpha = [9.02833091, 64.51534201,  1.23175149,  0.23848164, 1]
# taux, tauu, U_0 = [58.24952329715077,   0.560184338771549, 0.9377337949271534]
# k=0.009



tf = 5000
dt = 0.001

neur = [10,20,50,100,200]
reps = 10

run_time = np.zeros((len(neur),reps))
firing_rates = np.zeros((len(neur),reps))

seeds = np.arange(reps)

for i, N in enumerate(neur):
    for j in range(reps):
    
        time_start = time.time()
        
        # spiketimes = HHkin(N, tf, dt, k, seeds[j])  # For firing rates
        HHkin(N, tf, dt, k, seeds[j])  # For run times

        time_end = time.time()
        
        # firing_rates[i,j] = len(spiketimes)/(N*tf)*1000  # For firing rates
        run_time[i,j] = time_end - time_start  # For run times
    
        print((time_end-time_start)//60, 'm ', 60*(((time_end-time_start)/60)-((time_end-time_start)//60)), 's')
    print('N = '+str(N)+' complete')
    

# np.savetxt('Data/Network simulations/HH_network_reduced_model_firing_rates.csv', firing_rates, delimiter=',', fmt='%f')  # For firing rates
np.savetxt('Data/Network simulations/HH_network_reduced_model_run_times.csv',run_time, delimiter=',', fmt='%f')  # For run times


neur = [500]
reps = 3

run_time = np.zeros((len(neur),reps))
firing_rates = np.zeros((len(neur),reps))

seeds = np.arange(reps)

for i, N in enumerate(neur):
    for j in range(reps):
    
        time_start = time.time()
        
        # spiketimes = HHkin(N, tf, dt, k, seeds[j])
        HHkin(N, tf, dt, k, seeds[j])  # For run times

        time_end = time.time()
        
        # firing_rates[i,j] = len(spiketimes)/(N*tf)*1000
        run_time[i,j] = time_end - time_start  # For run times
    
        print((time_end-time_start)//60, 'm ', 60*(((time_end-time_start)/60)-((time_end-time_start)//60)), 's')
    print('N = '+str(N)+' complete')
    

# np.savetxt('Data/Network simulations/HH_network_reduced_model_firing_rates_500.csv', firing_rates, delimiter=',', fmt='%f')
np.savetxt('Data/Network simulations/HH_network_reduced_model_run_times_500.csv',run_time, delimiter=',', fmt='%f')  # For run times


