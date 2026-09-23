"""
This is currently for either a 100% TAARPless or TARPed regime depending on parameters chosen

Network of Hodgkin-Huxley neurons that are randomely coupled with the Kinetic Receptor model

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
vt = -45  # Will definitely spike past this param
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


# Cuopled Hodgkin-Huxley
@njit
def RHS(x, N, k):
    
    m = x[0:N]
    n = x[1*N:2*N]
    h = x[2*N:3*N]
    v = x[3*N:4*N]                  # v
    s = x[4*N:4*N + N*N].reshape(N,N)
    Rf = x[4*N + N*N : 4*N + 2*N*N].reshape(N,N)
    RGf = x[4*N + 2*N*N : 4*N + 3*N*N].reshape(N,N)
    O1f = x[4*N + 3*N*N : 4*N + 4*N*N].reshape(N,N)
    C1f = x[4*N + 4*N*N : 4*N + 5*N*N].reshape(N,N)
    D1f = x[4*N + 5*N*N : 4*N + 6*N*N].reshape(N,N)
    O2f = x[4*N + 6*N*N : 4*N + 7*N*N].reshape(N,N)
    C2f = x[4*N + 7*N*N : 4*N + 8*N*N].reshape(N,N)
    D2f = x[4*N + 8*N*N : 4*N + 9*N*N].reshape(N,N)
    
    dm = alpha_m(v)*(1-m)-beta_m(v)*m
    dn = alpha_n(v)*(1-n)-beta_n(v)*n
    dh = alpha_h(v)*(1-h)-beta_h(v)*h

    dv = (-(g_na*(m**3)*h*(v-E_na) + g_k*(n**4)*(v-E_k) + g_l*(v-E_l)) + I_ext + 2*k/N*np.sum(O1f+2*O2f, axis=0)*(vsyn-v))/Cm

    ds = (-s/tauS)

    dRf = k_minus1*RGf - briansconst*s*k_plus1*Rf
    dRGf = briansconst*s*k_plus1*Rf - k_minus1*RGf + CO*C1f - CC*RGf

    dO1f = beta*C1f - alpha*O1f
    dC1f = CC*RGf - CO*C1f + alpha*O1f - beta*C1f + gamma1*D1f - delta1*C1f + CS_minus1*C2f - CS_plus1*C1f
    dD1f = delta1*C1f - gamma1*D1f
    
    dO2f = beta*C2f - alpha*O2f
    dC2f = CS_plus1*C1f - CS_minus1*C2f + alpha*O2f - beta*C2f + gamma2*D2f - delta2*C2f
    dD2f = delta2*C2f - gamma2*D2f


   
    return np.concatenate((
                dm.ravel(),
                dn.ravel(),
                dh.ravel(),
                dv.ravel(),
                ds.ravel(),
                dRf.ravel(),
                dRGf.ravel(),
                dO1f.ravel(),
                dC1f.ravel(),
                dD1f.ravel(),
                dO2f.ravel(),
                dC2f.ravel(),
                dD2f.ravel()
    ))




@njit
def HHkin(N=2, T=300, dt=0.001, k=1, j=1):
    
    M = int(T / dt)                                 # number of time steps
    
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
    
    x = np.concatenate((
           m.ravel(),
           n.ravel(),
           h.ravel(),
           v.ravel(),    # flatten 1xN to N
           s.ravel(),    # NxN
           np.zeros((N, N)).ravel(),    # NxN
           np.zeros((N, N)).ravel(),    # NxN
           np.zeros((N, N)).ravel(),    # NxN
           np.zeros((N, N)).ravel(),     # NxN
           np.zeros((N, N)).ravel(),    # NxN
           np.zeros((N, N)).ravel(),    # NxN
           np.zeros((N, N)).ravel(),    # NxN
           np.zeros((N, N)).ravel()     # NxN
       ))
 

    # Loop over time steps
    # Simple Euler method


    # 4*N used a lot. This makes it neater
    jump = 4*N
    # 4*N is the length of m,n,h, and v combined
    
    for i in range(M):
        # Iterate all variable to end of timestep
        x_new = x + dt*RHS(x, N, k)

        # Add noise
        x_new[3*N:4*N] += 0.5*np.random.poisson(dt, size=N)   
    
       
        # Determine which, if any, neurons has reached threshold
        spike = np.where((x_new[3*N:4*N] >= vth) & (x[3*N:4*N] < vth)) 

        
        # if at least one neuron reached threshold
        if len(spike[0]) > 0:
            
            for spikeInd in spike[0]:

                # add spike pulse
                s_temp = x_new[jump:jump+N*N].reshape(N, N)
                if np.sum(W[spikeInd, :]) != 0.0:
                    s_temp[spikeInd, :] += amp*W[spikeInd, :] #Multipled by amp as stated earlier
                x_new[jump:jump+N*N] = s_temp.ravel()
                
                
            # Update all variables for next time step
        x = x_new
        
    return
    # return spike_times[:spikeCounter, :]




# Uncomment the parameters you wish to use

# TARPless
alpha = 6   # Deactivation rate
beta = 10    # Activation rate
k_plus1 = 1e4 # Binding rate of molecule to AMPAR (R to RG)
k_minus1 = 50  # Unbinding rate of molecule from AMPAR (RG to R)
CC = 36.5      # Transition from RG to RG* (channel activating)
CO = 4.55     # Transition from RG* to RG (Channel deactivating)
CS_plus1 = 0.3   # Transition from RG* to RG*2
CS_minus1 = 10.0 # Transition from desensitized state RG*2 to RG*
gamma1  = .003 # Recovery rate
gamma2 = .03
delta1 = 1.1    # Desensitization rate from D1 to RG*
delta2 = .3    # Desensitization rate from D2 to RG*2
k=0.08

# # #TARPed
# alpha = 6   # Deactivation rate
# beta = 100    # Activation rate
# k_plus1 = 1e4  # Binding rate of molecule to AMPAR (R to RG)
# k_minus1 = 50  # Unbinding rate of molecule from AMPAR (RG to R)
# CC = 25      # Transition from RG to RG* (channel activating)
# CO = 2.5     # Transition from RG* to RG (Channel deactivating)
# CS_plus1 = 1   # Transition from RG* to RG*2
# CS_minus1 = 3.4  # Transition from desensitized state RG*2 to RG*
# gamma1 = .03  # Recovery rate
# gamma2 = .03
# delta1 = 1.1    # Desensitization rate from D1 to RG*
# delta2 = .3    # Desensitization rate from D2 to RG*2
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
        
        # spiketimes = HHkin(N, tf, dt, k, seeds[j])   # For firing rates
        HHkin(N, tf, dt, k, seeds[j])   # For run times

        time_end = time.time()
        
        # firing_rates[i,j] = len(spiketimes)/(N*tf)*1000   # For firing rates
        run_time[i,j] = time_end - time_start   # For run times
    
        print((time_end-time_start)//60, 'm ', 60*(((time_end-time_start)/60)-((time_end-time_start)//60)), 's')
    print('N = '+str(N)+' complete')
    

# np.savetxt('Data/Network simulations/HH_network_KR_model_firing_rates.csv', firing_rates, delimiter=',', fmt='%f')   # For firing rates
np.savetxt('Data/Network simulations/HH_network_KR_model_run_times.csv',run_time, delimiter=',', fmt='%f')   # For run times


neur = [500]
reps = 3

run_time = np.zeros((len(neur),reps))
firing_rates = np.zeros((len(neur),reps))

seeds = np.arange(reps)

for i, N in enumerate(neur):
    for j in range(reps):
    
        time_start = time.time()
        
        # spiketimes = HHkin(N, tf, dt, k, seeds[j])   # For firing rates
        HHkin(N, tf, dt, k, seeds[j])   # For run times

        time_end = time.time()
        
        # firing_rates[i,j] = len(spiketimes)/(N*tf)*1000   # For firing rates
        run_time[i,j] = time_end - time_start   # For run times
    
        print((time_end-time_start)//60, 'm ', 60*(((time_end-time_start)/60)-((time_end-time_start)//60)), 's')
    print('N = '+str(N)+' complete')
    

# np.savetxt('Data/Network simulations/HH_network_KR_model_firing_rates_500.csv', firing_rates, delimiter=',', fmt='%f')   # For firing rates
np.savetxt('Data/Network simulations/HH_network_KR_model_run_times_500.csv',run_time, delimiter=',', fmt='%f')   # For run times


