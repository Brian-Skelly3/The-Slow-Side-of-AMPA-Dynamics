"""
A single neuron with certain Hz of stimulation
"""

# Hodgkin-Huxey network
import numpy as np
from numba import njit

# Excitatory parameters
Cm = 1
E_l = -70
E_na = 55
E_k = -80
g_l = 0.025
g_na = 60
g_k = 3
vsyn = 0
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


# Coupled Hodgkin-Huxley
@njit
def RHS(x, glutamate, tauK, taug, A1, A2, alpha, taux, tauu, U_0, k):
    
    m = x[0]
    n = x[1]
    h = x[2]
    v = x[3]
    X = x[4]
    U = x[5]
    K = x[6]
    g = x[7]

    # Bakcground drive
    I=0.08
      
    dm = alpha_m(v)*(1-m)-beta_m(v)*m
    dn = alpha_n(v)*(1-n)-beta_n(v)*n
    dh = alpha_h(v)*(1-h)-beta_h(v)*h

    dv = (-(g_na*(m**3)*h*(v-E_na) + g_k*(n**4)*(v-E_k) + g_l*(v-E_l)) + I + k*(K+g)*(vsyn-v))/Cm


    dU = (U_0 - U)/tauu + (U_0 * (1.0 - U))*glutamate
    
    dX = (1-X)/taux - (alpha * U * X)*glutamate
        
    dK = -K/tauK + (A1 * U * X)*glutamate
    dg = -g/taug + (A2 * U * X)*glutamate


    return np.array([dm, dn, dh, dv, dX, dU, dK, dg])


def HHkin(T, dt, freq, tauK, taug, A1, A2, alpha, taux, tauu, U_0, k, amp, var, on, seed):
    
    M = int(T / dt)                   # number of time steps
    t = np.linspace(0, T, M + 1)    
    
    np.random.seed(seed)
    
    # Initial conditions
    m = 0
    n = 0
    h = 1
    v = np.zeros((M + 1))
    v[0] = -70

    spikes =  0

    x = np.array([m, n, h, v[0], 1, 0, 0, 0])

    # Loop over time steps
    # Simple Euler method
    for i in range(M):
        
        # Set the spike at certain frequencies between 100 and 2100ms
        if t[i]%(1000.0/freq)<=1 and t[i]>100.0 and t[i]<2100:
              glutamate = on
        else:
              glutamate = 0
              
        # Update by time step
        x = x + dt*RHS(x,glutamate, tauK, taug, A1, A2, alpha, taux, tauu, U_0, k)

        # Adding noise to v
        x[3] += amp*np.random.poisson(var*(dt))   

        # Check for a spike
        if x[3]>0 and v[i]<0:
            spikes += 1 
        
        # Update variables we wish to keep track of
        v[i+1] = x[3]
       
    return spikes


tf = 2000
dt = 0.001
freq = [5, 10, 20]


n = 10
reps = 10
seeds = np.arange(1,1+reps,1)

# -------------------------------------------------------------------------
# Poisson kick amplitude sweeps
# -------------------------------------------------------------------------

var = 0.5
amp_range = np.linspace(0.3,1.2,n)

spikes_5_U = np.zeros((n, reps))
spikes_10_U = np.zeros((n, reps))
spikes_20_U = np.zeros((n, reps))

spikes_5_T = np.zeros((n, reps))
spikes_10_T = np.zeros((n, reps))
spikes_20_T = np.zeros((n, reps))

spikes_5_N = np.zeros((n, reps))
spikes_10_N = np.zeros((n, reps))
spikes_20_N = np.zeros((n, reps))

for j, amp in enumerate(amp_range):
    for m in range(reps):
        
        
        # TARPless parameters (change parameter values to obtain increased/decreased parameter curves)
        tauK, taug, A1, A2, alpha = [9.28782606e-01, 4.02802597e+02, 1.53943493e+00, 1.51940434e-03, 1]
        taux, tauu, U_0 = [426.05575527,   0.66710861,   0.62728]
        k=0.3 # TARPless coupling
        
        spikes_5_N[j,m] = HHkin(tf, dt, 5, tauK, taug, A1, A2, alpha, taux, tauu, U_0, k, amp, var, 0, seeds[m])
        spikes_10_N[j,m] = HHkin(tf, dt, 10, tauK, taug, A1, A2, alpha, taux, tauu, U_0, k, amp, var, 0, seeds[m])
        spikes_20_N[j,m] = HHkin(tf, dt, 20, tauK, taug, A1, A2, alpha, taux, tauu, U_0, k, amp, var, 0, seeds[m])
              
        spikes_5_U[j,m] = HHkin(tf, dt, 5, tauK, taug, A1, A2, alpha, taux, tauu, U_0, k, amp, var, 1, seeds[m])
        spikes_10_U[j,m] = HHkin(tf, dt, 10, tauK, taug, A1, A2, alpha, taux, tauu, U_0, k, amp, var, 1, seeds[m])
        spikes_20_U[j,m] = HHkin(tf, dt, 20, tauK, taug, A1, A2, alpha, taux, tauu, U_0, k, amp, var, 1, seeds[m])
        
        
        # TARPed parameters (change parameter values to obtain increased/decreased parameter curves)
        tauK, taug, A1, A2, alpha = [9.02833091, 64.51534201,  1.23175149,  0.23848164, 1]
        taux, tauu, U_0 = [58.2495233,   0.56018434,  0.93773]
        k=0.03 #TARPed coupling
        
        spikes_5_T[j,m] = HHkin(tf, dt, 5, tauK, taug, A1, A2, alpha, taux, tauu, U_0, k, amp, var, 1, seeds[m])
        spikes_10_T[j,m] = HHkin(tf, dt, 10, tauK, taug, A1, A2, alpha, taux, tauu, U_0, k, amp, var, 1, seeds[m])
        spikes_20_T[j,m] = HHkin(tf, dt, 20, tauK, taug, A1, A2, alpha, taux, tauu, U_0, k, amp, var, 1, seeds[m])
        
        print(f'Rep {m} of amp {amp} Finished ')
    print(f'Amp {amp} Finished \n')

# Save Poisson kick amplitude data
np.savetxt('increasing_noise_amplitude/amplitude_range.csv', amp_range, delimiter=',', fmt='%f')

np.savetxt('increasing_noise_amplitude/firing_rates_no_input_5Hz.csv', spikes_5_N/(tf/1000), delimiter=',', fmt='%f')
np.savetxt('increasing_noise_amplitude/firing_rates_no_input_10Hz.csv', spikes_10_N/(tf/1000), delimiter=',', fmt='%f')
np.savetxt('increasing_noise_amplitude/firing_rates_no_input_20Hz.csv', spikes_20_N/(tf/1000), delimiter=',', fmt='%f')

np.savetxt('increasing_noise_amplitude/firing_rates_TARPless_20Hz.csv', spikes_20_U/(tf/1000), delimiter=',', fmt='%f')
np.savetxt('increasing_noise_amplitude/firing_rates_TARPless_10Hz.csv', spikes_10_U/(tf/1000), delimiter=',', fmt='%f')
np.savetxt('increasing_noise_amplitude/firing_rates_TARPless_5Hz.csv', spikes_5_U/(tf/1000), delimiter=',', fmt='%f')

np.savetxt('increasing_noise_amplitude/firing_rates_TARPed_5Hz.csv', spikes_5_T/(tf/1000), delimiter=',', fmt='%f')
np.savetxt('increasing_noise_amplitude/firing_rates_TARPed_10Hz.csv', spikes_10_T/(tf/1000), delimiter=',', fmt='%f')
np.savetxt('increasing_noise_amplitude/firing_rates_TARPed_20Hz.csv', spikes_20_T/(tf/1000), delimiter=',', fmt='%f')


print('Amplitude run finished \n \n')


# -------------------------------------------------------------------------
# Poisson rate sweeps
# -------------------------------------------------------------------------

amp = 0.5
var_range = np.linspace(1,1.9,n)

spikes_5_U = np.zeros((n, reps))
spikes_10_U = np.zeros((n, reps))
spikes_20_U = np.zeros((n, reps))

spikes_5_T = np.zeros((n, reps))
spikes_10_T = np.zeros((n, reps))
spikes_20_T = np.zeros((n, reps))

for j, var in enumerate(var_range):
    for m in range(reps):
                
        # TARPless parameters (change parameter values to obtain increased/decreased parameter curves)
        tauK, taug, A1, A2, alpha = [9.28782606e-01, 4.02802597e+02, 1.53943493e+00, 1.51940434e-03, 1]
        taux, tauu, U_0 = [426.05575527,   0.66710861,   0.62727836]
        k=0.3 # TARPless coupling
        
        spikes_5_N[j,m] = HHkin(tf, dt, 5, tauK, taug, A1, A2, alpha, taux, tauu, U_0, k, amp, var, 0, seeds[m])
        spikes_10_N[j,m] = HHkin(tf, dt, 10, tauK, taug, A1, A2, alpha, taux, tauu, U_0, k, amp, var, 0, seeds[m])
        spikes_20_N[j,m] = HHkin(tf, dt, 20, tauK, taug, A1, A2, alpha, taux, tauu, U_0, k, amp, var, 0, seeds[m])
        
        spikes_5_U[j,m] = HHkin(tf, dt, 5, tauK, taug, A1, A2, alpha, taux, tauu, U_0, k, amp, var, 1, seeds[m])
        spikes_10_U[j,m] = HHkin(tf, dt, 10, tauK, taug, A1, A2, alpha, taux, tauu, U_0, k, amp, var, 1, seeds[m])
        spikes_20_U[j,m] = HHkin(tf, dt, 20, tauK, taug, A1, A2, alpha, taux, tauu, U_0, k, amp, var, 1, seeds[m])
        
        # TARPed parameters (change parameter values to obtain increased/decreased parameter curves)
        tauK, taug, A1, A2, alpha = [9.02833091, 64.51534201,  1.23175149,  0.23848164, 1]
        taux, tauu, U_0 = [58.2495233,   0.56018434,  0.93773379]
        k=0.03 #TARPed coupling
        
        spikes_5_T[j,m] = HHkin(tf, dt, 5, tauK, taug, A1, A2, alpha, taux, tauu, U_0, k, amp, var, 1, seeds[m])
        spikes_10_T[j,m] = HHkin(tf, dt, 10, tauK, taug, A1, A2, alpha, taux, tauu, U_0, k, amp, var, 1, seeds[m])
        spikes_20_T[j,m] = HHkin(tf, dt, 20, tauK, taug, A1, A2, alpha, taux, tauu, U_0, k, amp, var, 1, seeds[m])
        
        print(f'Rep {m} of var {var} Finished ')
    print(f'Var {var} Finished ')

# Save Poisson rate data
np.savetxt('increasing_noise_variance/variance_range.csv', var_range, delimiter=',', fmt='%f')

np.savetxt('increasing_noise_variance/firing_rates_no_input_5Hz.csv', spikes_5_N/(tf/1000), delimiter=',', fmt='%f')
np.savetxt('increasing_noise_variance/firing_rates_no_input_10Hz.csv', spikes_10_N/(tf/1000), delimiter=',', fmt='%f')
np.savetxt('increasing_noise_variance/firing_rates_no_input_20Hz.csv', spikes_20_N/(tf/1000), delimiter=',', fmt='%f')

np.savetxt('increasing_noise_variance/firing_rates_TARPless_5Hz.csv', spikes_5_U/(tf/1000), delimiter=',', fmt='%f')
np.savetxt('increasing_noise_variance/firing_rates_TARPless_10Hz.csv', spikes_10_U/(tf/1000), delimiter=',', fmt='%f')
np.savetxt('increasing_noise_variance/firing_rates_TARPless_20Hz.csv', spikes_20_U/(tf/1000), delimiter=',', fmt='%f')

np.savetxt('increasing_noise_variance/firing_rates_TARPed_5Hz.csv', spikes_5_T/(tf/1000), delimiter=',', fmt='%f')
np.savetxt('increasing_noise_variance/firing_rates_TARPed_10Hz.csv', spikes_10_T/(tf/1000), delimiter=',', fmt='%f')
np.savetxt('increasing_noise_variance/firing_rates_TARPed_20Hz.csv', spikes_20_T/(tf/1000), delimiter=',', fmt='%f')

print('Variance run finished ')

