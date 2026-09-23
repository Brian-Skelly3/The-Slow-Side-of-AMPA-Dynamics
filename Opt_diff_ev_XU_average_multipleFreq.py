"""
Performing 2nd optimisation on the factorisation and facilitation variables
"""

from scipy.integrate import solve_ivp
import numpy as np
from scipy.optimize import differential_evolution


#### Defining functions


# Kr model
def Milstein2007(t, y, CC=36.5, CO=4.55, CS_plus1=0.3, CS_minus1=10.0, gamma1 =.003, gamma2 = 0.03, beta=10, delta1=1.1, delta2=0.3, briansconst=1.0, burst_len=300):
    
    R = y[0]
    RG = y[1]
    O1 = y[2]
    C1 = y[3]
    D1 = y[4]
    O2 = y[5]
    C2 = y[6]
    D2 = y[7]



#   Periodic square pulses
    if t%(1000/5)<=1 and t<burst_len:
        glutamate = (0.01)
    elif t%(1000/(2*5))<=1 and t<2*burst_len and t>=burst_len:
        glutamate = (0.01)
    elif t%(1000/(4*5))<=1 and t<3*burst_len and t>=2*burst_len:
        glutamate = (0.01)
    elif t%(1000/(10*5))<=1 and t<4*burst_len and t>=3*burst_len:
        glutamate = (0.01)
    elif t%(1000/(20*5))<=1 and t<5*burst_len and t>=4*burst_len:
        glutamate = (0.01)
    elif t%(1000/(40*5))<=1 and t<6*burst_len and t>=5*burst_len:
        glutamate = (0.01)
    elif t%(1000/(60*5))<=1 and t<7*burst_len and t>=6*burst_len:
        glutamate = (0.01)
    elif t%(1000/(80*5))<=1 and t<8*burst_len and t>=7*burst_len:
        glutamate = (0.01)
    elif t%(1000/(100*5))<=1 and t<9*burst_len and t>=8*burst_len:
        glutamate = (0.01)
    else:
        glutamate = (0.0)

        
    
    dR_dt = k_minus1*RG - briansconst*glutamate*k_plus1*R 
    dRG_dt = briansconst*glutamate*k_plus1*R - k_minus1*RG + CO*C1 - CC*RG

    dC1_dt = CC*RG - CO*C1 + alpha*O1 - beta*C1 + gamma1*D1 - delta1*C1 + CS_minus1*C2 - CS_plus1*C1
    dO1_dt = beta*C1 - alpha*O1
    dD1_dt = delta1*C1 - gamma1*D1
    
    dC2_dt = CS_plus1*C1 - CS_minus1*C2 + alpha*O2 - beta*C2 + gamma2*D2 - delta2*C2
    dO2_dt = beta*C2 - alpha*O2
    dD2_dt = delta2*C2 - gamma2*D2

    
    return [dR_dt, dRG_dt, dO1_dt, dC1_dt, dD1_dt, dO2_dt, dC2_dt, dD2_dt]


# For KR model

R_0 = np.array([1.0])
RG_0 = np.array([0.0])
O1_0 = np.array([0.0])
C1_0 = np.array([0.0])
D1_0 = np.array([0.0])
O2_0 = np.array([0.0])
C2_0 = np.array([0.0])
D2_0 = np.array([0.0])

y3 = np.concatenate([R_0, RG_0, O1_0, C1_0, D1_0, O2_0, C2_0, D2_0])



# Setting the parameter values
# Some of these will change later

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
tauS = 3





# Reduced model
def time_2first_ord(t, x, taux=100, tauu=100, tauK=3, taug=5, A1=1, A2=1, alpha=1, U_0=0.5, burst_len=300):
    
    X = x[0]
    U = x[1]
    K = x[2]
    g = x[3]
    

#   Periodic square pulses

    if t%(5)<=1 and t<burst_len:
        glutamate = 1
    elif t%(1000/(2*5))<=1 and t<2*burst_len and t>=burst_len:
        glutamate = 1
    elif t%(1000/(4*5))<=1 and t<3*burst_len and t>=2*burst_len:
        glutamate = 1
    elif t%(1000/(10*5))<=1 and t<4*burst_len and t>=3*burst_len:
        glutamate = 1
    elif t%(1000/(20*5))<=1 and t<5*burst_len and t>=4*burst_len:
        glutamate = 1
    elif t%(1000/(40*5))<=1 and t<6*burst_len and t>=5*burst_len:
        glutamate = 1
    elif t%(1000/(60*5))<=1 and t<7*burst_len and t>=6*burst_len:
        glutamate = 1
    elif t%(1000/(80*5))<=1 and t<8*burst_len and t>=7*burst_len:
        glutamate = 1
    elif t%(1000/(100*5))<=1 and t<9*burst_len and t>=8*burst_len:
        glutamate = 1
    else:
        glutamate = (0.0)



    
    dU = (U_0 - U)/tauu + (U_0 * (1.0 - U))*glutamate
    
    dX = (1.0-X)/taux - (alpha * U * X)*glutamate

        
    dK = -K/tauK + (A1 * U * X)*glutamate
    dg = -g/taug + (A2 * U * X)*glutamate


    return np.array([dX, dU, dK, dg])




# Simulate the ODEs
def simulate(params, t_eval, x0):
    taux, tauu, tauK, taug, A1, A2, alpha, U_0 = params
    def odes(t, x):
        return time_2first_ord(t, x, taux, tauu, tauK, taug, A1, A2, alpha, U_0)
    # print(len(t_eval))
    sol = solve_ivp(odes, [t_eval[0], t_eval[-1]], x0, t_eval=t_eval, method='RK45')
    return sol.y





# Define the cost function
def cost_function_log(params, t_eval, x0, ydata, fixed_params):
    taux, tauu, U_0= params
    tauK, taug, A1, A2, alpha = fixed_params
    
    all_params = [taux, tauu, tauK, taug, A1, A2, alpha, U_0]
    
    sol = simulate(all_params, t_eval, x0)

    model_output = sol[2] + sol[3]


    ########################################
    # I have taken out the logs because now I actually do want to match the amplitudes as the decays are well matched
    #################################################
    error = np.sum(((model_output[1:]) - (ydata[1:]))**2)
    # Trying to take out the first value as it starts at 0. This stops the runtime warning
    return error









def get_av(TARPed=0, N=10):


    bounds = [(1, 900),     # tauX
            (0.5, 500),  # tauU
            (0.1, 1)]   # U_0


    if TARPed==0:
        # TARPless case

        fixed_params = [9.28782606e-01, 4.02802597e+02, 1.53943493e+00, 1.51940434e-03, 1] # From previous optimisation

        x0 = [1, 0.6, 0, 0]             # initial conditions


        solMilmul20 = solve_ivp(Milstein2007, [0, 3000.0], y3, args =  (36.5, 4.55, 0.3, 10.0, .003, 0.03, 10.0, 1.1, 0.3, 0.1), dense_output=True, rtol=1e-8, method='LSODA', max_step=0.1)
        xdata = solMilmul20.t
        ydata = solMilmul20.y[2] + 2*solMilmul20.y[5]
        print("For TARPless activity")


    if TARPed==1:
        # TARPed case

        fixed_params = [9.02833091, 64.51534201,  1.23175149,  0.23848164, 1] # From previous optimisation
        x0 = [1.0, 0.85, 0.0, 0.0]             # initial conditions

        solMil2mul20 = solve_ivp(Milstein2007, [0, 3000.0], y3, args = (25.0, 2.5, 1.0, 3.4, .03, 0.03, 100.0, 1.1, 0.3, 0.1), dense_output=True, rtol=1e-8, method='LSODA', max_step=0.1)
        xdata = solMil2mul20.t
        ydata = solMil2mul20.y[2] + 2*solMil2mul20.y[5]

        print("For TARPed activity")



    result = []

    for i in range(N):


        print('i = ', i)
        print('freq = ', freq)

        result2 = differential_evolution(
            cost_function_log, 
            bounds, 
            args=(xdata, x0, ydata, fixed_params, 3),
            maxiter=10000,
            popsize=150,
            workers=-1,      # parallelize if possible
            tol=1e-6,
            updating='deferred',
            polish=True
        )

        # Append the vector
        result.append(result2.x)

    # Convert once at the end
    result = np.array(result)

    print('tauX = ', np.mean(result[:,0]))
    print('tauU = ', np.mean(result[:,1]))
    print('U_0 = ', np.mean(result[:,2]))

    # np.save('Data/Optimised params/Opt_params_TARPed_XU_mixedfreq', result)


    

# "TARPed=0" for TARPless, "TARPed=1" for TARPed
TARPed=1
# If changing "TARPed," must change the save file name too
N=30


get_av(TARPed, N)
