#%%
from math import exp, gamma, sqrt
import numpy as np
from numpy import *
from numpy import short
from numpy import maximum
from numpy import dtype
from numpy import polyfit, poly1d
from numpy.random import seed, standard_normal
import scipy as scy
from scipy import stats
import matplotlib.pyplot as plot

#Group 2


#To accelerate the program, decrease the parameter N in the function. 

#%%
#Part 1: Black Scholes pricing

r = 0.01
sigma = 0.18
T = 2
S0 = 100
K1 = 80
K2 = 100
K3 = 120
delta = (1/252)

#Computation of the moments

E_x = S0*((np.exp(r*T)-1)/r)

E_x2 = (2*(S0**(2)))*((r*exp(((sigma**2)+(2*r))*T)-((sigma**2) + 2*r)*( exp(r*T))+((sigma**2) +r))/(((sigma**2)+r)*((sigma**2)+2*r)*r))

print(" Two first moments of the lognormal distribution: ")
print(E_x)
print(E_x2)

#Calculation of the parameters of the lognormal with the moments

var = (1/T)*np.log((E_x2/(E_x**2)))
mean = ((1/T)*np.log(E_x))-(var/2)

print(" parameters of the distribution: ")
print(" mean: ")
print(mean)
print(" variance: ")
print(var)

print(" We can now calculate the Black Scholes price. ")
print("-------------------")

d1K1 = (1/(np.sqrt(T)*np.sqrt(var)))*((np.log(S0/K1) + (r + (var/2))*T))
d1K2 = (1/(np.sqrt(T)*np.sqrt(var)))*((np.log(S0/K2) + (r + (var/2))*T))
d1K3 = (1/(np.sqrt(T)*np.sqrt(var)))*((np.log(S0/K3) + (r + (var/2))*T))

d2K1 = (1/(np.sqrt(T)*np.sqrt(var)))*((np.log(S0/K1) + (r - (var/2))*T))
d2K2 = (1/(np.sqrt(T)*np.sqrt(var)))*((np.log(S0/K2) + (r - (var/2))*T))
d2K3 = (1/(np.sqrt(T)*np.sqrt(var)))*((np.log(S0/K3) + (r - (var/2))*T))


print(" We first calculate d1 and d2 of Black Scholes formula: ")
print("d1 K1, d1 K2, d1 K3 : ")
print(d1K1)
print(d1K2)
print(d1K3)
print("")
print("d2 K1, d2 K2, d2 K3 : ")
print(d2K1)
print(d2K2)
print(d2K3)

BSK1 = S0*scy.stats.norm.cdf(d1K1) - K1*np.exp(-r*T)*scy.stats.norm.cdf(d2K1)
BSK2 = S0*scy.stats.norm.cdf(d1K2) - K2*np.exp(-r*T)*scy.stats.norm.cdf(d2K2)
BSK3 = S0*scy.stats.norm.cdf(d1K3) - K3*np.exp(-r*T)*scy.stats.norm.cdf(d2K3)

print("-------------------")
print("We plug these results in the black scholes formula and we obtain the prices for the different strike prices:")
print("Price K1")
print(BSK1)
print("")
print("Price K2")
print(BSK2)
print("")
print("Price K3")
print(BSK3)

Price = 3*BSK1 - 6*BSK2 + 3*BSK3

print("-------------------")
print("We compute now the price related to our position:")
print("3K1 - 6K2 + 3 K3 = ")
print(Price)
print("-------------------")


#%% ---------------------------------------------------------------------------------#

#Part 2: Monte Carlo simulation
def MCSim(S_0, sigma, r, T, Delta, N, GRAPH = False):
    TS = int(T/Delta)
    
    MonteCarlo = np.zeros([TS+1,N]) #TS+1 because we include S_0
    At = np.zeros(N)

    for i in range(0,N): #We run N simulations with TS time steps
        if i%100000 == 0:
            print(str((i/N)*100)+ "%")
        MonteCarlo[0][i] = S_0
        for j in range(1,TS+1):
            MonteCarlo[j][i] = MonteCarlo[j-1][i] * exp((r-(sigma**2)/2)*Delta +sigma * sqrt(Delta)*standard_normal(1))
        At[i] = mean(MonteCarlo[1:,i])

    K = np.array([80,100,120])
    Pos = np.array([3,-6,3])
    priceMC = 0
    for a in range(len(K)): #Computation of the price 
        priceMC += Pos[a] * exp(-r*T) * sum(maximum(At-K[a], 0))/N
        if GRAPH == True:
            print("Payoff for K = " + str(K[a]))
            print(exp(-r*T) * sum(maximum(At-K[a], 0))/N)
    
    if GRAPH == True: 
        plot.plot(MonteCarlo)
        plot.xlabel("Time step")
        plot.ylabel("Price")
        plot.title("First 500 iterations")
        plot.grid(True)
        plot.show()
    
    callprice = exp(-r*T) * sum(maximum(At-100, 0))/N
        
    return priceMC, At, callprice

#%% Question 2.1. Monte Carlo simulation

#Running the function with normal parameters
print("Price: " + str(MCSim(100,0.18, 0.01, 2, 1/252, 500000, True)[0]))
#price with 500 000 simulations: 27.87614315307863


#%% Question 2.2. 
upperbound = 15001
Sim = np.arange(1, upperbound, 200)
SE_sim = np.zeros(len(Sim))
At = MCSim(100,0.18,0.01,2,1/252,upperbound)[1]
K = np.array([80,100,120])
Pos = np.array([3,-6,3])
r, T = 0.01, 2

for i in range(len(Sim)):
    SimArray = np.zeros(Sim[i]+1)
    for j in range(len(SimArray)):
        Price = 0
        for a in range(len(K)):
            Price += Pos[a] * exp(-r*T) * maximum(At[j]-K[a], 0)
        SimArray[j] = Price

    SE_sim[i] = SimArray.std()

for i in range(len(Sim)):
    SE_sim[i] = SE_sim[i]/sqrt(Sim[i])
print("Evolution of SE when n is increasing:")
print(SE_sim)

plot.plot(Sim, SE_sim)
plot.title("Evolution of the volatility with respect to n")
plot.xlabel("Number of simulations")
plot.ylabel("Standard error")
plot.grid(True)
plot.show()

#----------------------------------------------------------------------------------#

# %% Question 2.3. Variation on parameters

#A. Sensitivity to volatility on 1 call
print("Sensitivity to volatility on 1 call")
sigma_Array = np.arange(0.06, 0.31, 0.02, dtype = float)
Call_Array = np.zeros(len(sigma_Array))
for i in range(len(sigma_Array)):
    Call_Array[i] = MCSim(100, sigma_Array[i], 0.01, 2, 1/252, 50000)[2]

plot.plot(sigma_Array, Call_Array)
plot.title('Sensitivity to 'r'$\sigma$' " on the call price")
plot.xlabel('Volatility 'r'$\sigma$')
plot.ylabel("Call price")
plot.grid(True)
plot.show()

#%% 

#B. Sensitivity to volatility on butterfly price
print("Sensitivity to volatility on butterfly price")
sigma_Array = np.arange(0.06, 0.31, 0.02, dtype = float)
Price_Array = np.zeros(len(sigma_Array))
for i in range(len(sigma_Array)):
    Price_Array[i] = MCSim(100, sigma_Array[i], 0.01, 2, 1/252, 50000)[0]

plot.plot(sigma_Array, Price_Array)
plot.title('Sensitivity to 'r'$\sigma$' " on the butterfly price")
plot.xlabel('Volatility 'r'$\sigma$')
plot.ylabel("Butterfly price")
plot.grid(True)
plot.show()

#%%

#C. Sensitivity to r on 1 call
print("Sensitivity to r on 1 call")
r_Array = np.arange(0,0.061, 0.003, dtype = float)
Call_Array = np.zeros(len(r_Array))
for i in range(len(r_Array)):
    Call_Array[i] = MCSim(100,0.18, r_Array[i], 2, 1/252, 50000)[2]
    print(i)

plot.scatter(r_Array, Call_Array)
plot.title("Trend for the sensitivity to r on the call price")
plot.xlabel("Interest rate r")
plot.ylabel("Call price")
z = polyfit(r_Array, Call_Array, 1)
p = poly1d(z)
plot.plot(r_Array, p(r_Array), "r--")
plot.grid(True)
plot.show()

plot.plot(r_Array, Call_Array)
plot.title("Sensitivity to r on the Call price")
plot.xlabel("Interest rate r")
plot.ylabel("Call price")
plot.grid(True)
plot.show()



# %%

#D. Sensitivity to r on butterfly price
print("Sensitivity to r on butterfly price")
r_Array = np.arange(0,0.061, 0.002, dtype = float)
Price_Array = np.zeros(len(r_Array))
for i in range(len(r_Array)):
    Price_Array[i] = MCSim(100,0.18, r_Array[i], 2, 1/252, 50000)[0]
    print(i)

plot.scatter(r_Array, Price_Array)
plot.title("Trend for the sensitivity to r on the butterfly price")
plot.xlabel("Interest rate r")
plot.ylabel("Butterfly price")
z = polyfit(r_Array, Price_Array, 1)
p = poly1d(z)
plot.plot(r_Array, p(r_Array), "r--")
plot.grid(True)
plot.show()

plot.plot(r_Array, Price_Array)
plot.title("Sensitivity to r on the butterfly price")
plot.xlabel("Interest rate r")
plot.ylabel("Butterfly price")
plot.grid(True)
plot.show()

# %%

#E. Sensitivity to period T on 1 call
T_Array = np.arange(1, 11, 1)
Call_Array = np.zeros(len(T_Array))
for i in range(len(T_Array)):
    Call_Array[i] = MCSim(100, 0.18, 0.01, T_Array[i], 1/252, 50000)[2]

plot.plot(T_Array, Call_Array)
plot.title("Sensitivity to T on the call price")
plot.xlabel("Period")
plot.ylabel("Call price")
plot.grid(True)
plot.show()

# %%

#F. Sensitivity to period T on butterfly price
T_Array = np.arange(1, 16, 1)
Price_Array = np.zeros(len(T_Array))
for i in range(len(T_Array)):
    Price_Array[i] = MCSim(100, 0.18, 0.01, T_Array[i], 1/252, 50000)[0]

plot.plot(T_Array, Price_Array)
plot.title("Sensitivity to T on the butterfly price")
plot.xlabel("Period")
plot.ylabel("Butterfly price")
plot.grid(True)
plot.show()


# %%

#G. Sensitivity to Strike K on 1 call
K_Array = np.arange(70, 131, 5)
Call_Array = np.zeros(len(K_Array))
Num = 50000
r = 0.01
T=2
Sim = MCSim(100, 0.18, 0.01, 2, 1/252, Num)[1]
for i in range(len(K_Array)):
    Call_Array[i] = exp(-r*T) * sum(maximum(Sim-K_Array[i], 0))/Num

plot.plot(K_Array, Call_Array)
plot.title("Sensitivity to K on the call price")
plot.xlabel("Strike")
plot.ylabel("Call price")
plot.grid(True)
plot.show()

# %%

#H. Sensitivity to K1/K2/K3 on butterfly price
Num, r, T = 50000, 0.01, 2

K1_Array = np.arange(20,141,5)
K2_Array = np.arange(40,161,5) #We shift the payoff curve from left to right
K3_Array = np.arange(60,181,5)
Price_Array = np.zeros(len(K1_Array))
Sim = MCSim(100, 0.18, 0.01, 2, 1/252, Num)[1]

for i in range(len(K1_Array)):
    K = np.array([K1_Array[i],K2_Array[i],K3_Array[i]])
    Pos = np.array([3,-6,3])

    priceMC = 0
    for a in range(len(K)): #Computation of the price 
        priceMC += Pos[a] * exp(-r*T) * sum(maximum(Sim-K[a], 0))/Num    
    Price_Array[i] = priceMC

plot.plot(K1_Array, Price_Array)
plot.title("Sensitivity to K1 on the butterfly price")
plot.xlabel("Value of K1")
plot.ylabel("Butterfly price")
plot.grid(True)
plot.show()

plot.plot(K2_Array, Price_Array)
plot.title("Sensitivity to K2 on the butterfly price")
plot.xlabel("Value of K2")
plot.ylabel("Butterfly price")
plot.grid(True)
plot.show()

plot.plot(K3_Array, Price_Array)
plot.title("Sensitivity to K3 on the butterfly price")
plot.xlabel("Value of K3")
plot.ylabel("Butterfly price")
plot.grid(True)
plot.show()

# %% --------------------------------------------------------------------------#
#Part 3. Heston model

#Function for Heston model
def HestonSim(S_0, r, T, Delta, N, rho, theta, kappa, V_0, xi):
    TS = int(T/Delta)
    Heston = np.zeros([TS+1,N]) #TS+1 because we include S_0
    At = np.zeros(N)

    for i in range(0,N):
        Heston[0][i] = S_0
        dz_v = np.zeros(TS+1)
        dz_v = np.random.normal(0, 1, TS+1)
        W_v = np.zeros(TS+1)
        for j in range(1,TS+1):
            W_v[j] = W_v[j-1] + dz_v[j] #First Brownian motion

        dz_S = np.zeros(TS+1)
        dz_S = np.random.normal(0, 1, TS+1)
        z_S = np.cumsum(dz_S)
        W_S = np.zeros(TS+1)
        for k in range(1,TS+1):
            W_S[k] = rho*W_v[k] + np.sqrt(1-pow(rho,2))*z_S[k] #Second Brownian motion


        vt = np.zeros(TS+1)
        vt[0] = V_0
        Heston[0][i] = S_0
        for l in range(1,TS+1):
            #Calculating the variance and the stock price evolution
            vt[l] = np.abs(vt[l-1] + kappa * (theta - vt[l-1]) * Delta + xi * np.sqrt(vt[l-1]) * (W_v[l] - W_v[l-1]) * sqrt(Delta))
            Heston[l][i] = Heston[l-1][i] + r * Heston[l-1][i] * Delta + np.sqrt(vt[l-1]) * Heston[l-1][i] * (W_S[l] - W_S[l-1]) * sqrt(Delta)
        At[i] = mean(Heston[1:,i])

    K = np.array([80,100,120])
    Pos = np.array([3,-6,3])
    priceMC = 0
    #Calculating the butterfly price
    for a in range(len(K)):
        priceMC += Pos[a] * exp(-r*T) * sum(maximum(At-K[a], 0))/N

    plot.plot(Heston[:,:500])
    plot.xlabel("Time step")
    plot.ylabel("Price")
    plot.title("First 500 iterations for Heston model")
    plot.show()
        
    return priceMC, At

#%%
S_0 = 100       # Initial stock price
V_0 = 0.0441      # Initial variance is square of volatility
kappa = 0.3       # kappa mean reversion speed
theta = 0.0441    # Long-run variance
rho = -0.75 
N = 500000
T = 2
Delta = 1/252
r = 0.01
xi=0.15

#We print the price estimate by the Heston model using the previous parameters: 
print("Heston model price:")
print(HestonSim(S_0, r, T, Delta, N, rho, theta, kappa, V_0, xi)[0])

