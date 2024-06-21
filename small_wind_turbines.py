# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:06:50 2024

@author: gusta
"""

#%% ############## Initialize ##############

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
#import statistics as stat
import math
#from windrose import WindroseAxes
import scipy.stats as stats
from scipy.stats import weibull_min
#from scipy.integrate import quad
from scipy.stats import norm

# Loading af datasættet
file = os.path.join(os.getcwd(), 'data_wind.csv')
df_oesterild = pd.read_csv(open(file))

"""
# Plot af datasættet
plt.figure()
plt.plot(df_oesterild["wsp_106m_LMN_mean"])
plt.title("Alle middel vindhastigheder (uden fordeling)")
plt.xlabel("Datapunkt")
plt.ylabel("Målte vindhastigheder [m/s]")
plt.grid()
plt.show()
"""

#%% ############## Vindklimaet ##############
###### W11 


# Sæt dataen som array, så det kan indekseres:
wind_speeds = np.array(df_oesterild["wsp_106m_LMN_mean"])

### Udregningsmetoden
# Tjek middelværdien
middel = wind_speeds.mean()
print("Middel af datasæt = ", round(middel,3), "[m/s]")

# Tjek standardfordelingen
std = np.std(wind_speeds)
print("Std af datasæt = ", round(std,3), "[m/s]")

# Find k og A ud fra formelen
udregnet_k = (std/middel)**(-1.086)
print("Udregnet k = ", round(udregnet_k,3))

udregnet_A = ((middel*udregnet_k**(2.6674)))/(0.184+0.816*udregnet_k**(2.73855))
print("Udregnet A = ", round(udregnet_A,3), "[m/s]")

# Plot Weibull distributionen med hvor ofte vindhastigheden opstår i hele datasættet
v = np.arange(0,30,1)
"""
plt.figure()
plt.hist(df_oesterild["wsp_106m_LMN_mean"], bins = math.ceil(max(df_oesterild["wsp_106m_LMN_mean"])), density=False, label = "Measured wind speed")
plt.plot(24*350*6 * (udregnet_k/udregnet_A) * ((v/udregnet_A)**(udregnet_k-1)) * np.exp(-((v/udregnet_A)**udregnet_k)), label = "Fitted Weibull distribution")
plt.ylabel("10 mins. pr. year [h]")
plt.xlabel("Wind speed [m/s]")
plt.title("Wind speeds in the year 2023")
plt.legend()
plt.grid()
plt.show()
"""

# Plot Weibull distributionen med hvor ofte vindhastigheden opstår procentvis
plt.figure()
plt.hist(df_oesterild["wsp_106m_LMN_mean"], bins = math.ceil(max(df_oesterild["wsp_106m_LMN_mean"])), density=True, label = "Measured wind speed")
plt.plot((udregnet_k/udregnet_A) * ((v/udregnet_A)**(udregnet_k-1)) * np.exp(-((v/udregnet_A)**udregnet_k)), lw=2, label = "PDF")
plt.ylabel("Probability of each wind speed")
plt.xlabel("Wind speed [m/s]")
plt.title("Udregnet Weibull fordeling")
plt.legend()
plt.grid()
plt.show()

weib_mean_udregnet = weibull_min.mean(udregnet_k, loc=0, scale=udregnet_A)
print("weib_mean_udregnet = ", round(weib_mean_udregnet,3), "[m/s]")
print("")

"""
# Find wind power density
n = 50024-3
WPD = (1/2*n)*1.204*(np.sum(df_oesterild["wsp_106m_LMN_mean"]**3))
WPD = (1/2*n)*1.204*middel**3
print ("WPD = ", round(WPD,2), "[W/m^2]")
"""

### Fitting metoden (med function fra scipy.stats)
# Fit Weibull distribution til dataen
params = stats.weibull_min.fit(wind_speeds, loc=0)

# Extract estimated parameters
A_fit, k_fit = params[2], params[0]

print("Estimeret k = ", round(k_fit,3))
print("Estimeret A = ", round(A_fit,3), "[m/s]")


# Plot histogram of generated samples
plt.figure()
plt.grid()
plt.hist(wind_speeds, bins=math.ceil(max(wind_speeds)), density=True, alpha=0.6, color='g', label = "Measured wind speed")

# Plot probability density function (PDF)
x = np.linspace(0, math.ceil(max(wind_speeds)), 100)
plt.plot(x, stats.weibull_min.pdf(x, k_fit, scale=A_fit), 'r-', lw=2, label='PDF')

plt.ylabel("Probability of each wind speed")
plt.xlabel("Wind speed [m/s]")
plt.title("Fittet Weibull fordeling")
plt.legend()

# Find middelhastigheden
weib_mean_fit = weibull_min.mean(k_fit,loc=0,scale=A_fit)
print("weib_mean_fit = ", round(weib_mean_fit,3), "[m/s]")
print("")


### Korrigeret fitting metoden 
# Fit Weibull distribution til dataen
params = stats.weibull_min.fit(wind_speeds, loc=0)

# Extract estimated parameters
A_kor, k_kor = 8.6, 2.2

print("Korrigeret k = ", round(k_kor,3))
print("Korrigeret A = ", round(A_kor,3), "[m/s]")


# Plot histogram of generated samples
plt.figure()
plt.grid()
plt.hist(wind_speeds, bins=math.ceil(max(wind_speeds)), density=True, alpha=0.6, color='y', label = "Measured wind speed")

# Plot probability density function (PDF)
x = np.linspace(0, math.ceil(max(wind_speeds)), 100)
plt.plot(x, stats.weibull_min.pdf(x, k_kor, scale=A_kor), 'g-', lw=2, label='PDF')

plt.ylabel("Probability of each wind speed")
plt.xlabel("Wind speed [m/s]")
plt.title("Fittet og korrigeret Weibull fordeling")
plt.legend()


weib_mean_kor = weibull_min.mean(k_kor,loc=0,scale=A_kor)
print("weib_mean_korrigeret = ", round(weib_mean_kor,3), "[m/s]")
print("")
print("")




#%% ############## Flow modellering ##############
###### W12

### Udregningsmetoden
# Udregn A med SpeedUpFactor
SUF=0.9
udregnet_A_med_SUF = ((middel*udregnet_k**(2.6674)))/(0.184+0.816*udregnet_k**(2.73855))*SUF
print("Udregnet A med SUF = ", round(udregnet_A_med_SUF,3), "[m/s]")

# Plot Weibull distributionen med hvor ofte vindhastigheden opstår procentvis
plt.figure()
plt.hist(wind_speeds, bins = math.ceil(max(wind_speeds)), density=True, label = "Measured wind speed")
plt.plot((udregnet_k/udregnet_A_med_SUF) * ((v/udregnet_A_med_SUF)**(udregnet_k-1)) * np.exp(-((v/udregnet_A_med_SUF)**udregnet_k)), lw=2, label = "PDF")
plt.ylabel("Probability of each wind speed")
plt.xlabel("Wind speed [m/s]")
plt.title("Udregnet Weibull fordeling med SUF")
plt.legend()
plt.grid()
plt.show()

# Find middelværdien på vindhastigheden med SUF
weib_mean_udregnet_med_SUF = weibull_min.mean(udregnet_k,loc=0,scale=udregnet_A_med_SUF)
print("weib_mean_udregnet_med_SUF = ", round(weib_mean_udregnet_med_SUF,3), "[m/s]")
print("")


### Fitting metoden (med function fra stats.)
# Fit Weibull distribution til dataen
params = stats.weibull_min.fit(wind_speeds, loc=0)

# Extract estimated parameters
A_fit_med_SUF, k_fit_med_SUF = params[2]*SUF, params[0]

print("Estimeret k = ", round(k_fit_med_SUF,3))
print("Estimeret A med SUF = ", round(A_fit_med_SUF,3), "[m/s]")

# Plot histogram of generated samples
plt.figure()
plt.grid()
plt.hist(wind_speeds, bins=math.ceil(max(wind_speeds)), density=True, alpha=0.6, color='g', label = "Measured wind speed")

# Plot probability density function (PDF)
x = np.linspace(0, math.ceil(max(wind_speeds)), 100)
plt.plot(x, stats.weibull_min.pdf(x, k_fit_med_SUF, scale=A_fit_med_SUF), 'r-', lw=2, label='PDF')
plt.ylabel("Probability of each wind speed")
plt.xlabel("Wind speed [m/s]")
plt.title("Fittet Weibull fordeling med SUF")
plt.legend()

# Find middelhastigheden
weib_mean_fit_med_SUF = weibull_min.mean(k_fit_med_SUF,loc=0,scale=A_fit_med_SUF)
print("weib_mean_fit_med_SUF = ", round(weib_mean_fit_med_SUF,3), "[m/s]")
print("")


### Korrigeret fitting metoden
# Fit Weibull distribution til dataen
params = stats.weibull_min.fit(wind_speeds, loc=0)

# Extract estimated parameters
A_kor_med_SUF, k_kor_med_SUF = 8.6*SUF, 2.2

print("Korrigeret k = ", round(k_kor_med_SUF,3))
print("Korrigeret A = ", round(A_kor_med_SUF,3), "[m/s]")

# Plot histogram of generated samples
plt.figure()
plt.grid()
plt.hist(wind_speeds, bins=math.ceil(max(wind_speeds)), density=True, alpha=0.6, color='y', label = "Measured wind speed")

# Plot probability density function (PDF)
x = np.linspace(0, math.ceil(max(wind_speeds)), 100)
plt.plot(x, stats.weibull_min.pdf(x, k_kor_med_SUF, scale=A_kor_med_SUF), 'g-', lw=2, label='PDF')
plt.ylabel("Probability of each wind speed")
plt.xlabel("Wind speed [m/s]")
plt.title("Fittet og korrigeret Weibull fordeling med SUF")
plt.legend()

# Find middelhastigheden
weib_mean_kor_med_SUF = weibull_min.mean(k_kor_med_SUF,loc=0,scale=A_kor_med_SUF)
print("weib_mean = ", round(weib_mean_kor_med_SUF,3), "[m/s]")
print()
print()



### Arbejde på usandsynlighed
ran_A = np.random.uniform(low=A_fit_med_SUF-2, high=A_fit_med_SUF+2, size = 10000)
#print(ran_A)

std_ran_A = np.std(ran_A)
print("std_ran_A = ", std_ran_A, "[m/s]")


array = []
#count = 0

for val in ran_A:
    array.append(weibull_min.mean(A_fit_med_SUF, loc=0, scale=val))
    #array[count] = weibull_min.mean(k_fit_med_SUF,loc=0,scale=ran_A[count])
    #count +=1

std_array = np.std(array)
print("std_array", std_array, "[m/s]")

print()
print()

# Påskeferie i W13
#%% ############## Estimering af energiproduktion og tab ##############
###### W14


### Opg. 1-3: Arbejde på model chain, find nye A og k værdier, og find Power Curve (PC)

# Vi vælger to SUF:
# De betyder virkelig meget for CF
S_top = 0.9
S_bui = 0.8

# Finder de nye A og k (k ændrer sig ikke da den ikke er afhængig af SUF)
A1, k1 = udregnet_A * S_top, udregnet_k
A2, k2 = A1 * S_bui, k1

# Konstanter 
"""
Note: der blev i starten taget en 1MW vindmølle fra ELKRAFT som udgangspunkt, nu arbejdes
der med en større vindmølle på 10MW. Den bliver nok endnu mindre til bachelorprojektet)
"""
D = 198  # Diameter af turbinen (m)
C_p = 0.45  # Power coefficient (standarden er 0.4-0.5 for vindmøller)
rho = 1.225  # Luftdensitet (kg/m^3)
P_gen = 10 * 10**6  # Generator power (W) (10MW)
cut_in_speed = 4  # Cut-in wind speed (m/s)
cut_out_speed = 25 # Cut-out wind speed (m/s)

# Indfører vilkårlige loss konstanter
L_env = 0.9
L_ava = 0.9


# Wind speed range
v_pc = np.linspace(0, 30, 30)  # Wind speed i m/s

# Calculate power output for each wind speed
power_output = np.zeros_like(v_pc)

for i, v in enumerate(v_pc):
    if v < cut_in_speed:
        power_output[i] = 0
    elif v >= cut_in_speed and v < cut_out_speed:
        power_output[i] = min(1/2 * rho * v**3 * C_p * np.pi * (D/2) ** 2, P_gen)
    else:
        power_output[i] = 0

# Plotting the power curve
plt.figure()
plt.plot(v_pc, power_output, label='Power Curve')
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Power Output (W)')
plt.title('Wind Turbine Power Curve')
plt.grid()
plt.legend()
plt.show()


### Opg. 4-5: GAEP og NAEP

# Wind speed range
"""
Grundet til at vindhastighedsintervallet er defineret igen er fordi der blev leget med 
ideen om at have flere punkter i intervallet for at gøre kurven mere glat, men det drillede
"""
v_wp = np.linspace(0, 30, 30) # Wind speed i m/s

# Plot the Weibull distribution
plt.figure()
plt.plot((k2/A2)*((v_wp/A2)**(k2-1)) * np.exp(-((v_wp/A2)**k2)), lw=2, label = "PDF")
plt.ylabel("Probability of each wind speed")
plt.xlabel("Wind speed [m/s]")
plt.title("Weibull fordeling til PC")
plt.legend()
plt.grid()
plt.show()


# Calculate GAEP
GAEP = power_output*((k2/A2)*((v_wp/A2)**(k2-1)) * np.exp(-((v_wp/A2)**k2)))

#print(GAEP)
print("GAEP = ", sum(GAEP)*8766/1000000, "MWh")


# Ganger de to losses sammen så de bliver til én enkel faktor total_loss
total_loss = L_env*L_ava

# Finder NAEP 
NAEP = GAEP * total_loss

#print(NAEP)
print("NAEP = ", sum(NAEP)*8766/1000000, "MWh")



#%% ############## Turbiner på bygninger og usikkerhed & Kost og indtægter ##############
###### W15 & W16 

# Definér alpha som -0.06 (ud fra grafen)
# Find middel vindhastigheden fra datasættet
# Definér V_tilde
# Definér AEV ud fra AEP*V_tilde

### Start på arbejde hvor vi finder usandsynlighederne på variablerne. 
# Vi starter med at definere middelværdier og standarafvigelserne for de 8 inputs

# Husk at sætte S det rigtige sted
mu_A2, sigma_A2 = udregnet_A, 0.5
mu_k2, sigma_k2 = udregnet_k, 0.01
mu_S_top, sigma_S_top = S_top, 0.01
mu_S_bui, sigma_S_bui = S_bui, 0.01
mu_C_p, sigma_C_p = C_p, 0.01
mu_L_env, sigma_L_env = L_env, 0.05
mu_L_ava, sigma_L_ava = L_ava, 0.05

#Gennemsnitlig elpris = 2.7 kr/kWh
mu_avgPrice, sigma_avgPrice = 2.7, 0.5
# (Nok lidt lavere, men det er det vi bruger lige nu)


# Vi genererer 10000 tilfældige værdier for de 8 inputs
n = 10000
A_ran = np.random.normal(mu_A2, sigma_A2, n)
k_ran = np.random.normal(mu_k2, sigma_k2, n)
S_top_ran = np.random.normal(mu_S_top, sigma_S_top, n) 
S_bui_ran = np.random.normal(mu_S_bui, sigma_S_bui, n) 
C_p_ran = np.random.normal(mu_C_p, sigma_C_p, n) 
L_env_ran = np.random.normal(mu_L_env, sigma_L_env, n) 
L_ava_ran = np.random.normal(mu_L_ava, sigma_L_ava, n) 
avgPrice_ran = np.random.normal(mu_avgPrice, sigma_avgPrice, n)

# A skal tage hensyn til SUF:
A_ran = A_ran * S_top_ran * S_bui_ran


v_pc = np.linspace(0, 30, 30)  # Wind speed i m/s

# Calculate power output for each wind speed
power_output = np.zeros_like(v_pc)


# Sort V_ran
#V_ran = np.sort(V_ran)

# Apply constraints and define power curve
GAEP_values = []
GAEV_values = []
#value_output = np.zeros_like(v_pc)
value_factor_output = np.zeros_like(v_pc)
#V_tilde = alpha*(U-U_mean)+1
#V_tilde = float()
alpha = float(-0.06)

for y in range(10000):
    for i, v in enumerate(v_pc):
        if v < cut_in_speed:
            power_output[i] = 0
        elif v >= cut_in_speed and v < cut_out_speed:
            power_output[i] = min(1/2 * rho * v**3 * C_p_ran[y] * np.pi * (D/2) ** 2, P_gen)
            # Tilføj prisfunktion her
            # power_output[i] * V_tilde
            # Tag højde for om man bare kan bruge A som gennemsnit
            # Weibull funktionen har sikkert en måde at finde middelhastigheden på
            V_tilde = alpha * (v - A_ran[y]) + 1
            value_factor_output[i] = power_output[i] * V_tilde
        else:
            power_output[i] = 0
    weibull = np.zeros_like(v_pc)
    weibull = (k_ran[y]/A_ran[y])*((v_pc/A_ran[y])**(k_ran[y]-1)) * np.exp(-((v_pc/A_ran[y])**k_ran[y]))
    
    GAEP = power_output * weibull
    total_GAEP = sum(GAEP)*8766/1000000
    GAEP_values = np.append(GAEP_values, total_GAEP)
    
    # Man kan sammenligne GAEP og GAEV for at finde V_tilde
    #GAEV = value_factor_output * weibull * avgPrice_ran[y] / 1000
    GAEV = value_factor_output * weibull 
    total_GAEV = sum(GAEV)*8766/1000000
    GAEV_values = np.append(GAEV_values, total_GAEV)

print("STD GAEP_values = ", np.std(GAEP_values), "MWh")
print("mean GAEP_values = ", np.mean(GAEP_values), "MWh")
print()
print("STD GAEV_values = ", np.std(GAEV_values), "tusind DKK")
print("mean GAEV_values = ", np.mean(GAEV_values), "tusind DKK")

#print(sum(weibull))


"""
Vi startede med at kigge på VF og CF, men nåede ikke længere
"""

# Value Factor
VF = total_GAEV/total_GAEP 

# Capacity Factor
CF = total_GAEP/(8766 * P_gen/1000000)

print()
print("VF = ", VF)
print("CF = ", CF)
print()


# Sammenlign GAEP og GAEV
plt.figure()
plt.scatter(GAEP_values, GAEV_values, s = 1, label = "GAEP vs GAEV")
plt.xlabel("GAEP [MWh]")
plt.ylabel("GAEV [tusind DKK]")
plt.grid()
plt.legend()
plt.show()


# Plot price curve funktionen
plt.figure()
plt.plot(v_pc, alpha*(v_pc-8)+1, label = "Alpha = -0.06")
plt.xlabel("Vindhastighed [m/s]")
plt.ylabel("Price Curve [-]")
plt.grid()
plt.legend()
plt.show()


# Definér middelværdi og standardfordeling for GAEP
mu, sigma = np.mean(GAEP_values), np.std(GAEP_values)


# Find P90 med Python funktionen
P90_py = norm.ppf(0.1,mu,sigma)
#print("P90_py = ", P90_py)

# Find P90 matematisk
z = 1.282  # konstant værdi for P90
P90_math = mu - z * sigma
#print("P90_math = ", P90_math)

# Find højden af PDF (bruger Python funktionen og ikke den matematiske værdi. De er næsten ens)
y_at_P90 = norm.pdf(P90_py, mu, sigma)

# Plot PDF og P90
x_norm = np.linspace(10000, 32000, 100)
pdf = norm.pdf(x_norm, mu, sigma)
plt.figure()
plt.plot(x_norm,pdf,label="PDF")
plt.axvline(P90_py, ymin = 0, ymax = y_at_P90 / pdf.max(), color = 'red', linestyle = '--', label = 'P90')
plt.ylabel("Probability for each unit of GAEP")
plt.xlabel("GAEP with uncertainty [MWh]")
plt.title("PDF of GAEP and P90 over a year")
plt.legend()
plt.grid()
plt.show()



# Definér middelværdi og standardfordeling for GAEV
mu, sigma = np.mean(GAEV_values), np.std(GAEV_values)


# Find P90 med Python funktionen
P90_py = norm.ppf(0.1, mu, sigma)
#print("P90_py = ", P90_py)

# Find P90 matematisk
z = 1.282  # konstant værdi for P90
P90_math = mu - z * sigma
#print("P90_math = ", P90_math)

# Find højden af PDF (bruger Python funktionen og ikke den matematiske værdi. De er næsten ens)
y_at_P90 = norm.pdf(P90_py, mu, sigma)

# Plot PDF og P90
x_norm = np.linspace(9000, 27000,100)
pdf = norm.pdf(x_norm, mu, sigma)
plt.figure()
plt.plot(x_norm, pdf, label="PDF")
plt.axvline(P90_py, ymin = 0, ymax = y_at_P90 / pdf.max(), color = 'red', linestyle = '--', label = 'P90')
plt.ylabel("Probability for each unit of Value")
plt.xlabel("Value with uncertainty [tusind DKK]")
plt.title("PDF of GAEV and P90 over a year")
plt.legend()
plt.grid()
plt.show()


 


