#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

from sklearn.linear_model import LinearRegression
from sklearn import svm

from matplotlib import rc

plt.rcParams.update({'font.size': 24})

rc('text', usetex=True)

import os
import datetime

import warnings
from scipy import interpolate

#import pingouin as pg
from scipy.stats import ks_2samp, kruskal

warnings.filterwarnings('ignore')


# In[ ]:


def compute_GDD_w_opt_max(T, T_base, T_opt, T_max):
    
    GDD = np.zeros(len(T))
    
    m = - (T_opt - T_base) / (T_max - T_opt)
    n = -m * T_max
    
    for i in range(len(T)):
        
        if T[i] < T_base:
            
            pass
        
        elif T[i] <= T_opt:
            
            GDD[i] += T[i] - T_base
            
        elif T[i] <= T_max:
            
            GDD[i] += m*T[i] + n
            
    return GDD

def compute_GDD_opt_max(filename, T_base, T_opt, T_max, T_base_CDD=7):
    
    df = pd.read_csv(filename)
    
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    
    df["GDD"] = compute_GDD_w_opt_max(df["Temperature"].values, T_base, T_opt, T_max)
    df["CDD"] = (T_base_CDD - df["Temperature"]).apply(lambda x: x if x > 0 else 0)
    
    GDD = df.resample('D', on="Date").mean()
    
    return GDD

def compute_GDD_range(df, start, end):
    
    return df[(df.index >= start) & (df.index <= end)]["GDD"].sum()

def compute_CDD_range(df, start, end):
    
    return df[(df.index >= start) & (df.index <= end)]["CDD"].sum()

def compute_avg_humidity_range(df, start, end):
    
    return df[(df.index >= start) & (df.index <= end)]["Humidity"].mean()


def compute_avg_temperature_range(df, start, end):
    
    return df[(df.index >= start) & (df.index <= end)]["Temperature"].mean()

def egg_hatching_GDD_opt_max(filename, T_base, T_opt, T_max, T_base_CDD, loc, filename_results="Resultados GD.xlsx"):
    
    df = compute_GDD_opt_max(filename, T_base, T_opt, T_max, T_base_CDD)
    
    eggs = pd.read_excel(filename_results)

    eggs.sort_values(by="Fecha puesta")
    
    eggs_loc = eggs[eggs["Localización"] == loc]
    
    GDD_eclosion = []
    CDD_eclosion = []
    avg_humidity = []
    avg_temperature = []

    for i in range(len(eggs_loc)):

        start = eggs_loc.iloc[i]["Fecha puesta"]
        end = eggs_loc.iloc[i]["Fecha emergencia"]

        if not type(start) == str:

            GDD_eclosion.append(compute_GDD_range(df, start, end))
            CDD_eclosion.append(compute_CDD_range(df, start, end))
            
            avg_humidity.append(compute_avg_humidity_range(df, start, end))
            avg_temperature.append(compute_avg_temperature_range(df, start, end))
            
    eggs_loc["GDD_eclosion"] = GDD_eclosion
    eggs_loc["CDD_eclosion"] = CDD_eclosion
    eggs_loc["Humidity_eclosion"] = avg_humidity
    eggs_loc["Temperature_eclosion"] = avg_temperature
    
    return eggs_loc

def construct_df_opt_max(filenames, locs, T_base=7, T_opt=25, T_max=35, T_base_CDD=7, filename_results="Resultados GD.xlsx"):
    
    for filename, loc in zip(filenames, locs):
            
        if filename == filenames[0]:
            
            df = egg_hatching_GDD_opt_max("Processed_climatic_data/%s" % filename, T_base, T_opt,
                                      T_max, T_base_CDD, loc, filename_results)
        else:

            df_new = egg_hatching_GDD_opt_max("Processed_climatic_data/%s" % filename, T_base, T_opt,
                                          T_max, T_base_CDD, loc, filename_results)

            df = df.append(df_new, ignore_index=True)
            
    return df

def sigmoid(x, x0, k):
    
    y = 1 / (1 + np.exp(-k*(x-x0)))
    
    return (y)

def egg_hatching_probability(df, T_base, T_opt, T_max, N_points=40):
    
    GDD_max = np.amax(df["GDD_eclosion"])

    GDD_slices = np.linspace(0, GDD_max, N_points)

    ninfas = []

    for item in GDD_slices:

        ninfas.append(df[df["GDD_eclosion"] < item]["Numero ninfas"].sum())

    ninfas = ninfas / ninfas[-1]

    p0 = [np.median(GDD_slices), -0.01] # this is an mandatory initial guess

    popt, pcov = curve_fit(sigmoid, GDD_slices, ninfas, p0, method='dogbox')
    
    xrange = np.linspace(min(GDD_slices), max(GDD_slices), 10000)

    plt.figure(figsize=(12, 8))

    plt.scatter(GDD_slices, ninfas, s=250, facecolor='w', color='k', label="Experimental data")
    plt.plot(xrange, sigmoid(xrange, *popt), color="r", lw=3, 
            label=r"$\hat{F}(GDD)=\frac{1}{1+\exp(-%.4f\cdot(GDD-%.2f))}$" % (popt[1], popt[0]))

    plt.xlabel("Accumulated GDD base %sºC" % T_base, fontsize=24, labelpad=20)
    plt.ylabel("Hatching probability", fontsize=24, labelpad=20)

    plt.legend(fontsize=20);

    #plt.savefig("egg_hatching_probability.png", bbox_inches="tight", dpi=300)

    return ninfas, GDD_slices, popt

def optimal_T_base_T_opt_T_max(filenames, locs, T_bases, T_opts, T_maxs, N_points=30):
    
    T_base_CDD = 7
    
    errors = np.zeros((len(T_bases), len(T_opts), len(T_maxs)))
    M = np.zeros((len(T_bases), len(T_opts), len(T_maxs), 3))
    
    i = -1
    j = -1
    k = -1
    
    count = 0
    
    N = len(T_bases) * len(T_opts) * len(T_maxs)
    
    for T_base in T_bases:
        
        i += 1
        j = -1
    
        for T_opt in T_opts:
            
            j += 1
            k = -1
            
            for T_max in T_maxs:
            
                k += 1
                count += 1
                
                for filename, loc in zip(filenames, locs):

                    if filename == filenames[0]:

                        df = egg_hatching_GDD_opt_max("Processed_climatic_data/%s" % filename, T_base, T_opt, 
                                                  T_max, T_base_CDD, loc)
                    else:

                        df_new = egg_hatching_GDD_opt_max("Processed_climatic_data/%s" % filename, T_base,T_opt,
                                              T_max, T_base_CDD, loc)

                        df = df.append(df_new, ignore_index=True)

                GDD_max = np.amax(df["GDD_eclosion"])

                GDD_slices = np.linspace(0, GDD_max, N_points)

                ninfas = []

                for item in GDD_slices:

                    ninfas.append(df[df["GDD_eclosion"] < item]["Numero ninfas"].sum())

                ninfas = ninfas / ninfas[-1]

                p0 = [np.median(GDD_slices), -0.01] # this is a mandatory initial guess

                popt, pcov = curve_fit(sigmoid, GDD_slices, ninfas, p0, method='dogbox')

                error = np.sum(np.abs(ninfas[ninfas < 1] - sigmoid(GDD_slices[ninfas < 1], *popt))) / len(ninfas[ninfas<1]) * 100

                errors[i, j, k] = error
                M[i, j, k] = (T_base, T_opt, T_max)
                
                print("Computing... %.2f %% done." % (count/N*100 ))
        
    return errors, M


# In[ ]:


filenames = ['Ensayo_filenus_BUSTARVIEJO.csv',
       'Ensayo_filenus_IMIDRA.csv', 'Ensayo_filenus_MATAELPINO.csv',
       'Ensayo_filenus_PEDREZUELA.csv']

locs = ["bustarviejo", "encin", "mataelpino", "pedrezuela"]

dT = 0.1

T_bases = np.arange(4, 12 + dT, dT)

T_opts = np.arange(20, 28 + dT, dT)

T_maxs = np.arange(34, 42 + dT, dT)

N_points = 30

errors, M = optimal_T_base_T_opt_T_max(filenames, locs, T_bases, T_opts, T_maxs, N_points)


# In[ ]:


print(M[errors == np.amin(errors)], np.amin(errors))

