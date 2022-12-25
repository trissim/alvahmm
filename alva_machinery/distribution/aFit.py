#!/usr/bin/env python
# coding: utf-8

# # Home-made machinery for distribution fitting including:
# ## A. Likelihood-fitting algorithm
# ### 1. Power_Law
# ### 2. Yule_Simon
# ### 3. Ewens_Sampling

# In[1]:


'''
author: Alvason Zhenhua Li
date:   01/21/2016

Home-made machinery
'''
### open_package +++
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import time
import os
import datetime
### open_package ---
### local_package +++
import alva_machinery.tool.aBox as alva_tool
iWatcher = alva_tool.iWatcher
### local_package ---
###
if __name__ == '__main__':
    previous_running_time = datetime.datetime.now()
    print ('Previous running time is {:}'.format(previous_running_time))


# In[2]:


'''
author: Alvason Zhenhua Li
date:   03/03/2016

Home-made machinery for Power-Law distribution fitting by Likelihood estimate
'''

class AlvaFit(object):
    def __init__(cell, dataX, minX = None, maxX = None, fit_method = None, **kwargs):
        dataX = dataX[dataX > 0]
        # dataX dtype = 'float' is for mpmath because 'int' will raise TypeError("cannot create mpf from " + repr(x)) 
        dataX = np.asarray(dataX, dtype = float)
        if minX is None:
            minX = min(dataX)       
        if maxX is None:
            maxX = max(dataX)
        dataX = dataX[dataX >= minX]
        dataX = dataX[dataX <= maxX]
        cell.dataX = dataX
        cell.minX = minX
        cell.maxX = maxX
        if fit_method is None:
            fit_method = 'Likelihood'
        cell.fit_method = fit_method
        cell.support_distribution = {'Power_Law': PowerLaw,
                                     'Yule_Simon': YuleSimon,
                                     'Ewens_Sampling': EwensSampling}
    # lively loading the additional functional_object (attribute, including Class-object, Function-object)    
    def __getattr__(cell, name):
        if name in cell.support_distribution.keys():
            additional_object = cell.support_distribution[name]
            setattr(cell, name
                    , additional_object(dataX = cell.dataX
                                        , minX = cell.minX
                                        , maxX = cell.maxX
                                        , fit_method = cell.fit_method))
            return getattr(cell, name)
        else:
            raise AttributeError(name)
            
class AlvaDistribution(object):
    def __init__(cell, dataX, minX = None, maxX = None, fit_method = None, parameter = None, **kwargs):
        dataX = dataX[dataX > 0]
        # dataX dtype = 'float' is for mpmath because 'int' will raise TypeError("cannot create mpf from " + repr(x)) 
        dataX = np.asarray(dataX, dtype = float)
        if minX is None:
            minX = min(dataX)       
        if maxX is None:
            maxX = max(dataX)
        dataX = dataX[dataX >= minX]
        dataX = dataX[dataX <= maxX]
        cell.dataX = dataX
        cell.minX = minX
        cell.maxX = maxX
        if fit_method is None:
            fit_method = 'Likelihood'
        cell.fit_method = fit_method
        if parameter is None:
            cell.parameter = cell._initial_parameter()
        else:
            cell._set_parameter(parameter)
        
    def fit(cell):
        if cell.current_distribution == 'EwensSampling':
            EwensPMF = cell._probability_mass_function(cell.dataX, cell.minX)
            cell.EwensPMFxx = EwensPMF[0]
            cell.EwensPMFyy = EwensPMF[1]
            optimized_meter = EwensPMF[2]
            # logLikelihood
            # firstly, for accurate AIC, need to restore to original-size from unique-size
            ePMF = AlvaPMF(cell.dataX, minX = cell.minX, normalization = False)
            all_data = cell.EwensPMFyy * ePMF[1][0:len(cell.EwensPMFyy)]
            log_pmf = np.log(all_data)
            logLike = np.sum(log_pmf)
            cell.max_logLikelihood = logLike
            # update parameter
            cell._set_parameter(optimized_meter) 
            return (optimized_meter)
        else:
            if cell.fit_method == 'Likelihood':
                def fit_function(alpha_meter):
                    return cell.logLikelihood(alpha_meter)
            # search the maximum (minimum for negative values)
            from scipy import optimize
            optimizing = optimize.fmin(lambda alpha_meter: -fit_function(alpha_meter)
                                       , cell._initial_parameter(), full_output = 1, disp = False)

            optimized_meter = optimizing[0]
            negative_max_logLike = optimizing[1]
            cell.max_logLikelihood = -negative_max_logLike
        # update parameter
        cell._set_parameter(optimized_meter) 
        return (optimized_meter)

    def logLikelihood(cell, alpha_meter):
        cell._set_parameter(alpha_meter)
        log_pmf = np.log(cell.pmf(cell.dataX, cell.minX)[1])
        logL = np.sum(log_pmf)
        return (logL)

    # distribution of probability_mass_function
    def pmf(cell, dataX = None, minX = None, uniqueX = False):
        if dataX is None:
            dataX = cell.dataX
        if minX is None:
            minX = cell.minX
        if cell.current_distribution == 'EwensSampling':
            xx = cell.EwensPMFxx
            probability = cell.EwensPMFyy
        else:
            xx = dataX
            probability = cell._probability_mass_function(xx, minX)
            # unique dataX is good for one-to-one plotting only (not for Likelihood)
            if uniqueX is True:
                xx = np.unique(dataX)
                probability = cell._probability_mass_function(xx, minX)
        return (xx, probability)

    # cumulative_distribution_function
    def cdf(cell, dataX = None, minX = None, uniqueX = False):
        if dataX is None:
            dataX = cell.dataX
        if minX is None:
            minX = cell.minX
        xx = dataX
        probability = cell._cumulative_distribution_function(xx, minX)
        # unique dataX is good for one-to-one plotting only (not for Likelihood)
        if uniqueX is True:
            xx = np.unique(dataX)
            probability = cell._cumulative_distribution_function(xx, minX)
        return (xx, probability)
    
    def random_integer_simulator(cell, total_dataX = None, alpha = None, minX = None):
        # since the y = CDF(x) is a simple increasing function (one unique y to one unique x)
        # while y = PDF(x) will not be a one-to-one function (different x having the same y)
        # so that, CDF is a general way to do random simulator
        if total_dataX is None:
            total_dataX = len(cell.dataX)
        if alpha is None:
            alpha = cell.alpha
        if minX is None:
            minX = cell.minX
        randomSea = np.random.uniform(size = total_dataX)
        ##
        def cdf(x):
            probabilityF_x = cell._cumulative_distribution_function(dataX = x, minX = minX, alpha = alpha)
            return probabilityF_x
        ##
        from scipy import optimize
        xx = []
        for y in randomSea:
            ## by solving y = CDF(x), it is able to get the x (in other way, x = inverse_CDF(y))
            solving = optimize.root(lambda x: y - cdf(x) , minX)
            xx.append(solving.x[0])
        xx = np.asarray(xx, dtype = int)
        return xx

    def information_criterion(cell, logLike = None, total_parameter = None, total_sample = None):
        if logLike is None:
            logLike = cell.max_logLikelihood
        if total_parameter is None:
            total_parameter = len(cell.parameter)
        if total_sample is None:
            total_sample = len(np.unique(cell.dataX))
        AIC = -2 * logLike + (2 * total_parameter)
        BIC = -2 * logLike + total_parameter * np.log(total_sample)
        return np.array([AIC, BIC])

#################################
class PowerLaw(AlvaDistribution):
    def __init__(cell, dataX, **kwargs):
        AlvaDistribution.__init__(cell, dataX, **kwargs)
        cell.current_distribution = 'PowerLaw'
    
    def _probability_mass_function(cell, dataX = None, minX = None, alpha = None):
        if dataX is None:
            dataX = cell.dataX
        if minX is None:
            minX = cell.minX
        if alpha is None:
            alpha = cell.alpha
        from scipy.special import zeta
        constantN = 1.0 / zeta(cell.alpha, minX)
        xPower = dataX**(-cell.alpha)
        c_xPower = constantN * xPower 
        return c_xPower 

    def _cumulative_distribution_function(cell, dataX = None, minX = None, alpha = None):
        if dataX is None:
            dataX = cell.dataX
        if minX is None:
            minX = cell.minX
        if alpha is None:
            alpha = cell.alpha
        from scipy.special import zeta
        power_cdf = 1 - zeta(alpha, dataX) / zeta(alpha, minX)
        return power_cdf
        
    # distributions with alpha <= 1 are not valid (non-normalizable)
    def _valid_parameter_range(cell):
        return (cell.alpha > 1)
    
    def _initial_parameter(cell):
        # using the exact value of continuous-case as the initial guessing-value of discrete-case fitting
        # cell.alpha = 1 + (len(cell.dataX) / np.sum(np.log(cell.dataX / (cell.minX))))
        cell.alpha = 2.0
        cell.parameter = np.array([cell.alpha])
        return (cell.parameter)

    def _set_parameter(cell, parameter):
        # if parameter is a scalar not array
        if isinstance(parameter, (int, float)):
            parameter = [parameter]
        # for converting numpy-scalar (0-dimensional array()) to 0-dimensional array([]) 
        parameter = np.atleast_1d(parameter)
        parameter = np.asarray(parameter, dtype = float)
        cell.parameter = parameter
        cell.alpha = cell.parameter[0]
        return(cell.parameter)

##################################
class YuleSimon(AlvaDistribution):
    def __init__(cell, dataX, **kwargs):
        AlvaDistribution.__init__(cell, dataX, **kwargs)
        cell.current_distribution = 'YuleSimon'
    def Modify_probability_mass_function(cell, dataX = None, minX = None):
        if dataX is None:
            dataX = cell.dataX
        if minX is None:
            minX = cell.minX
        # from scipy.special import gamma
        # mpmath is more numerical deeper than scipy
        from mpmath import beta
        # if minX > 1, then self-recursively
        if minX > 1:
            factor_minX = (minX - 1)* beta((minX - 1), cell.alpha) 
        else:
            factor_minX = 1.0
        constantN = (cell.alpha - 1) / factor_minX  
        total_level = len(dataX)
        xYule = np.zeros(total_level)
        for xn in range(total_level):
            xYule[xn] = beta(dataX[xn], cell.alpha) 
        c_xYule = constantN * xYule  
        c_xYule = np.asarray(c_xYule, dtype = float)
        return (c_xYule)    

    def _probability_mass_function(cell, dataX = None, minX = None):
        if dataX is None:
            dataX = cell.dataX
        if minX is None:
            minX = cell.minX
        # mpmath is more numerical deeper than scipy
        from mpmath import beta
        # if minX > 1, then self-recursively
        if minX > 1:
            factor_minX = (minX - 1) * beta((minX - 1), cell.alpha + 1) 
        else:
            factor_minX = 1.0
        constantN = cell.alpha / factor_minX  
        total_level = len(dataX)
        xYule = np.zeros(total_level)
        for xn in range(total_level):
            xYule[xn] = beta(dataX[xn], cell.alpha + 1)
        c_xYule = constantN * xYule 
        c_xYule = np.asarray(c_xYule, dtype = float)
        return (c_xYule) 

    def _cumulative_distribution_function(cell, dataX = None, minX = None, alpha = None):
        if dataX is None:
            dataX = cell.dataX
        if minX is None:
            minX = cell.minX
        # mpmath is numerically deeper than scipy
        from mpmath import beta
        total_level = len(dataX)
        xYule_cdf = np.zeros(total_level)
        for xn in range(total_level):
            xYule_cdf[xn] = 1 - dataX[xn] * beta(dataX[xn], cell.alpha + 1)
        xYule_cdf = np.asarray(xYule_cdf, dtype = float)
        return (xYule_cdf) 

    def Gamma_probability_mass_function(cell, dataX = None, minX = None):
        if dataX is None:
            dataX = cell.dataX
        if minX is None:
            minX = cell.minX
        # mpmath is numerically deeper than scipy
        from scipy.special import gamma
        total_level = len(dataX)
        xYule = np.zeros(total_level)
        for xn in range(total_level):
            xYule[xn] = gamma(dataX[xn]) / gamma(dataX[xn] + cell.alpha + 1.0)   
        constantNorm = cell.alpha * gamma(cell.alpha + 1.0)
        c_xYule = constantNorm * xYule
        c_xYule = np.asarray(c_xYule, dtype = float)
        return (c_xYule)        

    def Alva_probability_mass_function(cell, dataX = None, minX = None):
        if dataX is None:
            dataX = cell.dataX
        if minX is None:
            minX = cell.minX
        ###
        # Yule–Simon distribution fitting
        # Euler Beta integral (Beta function)
        def BetaIntegral(n, b):
            min_T = float(0)
            max_T = float(1)
            totalPoint_T = 10**4
            spacing_T = np.linspace(min_T, max_T, num = totalPoint_T, retstep = True)
            gT = spacing_T[0]
            dt = spacing_T[1]
            outArea = np.sum(gT[:]**(n - 1) * (1 - gT[:])**(b - 1))*dt  
            return (outArea)
        # verify the exact value of Beta(5, 4) = 1.0/280
        # print ('my function', BetaIntegral(5, 4))
        # print ('exact value', 1.0/280)        
        ###
        total_level = len(dataX)
        g_Yule = np.zeros(total_level)
        for xn in range(total_level):
            g_Yule[xn] = cell.alpha * BetaIntegral(dataX[xn], cell.alpha + 1)
        return (g_Yule)
        
    # distributions with alpha <= 1 are not valid (non-normalizable)
    def _valid_parameter_range(cell):
        return (cell.alpha > 0)
    
    def _initial_parameter(cell):
        cell.alpha = 2.0
        cell.parameter = np.array([cell.alpha])
        return (cell.parameter)

    def _set_parameter(cell, parameter):
        # if parameter is a scalar not array
        if isinstance(parameter, (int, float)):
            parameter = [parameter]
        # for converting numpy-scalar (0-dimensional array()) to 0-dimensional array([]) 
        parameter = np.atleast_1d(parameter)
        parameter = np.asarray(parameter, dtype = float)
        cell.parameter = parameter
        cell.alpha = cell.parameter[0]
        return(cell.parameter) 
    
######################################
class EwensSampling(AlvaDistribution):
    def __init__(cell, dataX, **kwargs):
        AlvaDistribution.__init__(cell, dataX, **kwargs)
        cell.current_distribution = 'EwensSampling'
    # statistical testing for non-Darwinian theory  
    def _probability_mass_function(cell, dataX = None, minX = None):
        if dataX is None:
            dataX = cell.dataX
        if minX is None:
            minX = cell.minX
        total_level = len(np.unique(dataX))
        #print ('total_level', total_level)
        # n (total-unit of data) and s (a whole-list of all-possible-team [team-size from 1-unit to n-units])
        n = int(np.sum(dataX)) # total basic-units of data = total cell
        s = np.arange(1, n + 1) # a whole-list of all-possible-team for n-units [team-size from 1-unit to n-units]
        c = np.zeros(len(s)) # a whole-list for holding the count of each-possible-team (partition vector)
        # given partition vector c
        aPMF = AlvaPMF(dataX, minX = cell.minX, normalization = False, empty_leveler_filter = False)
        #print ('aPMF=', aPMF[1])
        c[0:len(aPMF[0])] = aPMF[1] 
        k = int(np.sum(c)) # total-existed-team of n-units = total clone (cluster) 
        print ('n, k, sc = n, length(dataX) = k', n, k, np.sum(s*c), len(dataX))
        ###
        def theta_MaxLike(n, k):
            # Given (n, k), Ewens Sampling assigns probability as:
            def eLogLikelihood(theta, n, k):
                eCore = np.zeros(n)
                for i in range(n):
                    eCore[i] = 1.0 / (theta + i)
                eLL = (k / theta) - np.sum(eCore)
                return eLL
            from scipy import optimize 
            optimizing = optimize.root(eLogLikelihood, 1.0/10**9, args = (n, k))
            theta = optimizing.x[0]
            return (theta)
        ###
        ### Pr(a most likely histogram of k)
#         # mpmath is numerically deeper than scipy
#         from mpmath import factorial, nprod, power
#         # Given partition vector c, Ewens Sampling assigns probability as:
#         pEwens = []
#         #for kn in range(1, k + 1):
#         for sj in np.asarray(aPMF[0], dtype = int):
#             n = int(np.sum(dataX))
#             s = np.arange(1, n + 1)
#             c = np.zeros(len(s)) # a whole-list for holding the count of each-possible-team (partition vector)
#             #c[0] = n - sj * (n / sj)
#             c[int(sj-1)] = n / (2*sj)
#             c[0:len(aPMF[0])] = aPMF[1]
#             n = int(np.sum(s*c))
#             k = int(np.sum(c))
#             #theta = theta_MaxLike(n, k)
#             constantN = factorial(n) / nprod(lambda x: theta + x, [0, n - 1])
#             core = nprod(lambda j: power(float(theta) / j, c[int(j-1)]) / factorial(c[int(j-1)]), [1, n], nsum = True)
#             prob = constantN * core
#             print ('sj, cs=n , k, theta, prob', sj, np.sum(s*c), k, theta, prob)
#             pEwens.append(prob) 
        ### Ep(an expectation of the size of each team under a most likely histogram of k)
        # mpmath is numerically deeper than scipy
        from mpmath import factorial, nprod, power
        theta = theta_MaxLike(n, k)  
        existed_team_size = np.unique(dataX) # reducing data for fast result
        if len(existed_team_size) > 100:
            pmfXX =  existed_team_size[0:100]
        else:
            pmfXX =  existed_team_size
        pEwens = []
        for sj in iWatcher(pmfXX):
            partA = theta * nprod(lambda x: theta + x, [0, n - sj - 1]) * factorial(n)
            partB = k * sj * nprod(lambda x: theta + x, [0, n - 1]) * factorial(n - sj)
            prob = partA / partB 
            #print ('sj, theta, prob', sj, theta, prob)
            pEwens.append(prob)  
        c_pEwens = np.asarray(pEwens, dtype = float) 
        return (pmfXX, c_pEwens, theta)          

    def Stirling_number_1st(cell, n, m):
        """
        Stirling number of the first kind.

        S denotes the Stirling number, (x)_n falling factorial, then
          (x)_n = \sum_{i=0}^{n} s(n, i) * x**i
        and s satisfies the recurrence relation:
          s(n, m) = s(n-1, m-1) - (n-1)*s(n-1, m)
        """
        if m == 0 and n != 0:
            return 0
        elif m == n:
            return 1
        else:
            # recursively calculate by s(n, m) = s(n-1, m-1) - (n-1)*S(n-1, m)
            slist = (1,) # S(1, 1) = 1
            for i in range(1, n):
                l, u = max(1, m - n + i + 1), min(i + 2, m + 1)
                if l == 1 and len(slist) < n - l:
                    nlist = [-i * slist[0]]
                else:
                    nlist = []
                for j in range(len(slist) - 1):
                    nlist.append(slist[j] - i * slist[j + 1])
                if len(slist) <= u - l:
                    nlist.append(slist[-1])
                slist = tuple(nlist)
            return abs(slist[0])
        
    # distributions with alpha <= 1 are not valid (non-normalizable)
    def _valid_parameter_range(cell):
        return (cell.alpha > 0)
    
    def _initial_parameter(cell):
        cell.alpha = 2
        cell.parameter = cell.alpha
        return (cell.parameter)

    def _set_parameter(cell, parameter):
        # if parameter is a scalar not array
        if isinstance(parameter, (int, float)):
            parameter = [parameter]
        # for converting numpy-scalar (0-dimensional array()) to 0-dimensional array([]) 
        parameter = np.atleast_1d(parameter)
        parameter = np.asarray(parameter, dtype = float)
        cell.parameter = parameter
        cell.alpha = cell.parameter[0]
        return(cell.parameter)
    
#     def remembering(func):
#         S = {}
#         def wrappingfunction(*args):
#             if args not in S:
#                 S[args] = func(*args)
#             return S[args]
#         return wrappingfunction

#     @remembering
#     def Stirling_number_1st(cell, n, k):
#         if k == 0 and n != 0:
#             return 0
#         elif k == n:
#             return 1
#         else:
#             return (n-1)*cell.Stirling_number_1st(n-1, k) + cell.Stirling_number_1st(n-1, k-1)
######################################
# def remembering(func):
#     S = {}
#     def wrappingfunction(*args):
#         if args not in S:
#             S[args] = func(*args)
#         return S[args]
#     return wrappingfunction

# @remembering
# def Stirling_number_1st(n, k):
#     if k == 0 and n != 0:
#         return 0
#     elif k == n:
#         return 1
#     else:
#         return (n-1)*Stirling_number_1st(n-1, k) + Stirling_number_1st(n-1, k-1)

def power_law_core(dataX, alpha, minX):
    from scipy.special import zeta
    constantN = 1.0 / zeta(alpha, minX)
    xPower = dataX**(-alpha)
    c_xPower = constantN * xPower 
    return (c_xPower) 

def yule_simon_beta(dataX, alpha, minX):
    # dataX dtype = 'float' is for mpmath because 'int' will raise TypeError("cannot create mpf from " + repr(x)) 
    dataX = np.asarray(dataX, dtype = float)
    from mpmath import beta
    # if minX > 1, then self-recursively
    if minX > 1:
        factor_minX = (minX - 1) * beta((minX - 1), alpha + 1) 
    else:
        factor_minX = 1.0
    constantN = alpha / factor_minX 
    total_level = len(dataX)
    xYule = np.zeros(total_level)
    for xn in range(total_level):
        xYule[xn] = beta(dataX[xn], alpha + 1)
    yuleSimon = constantN * xYule 
    yuleSimon = np.asarray(yuleSimon, dtype = float)
    return (yuleSimon)

### 
def productA(xx):
    # if xx is a scalar not array
    if isinstance(xx, (int, float)):
        xx = [xx]
    # for converting numpy-scalar (0-dimensional array()) to 0-dimensional array([]) 
    xx = np.atleast_1d(xx)
    xx = np.asarray(xx, dtype = int)
    total_point = len(xx)
    # set 0! = 1
    productX = np.zeros(total_point) + 1
    for j in range(total_point):
        for k in range(1, xx[j] + 1):        
            productX[j] = productX[j]*k
    return productX

def productMP(xx):
    # if xx is a scalar not array
    if isinstance(xx, (int, float)):
        xx = [xx]
    # dtype = 'float' is for mpmath because 'int' will raise TypeError("cannot create mpf from " + repr(x)) 
    xx = np.asarray(xx, dtype = float)
    total_point = len(xx)
    from mpmath import gamma
    productX = []
    for xn in range(total_point):
        productX.append(gamma(xx[xn] + 1))
    productX = np.asarray(productX)
    return productX

def random_integer_rPower_cCase(r, alpha, minX):
    # valid for the continuous-case (closed form existed)
    rPower = ((1 - r) / minX) ** (1.0/(1 - alpha)) 
    rPower = np.floor(rPower)
    rPower = rPower.astype(int)
    return (rPower)
   
#########################################################
'''
author: Alvason Zhenhua Li
date:   07/16/2015

Home-made machinery for correlation-coefficient 
'''
# coefficient of determination --- R2 (for a linear fitness)
def AlvaLinearFit(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    meanRaw = y.sum(axis = 0) / len(y)
    variance_raw = np.sum((y - meanRaw)**2)  
    # linear fitting
    linearFit = np.polyfit(x, y, 1)
    slopeFit = linearFit[0]
    constantFit = linearFit[1]
    yFit = slopeFit*x + linearFit[1]
    variance_fit = np.sum((y - yFit)**2)
    R2 = 1 - variance_fit/variance_raw
    return (slopeFit, constantFit, R2)


'''
author: Alvason Zhenhua Li
date:   04/16/2015

Home-made machinery for sorting a list from min-max
Home-made machinery for leveling a list by using min-max way
'''
# min-max sorting
def minMaxA(data):
    totalDataPoint = np.size(data)
    minMaxListing = np.zeros(totalDataPoint)   
    for i in range(totalDataPoint):
        # searching the minimum in current array
        jj = 0 
        minMaxListing[i] = data[jj] # suppose the 1st element [0] of current data-list is the minimum
        for j in range(totalDataPoint - i):
            if data[j] < minMaxListing[i]: 
                minMaxListing[i] = data[j]
                jj = j # recording the position of selected element
        # reducing the size of searching zone (removing the minmum from current array)
        data = np.delete(data, jj)
    return (minMaxListing)

# leveling by using numpy way
def AlvaPDF(dataX, minX = None, maxX = None, total_level = None, normalization = True, empty_leveler_filter = True):
    dataX = np.asarray(dataX)
    if minX is None:
        minX = min(dataX)       
    if maxX is None:
        maxX = max(dataX)
    if total_level is None:
        leveler = np.linspace(minX, maxX, num = 10 + 1)[1:]
        total_level = len(leveler)
    else:
        leveler = np.linspace(minX, maxX, num = total_level + 1)[1:]
    leveleee = np.zeros([total_level])
    for i in range(total_level):
        total_under = np.sum([dataX[:] <= leveler[i]]) 
        leveleee[i] = total_under - np.sum(leveleee[0:i])
    if normalization:
        leveleee = leveleee / np.sum(leveleee)
    # associating (leveler_n, levelee_n)...it is important for the next filter-zero step
    PDF_distribution = np.array([leveler, leveleee]).T   
    # filter out empty-leveler (nothing inside the leveler)...it is important for the future log-step
    if empty_leveler_filter:
        PDF_distribution_update = PDF_distribution[PDF_distribution[:, 1] > 0.0].T
    else: PDF_distribution_update = PDF_distribution.T
    return (PDF_distribution_update)
# # leveling by using min-max way
# def AlvaPDF(dataX, total_level, normalization = True, empty_leveler_filter = True):
#     totalDataPoint = np.size(data)
#     minMaxListing = minMaxA(data)
#     # searching minimum and maximum values
#     minValue = minMaxListing[0]
#     maxValue = minMaxListing[-1]
#     spacingValue = np.linspace(minValue, maxValue, num = totalLevel + 1, retstep = True)        
#     leveler = np.delete(spacingValue[0], 0)
#     # catogerizing the level set
#     # initialize the levelspace by a 'null' space
#     levelSpace = np.zeros([2])
#     levelee = np.zeros([totalLevel])
#     jj = 0 # counting the checked number
#     for i in range(totalLevel): 
#         n = 0 # counting the number in each level
#         for j in range(jj, totalDataPoint):
#             if minMaxListing[j] <= gLevel[i]: 
#                 levelSpace = np.vstack((levelSpace, [i, minMaxListing[j]]))
#                 n = n + 1
#         levelee[i] = n
#         jj = jj + n
#     # delete the inital 'null' space
#     levelSpace = np.delete(levelSpace, 0, 0) 
#     if normalization == True:
#         levelee = levelee / np.sum(minMaxA(levelee))
#     # associating (leveler_n, levelee_n)...it is important for the next filter-zero step
#     PDF_distribution = np.array([leveler, leveleee]).T   
#     # filter out empty-leveler (nothing inside the leveler)...it is important for the future log-step
#     if empty_leveler_filter:
#         PDF_distribution_update = PDF_distribution[PDF_distribution[:, 1] > 0.0].T
#     else: PDF_distribution_update = PDF_distribution.T
#     return (PDF_distribution_update)

'''
author: Alvason Zhenhua Li
date:   02/14/2016

Home-made machinery for probability mass function 
'''
def AlvaPMF(dataX, minX = None, maxX = None, total_level = None, normalization = True, empty_leveler_filter = True):
    dataX = np.asarray(dataX)
    # a integering-data step will secure the next leveling-data step
    dataX = np.int64(dataX)
    # filter out negative and zero data
    dataX = dataX[dataX > 0]
    if minX is None:
        minX = min(dataX)       
    if maxX is None:
        maxX = max(dataX)
    if total_level is None:
        leveler = np.arange(int(minX), int(maxX) + 1) 
        total_level = len(leveler)
    else:
        leveler = np.arange(int(minX), int(maxX) + 1, (maxX - minX) / total_level) 
    leveleee = np.zeros([total_level])
    # sorting data into level (leveling-data step)
    for i in range(total_level):
        leveleee[i] = np.sum([dataX[:] == leveler[i]]) 
    if normalization:
        leveleee = leveleee / np.sum(leveleee)
    # associating (leveler_n, levelee_n)...it is important for the next filter-zero step
    PMF_distribution = np.array([leveler, leveleee]).T   
    # filter out empty-leveler (nothing inside the leveler)...it is important for the future log-step
    if empty_leveler_filter:
        PMF_distribution_update = PMF_distribution[PMF_distribution[:, 1] > 0.0].T
    else: PMF_distribution_update = PMF_distribution.T
    return (PMF_distribution_update)
####################################


# In[ ]:




